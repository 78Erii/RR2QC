import torch
from torch import nn
from transformers import AutoModel, BertConfig, BertForMaskedLM
from torch.nn import CrossEntropyLoss


class MoCoTemplate(nn.Module):
    """From https://github.com/facebookresearch/moco/blob/master/moco/builder.py"""

    def __init__(self, d_rep=128, K=61440, m=0.999, temperature=0.07, encoder_params={}):  # 61440 = 2^12 * 3 * 5
        """
        d_rep: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        target_input: softmax temperature (default: 0.07)
        """
        super().__init__()
        self.config = dict(**{"moco_num_keys": K, "moco_momentum": m, "moco_temperature": temperature},
                           **encoder_params)
        self.K = K  # 队列长度，放着一堆等待计算的keys
        self.m = m  # key编码器的更新动量
        self.temperature = temperature  # 缩放温度

        self.encoder_q = self.make_encoder(**encoder_params)
        self.encoder_k = self.make_encoder(**encoder_params)
        self.cls = self.make_mlm_head(**encoder_params)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.register_buffer("queue", torch.randn(d_rep, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("labels_queue", torch.zeros((encoder_params['config'].num_labels, K), dtype=torch.long))

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.encoder_q.load_state_dict(checkpoint['encoder_q'])
        self.encoder_k.load_state_dict(checkpoint['encoder_k'])
        self.cls.load_state_dict(checkpoint['cls'])
        self.queue = checkpoint['queue']
        self.queue_ptr = checkpoint['queue_ptr']
        self.labels_queue = checkpoint['labels_queue']
        print("Model loaded successfully.")

    def save_model(self, checkpoint_path):
        checkpoint = {
            'encoder_q': self.encoder_q.state_dict(),
            'encoder_k': self.encoder_k.state_dict(),
            'cls': self.cls.state_dict(),
            'queue': self.queue,
            'queue_ptr': self.queue_ptr,
            'labels_queue': self.labels_queue,
        }
        torch.save(checkpoint, checkpoint_path)
        print("Model saved successfully.")

    def make_encoder(self, **kwargs):
        raise NotImplementedError()

    def make_mlm_head(self, **kwargs):
        raise NotImplementedError()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):

        batch_size = keys.shape[0]

        # ptr = int(self.queue_ptr)
        ptr = int(self.queue_ptr.item())  # 指针
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        # 拼接了[0:ptr],Keys转置,[ptr+batch_size:-1]这三段，Keys转置替代了[ptr:ptr+batch_size]这段
        self.queue = torch.cat([self.queue[:, :ptr], keys.T, self.queue[:, ptr + batch_size:]],
                               dim=1).detach()  # detach()将新的队列分配给 self.queue，避免梯度传递到原始的队列
        # 同上
        self.labels_queue = torch.cat(
            [self.labels_queue[:, :ptr], labels.T, self.labels_queue[:, ptr + batch_size:]],
            dim=1).detach()
        ptr = (ptr + batch_size) % self.K  # move pointer 移动指针

        self.queue_ptr[0] = ptr  # queue_ptr： tensor([0], device='cuda:0')

    def embed_x(self, img, lens):
        return self.encoder_q(img, lens)

    def forward(self, target_input, labels, mlm_input, mlm_labels, attention_mask):
        """
        Input:
            target_input: a batch of query images
            mlm_input: a batch of masked images
        Output:
            l_pos: the similarity of target question and masked question in current batch
            l_neg: the similarity of target question and queue in whole queue
            labels: the multi-hot vectors of leaf label groups in current batch
            labels_queue: the multi-hot vectors of leaf label groups in whole queue
            queue_ptr[0]: the current ptr
            loss_mlm: loss of mask language modeling
        """
        # project_out, mlm_out
        t_contrastive, t_representation = self.encoder_q(target_input, mlm_input, attention_mask)
        loss_mlm = self.cls(t_representation, mlm_labels)

        t_contrastive = nn.functional.normalize(t_contrastive, dim=1)  # l2-normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k_contrastive, k_representation = self.encoder_k(target_input, mlm_input, attention_mask)  # keys: NxC
            k_contrastive = nn.functional.normalize(k_contrastive, dim=1)  # l2-normalized

        l_pos = torch.einsum("nc,nc->n", *[t_contrastive, k_contrastive]).unsqueeze(-1)  # compute similarity of target and its masked
        l_neg = torch.einsum("nc,ck->nk", *[t_contrastive, self.queue.detach()])  # compute similarity of target and queue

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_contrastive, labels)

        return l_pos, l_neg, labels, self.labels_queue.T, self.queue_ptr[0], loss_mlm


class RCPTEncoder(nn.Module):
    def __init__(
            self,
            bert_type,
            d_rep=128,
            project=False,
            tokenizer=None,
            **kwargs
    ):
        super().__init__()
        self.config = {k: v for k, v in locals().items() if k != "self"}
        self.encoder = AutoModel.from_pretrained(bert_type, config=kwargs["config"])  # BERT
        if project:
            self.project_layer = nn.Sequential(nn.Linear(self.encoder.config.hidden_size,
                                                         self.encoder.config.hidden_size),
                                               nn.ReLU(),
                                               nn.Linear(self.encoder.config.hidden_size, d_rep))
        # NOTE: We use the default PyTorch intialization, so no need to reset parameters.

    def forward(self, t, mlm, attention_mask, no_project_override=False):
        # x: (batch_size, max_len)
        # out = self.encoder(x).last_hidden_state
        # t_out = self.encoder(t, output_hidden_states=True, attention_mask=attention_mask)[0]
        mlm_out = self.encoder(mlm, output_hidden_states=True, attention_mask=attention_mask)[0]
        if not no_project_override and self.config["project"]:
            # project_out = self.project(t_out)
            project_out = self.project(mlm_out)
            return project_out, mlm_out

        return mlm_out

    def project(self, out):
        # out: (batch_size, max_len, hidden_dim)
        assert self.config["project"]
        # NOTE: This computes a mean pool of the token representations across ALL tokens,
        # including padding from uneven lengths in the batch.
        # (batch_size, RCPTtion_dim)
        return self.project_layer(out.mean(dim=1))


class MLM_head(nn.Module):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__()
        self.config = kwargs['config']
        self.vocab_size = self.config.vocab_size
        self.cls = BertForMaskedLM.from_pretrained(kwargs['bert_type'], config=kwargs['config'], ignore_mismatched_sizes=True).cls

    def resize_cls_vocab(self, new_vocab_size):
        self.vocab_size = new_vocab_size
        new_decoder = nn.Linear(in_features=768, out_features=new_vocab_size, bias=True)
        self.cls.predictions.decoder = new_decoder

    def forward(self, x, labels):
        sequence_output = x
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token  # 只计算非padding位置的损失
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size), labels.view(-1))

        return masked_lm_loss


class RCPTMoCo(MoCoTemplate):
    def __init__(self, bert_type, d_rep=128, project=True, K=8000, m=0.999, target_input=0.07, config=None):
        super().__init__(
            d_rep,
            K,
            m,
            target_input,
            encoder_params=dict(bert_type=bert_type, d_rep=d_rep, project=project, config=config),
        )

    def make_encoder(
            self,
            bert_type,
            d_rep,
            project=True,
            tokenizer=None,
            **kwargs
    ):
        return RCPTEncoder(
            bert_type, project=project, d_rep=d_rep, tokenizer=tokenizer, **kwargs
        )

    def make_mlm_head(self,
                      **kwargs
                      ):
        return MLM_head(**kwargs)

    def forward(self, target_input, labels=None, mlm_input=None, mlm_labels=None, attention_mask=None):
        """
        Input:
            target_input: a batch of target questions, [batch_size, max_len]
            labels: a batch of leaf label multi-hot vectors, [batch_size, num_labels]
            mlm_input: a batch of masked questions, [batch_size, max_len]
            mlm_labels: record the ture token of mlm_input, [batch_size, max_len]
            attention_mask: record which token is padded as 0, [batch_size, max_len]
        Output:
            logits, targets
        """
        return super().forward(target_input, labels=labels, mlm_input=mlm_input, mlm_labels=mlm_labels,
                               attention_mask=attention_mask)
