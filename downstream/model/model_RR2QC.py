import os
import torch
from torch import nn
from pathlib import Path
from torch.nn.functional import binary_cross_entropy_with_logits
from safetensors.torch import load_file
from transformers import AutoModel, BertPreTrainedModel, BertModel, RobertaModel
from typing import List, Optional, Tuple, Union
from downstream.loss.loss_CCL import Class_Centering_Learning_loss


class RetrievalRerankingToQuestionClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # self.encoder = AutoModel.from_pretrained(config)
        self.bert = BertModel(config)
        self.lamda = config.lamda
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier4 = nn.Linear(config.hidden_size, self.num_labels)
        self.class_center = nn.Parameter((torch.empty(self.num_labels, config.hidden_size)), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.class_center)

        # Initialize weights and apply final processing
        self.post_init()

    def load_local_states(self, checkpoint_path):
        checkpoint_file = os.listdir(checkpoint_path)
        if 'model.safetensors' in checkpoint_file:
            checkpoint_path = os.path.join(str(checkpoint_path), 'model.safetensors')
            checkpoint = load_file(checkpoint_path)
        elif 'pytorch_model.bin' in checkpoint_file:
            checkpoint_path = os.path.join(str(checkpoint_path), 'pytorch_model.bin')
            checkpoint = load_file(checkpoint_path)
        self.load_state_dict(checkpoint, strict=False)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_CCL: Optional[bool] = False
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        cls_pooled_output = self.dropout(outputs[1])
        logits = self.classifier4(cls_pooled_output)

        loss_BCE = binary_cross_entropy_with_logits(logits, labels, weight=None)

        if use_CCL:
            loss_CCL = Class_Centering_Learning_loss(cls_pooled_output, labels, self.class_center)
            loss_Retrieval = loss_BCE + self.lamda * loss_CCL
            loss_parts = (loss_BCE, loss_CCL)
            outputs = (loss_Retrieval, logits, cls_pooled_output, loss_parts)
        else:
            outputs = (loss_BCE, logits, cls_pooled_output)
        return outputs
