import os
import re
import tqdm
import torch
import random
import pickle
import numpy as np
from transformers import BertTokenizer, BertConfig, RobertaConfig
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, f1_score
# necessary setting
from parameters import parse_args
from pathlib import Path
from model.model_RR2QC import RetrievalRerankingToQuestionClassification
from model.TextCNN import TextCNNModel
# load label dict
from data.Math_Senior.senior_three_level_label_dict import label2metalabel, metalabel_key_index_dict


class RR2QC_Dataset(Dataset):
    def __init__(self, data_path, tokenizer, seq_len, data_lines=None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len  # default 512
        self.data_lines = data_lines
        self.data_path = data_path  # dataset path

        with open(data_path, 'r', encoding='utf-8') as f:
            self.lines = [line[:-1].split('\t') for line in
                          tqdm.tqdm(f, desc='Loading Dataset', total=data_lines)]
            self.data_lines = len(self.lines)

    def __len__(self):
        return self.data_lines

    def __getitem__(self, idx):
        ques_id, content, labels = self.lines[idx][0], self.lines[idx][1], self.lines[idx][2:]

        tokens = self.tokenizer.encode(content, add_special_tokens=True)[:-1]

        # make multi-hot meta-label vector
        ques_meta_labels = [0.0 for _ in range(len(metalabel_key_index_dict))]
        for label in labels:
            meta_labels = label2metalabel[label]
            for meta_label in meta_labels:
                one_idx = metalabel_key_index_dict[meta_label]
                ques_meta_labels[one_idx] = 1.0

        ques_input = {'input_ids': tokens[:self.seq_len]}
        ques_input = self.tokenizer.pad(ques_input, max_length=self.seq_len, padding='max_length')
        # model_input['input_ids'][-1] = 0  # for ReBERTa setting
        output = {'ques_id': ques_id,
                  'ques_input': ques_input['input_ids'],
                  'attention_mask': ques_input['attention_mask'],
                  'ques_meta_labels': ques_meta_labels,
                  }

        dic_output = {key: value if key == 'ques_id' else torch.tensor(value) for key, value in output.items()}
        return dic_output


def search_local_trained(out_path):
    out_path = out_path / 'Reranking_trained_model'
    model_nums = os.listdir(out_path)
    nums = []
    for i in range(len(model_nums)):
        num = int(re.findall(r'model(\d+)', model_nums[i])[0])
        nums.append(num)
    if nums:
        max_nums = max(nums)
    else:
        max_nums = 0
    return max_nums


def print_trainable_params(model: torch.nn.Module) -> None:
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, 'ds_numel'):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print('trainable params: {:d} || all params: {:d} || trainable%: {:.4f}'.format(
        trainable_params, all_param, 100 * trainable_params / all_param))


def create_optimizer_and_scheduler(num_training_steps, learning_rate, model):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,
                                  weight_decay=1e-4)

    def linear_decay(current_step):
        return max(0, 1 - current_step / num_training_steps)

    scheduler = LambdaLR(optimizer, lr_lambda=linear_decay)
    return optimizer, scheduler


def set_seed(args):
    random.seed(args.seed)  # py
    np.random.seed(args.seed)  # numpy
    torch.manual_seed(args.seed)  # torch
    torch.cuda.manual_seed(args.seed)  # cuda
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(args):
    if args.local_trained_epoch:
        local_path = out_path / 'Reranking_trained_model' / 'model{}'.format(args.local_trained_epoch)
        if args.base_model_name == 'BERT' or 'RoBERTa':
            config = BertConfig.from_pretrained(local_path)
            model = RetrievalRerankingToQuestionClassification(config=config)
            model.load_local_states(local_path)
        elif args.base_model_name == 'TextCNN':
            model = TextCNNModel()
            checkpoint = torch.load(local_path)
            model.load_state_dict(checkpoint)
    else:
        if args.base_model_name == 'BERT' or 'RoBERTa':
            config = BertConfig.from_pretrained(args.base_encoder_path)
            config.num_labels = len(metalabel_key_index_dict)
            config.lamda = args.lamda
            model = RetrievalRerankingToQuestionClassification(config=config)
            model.load_local_states(args.base_encoder_path)
            model.bert.resize_token_embeddings(len(tokenizer))
        elif args.base_model_name == 'TextCNN':
            model = TextCNNModel()
            config = BertConfig.from_pretrained(args.base_encoder_path)
            config.num_labels = len(metalabel_key_index_dict)
            base_model = RetrievalRerankingToQuestionClassification(config=config)
            base_model.load_local_states(args.base_encoder_path)
            base_model.bert.resize_token_embeddings(len(tokenizer))
            pretrained_embeddings = base_model.bert.embeddings.word_embeddings.weight
            model.word_embedding = torch.nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
    model.to(args.device)
    return model


def freeze_encoder_layer(args, model):
    if args.base_model_name == 'BERT':
        for layer in model.bert.encoder.layer[:6]:
            for param in layer.parameters():
                param.requires_grad = False
    elif args.base_model_name == 'RoBERTa':
        num_layers = len(model.roberta.encoder.layer)
        for i in range(num_layers // 4 * 2):
            for param in model.roberta.encoder.layer[i].parameters():
                param.requires_grad = False
    print_trainable_params(model)


if __name__ == '__main__':
    args = parse_args()
    out_path = Path(args.out_path) / f'{args.name}-{args.dataset}'
    out_path.mkdir(exist_ok=True, parents=True)
    out_model_path = out_path / 'Reranking_trained_model'
    out_model_path.mkdir(exist_ok=True, parents=True)
    out_result_path = out_path / 'Reranking_result'
    out_result_path.mkdir(exist_ok=True, parents=True)
    args.is_init_CCL = False
    args.is_use_CCL = False

    set_seed(args)

    print('Loading tokenizer')
    tokenizers_path = Path(args.data_path) / args.dataset / 'Custom_tokenizer'
    tokenizer = BertTokenizer.from_pretrained(tokenizers_path)
    print('vocab Size:', len(tokenizer))

    train_data_path = Path(args.data_path) / args.dataset / args.train_set
    val_data_path = Path(args.data_path) / args.dataset / args.val_set
    train_dataset = RR2QC_Dataset(train_data_path, tokenizer, seq_len=args.max_len, data_lines=None)
    val_dataset = RR2QC_Dataset(val_data_path, tokenizer, seq_len=args.max_len, data_lines=None)

    print('Creating Dataloader')
    if args.base_model_name == 'RoBERTa':
        args.batch_size = 16
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0,
                                   shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)

    args.local_trained_epoch = search_local_trained(out_path)
    print('Already trained {} epochs'.format(args.local_trained_epoch))

    print('Building Reranking model')
    model = build_model(args)

    # freeze half of encoder layers
    freeze_encoder_layer(args, model)

    total_steps = len(train_data_loader) * args.epochs
    optimizer, scheduler = create_optimizer_and_scheduler(total_steps, args.lr, model)

    '''train and val'''
    for epoch_idx in range(args.epochs):
        train_data_loader_iterator = iter(train_data_loader)
        train_pbar = tqdm.tqdm(range(len(train_data_loader)), desc=f'Epoch {epoch_idx + args.local_trained_epoch + 1}',
                               unit='batch')

        '''train'''
        model.train()
        total_train_loss = 0.
        total_BCE_loss = 0.
        total_CCL_loss = 0.
        for train_idx in train_pbar:
            train_datas = next(train_data_loader_iterator)
            train_datas = {key: value if key == 'ques_id' else value.to(args.device) for key, value in
                           train_datas.items()}
            output = model(input_ids=train_datas['ques_input'],
                           labels=train_datas['ques_meta_labels'],
                           attention_mask=train_datas['attention_mask'],
                           use_CCL=False
                           )
            loss = output[0]
            total_train_loss += loss.item()
            meta_labels = train_datas['ques_meta_labels'].cpu().numpy()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            train_pbar.set_postfix(loss_all=loss.item(), lr=current_lr)

        avg_train_loss = total_train_loss / len(train_data_loader)  # average loss per epoch

        with open(out_result_path / 'Reranking_result.txt', 'a', encoding='utf-8') as f:
            f.write('Epoch : {}, Train Loss : {}\n'.format(
                epoch_idx + args.local_trained_epoch + 1, avg_train_loss))

        # # save model states per epoch
        model.save_pretrained(
            out_model_path / 'model{num}'.format(num=epoch_idx + args.local_trained_epoch + 1))

        '''val'''
        val_data_loader_iterator = iter(val_data_loader)
        val_pbar = tqdm.tqdm(range(len(val_data_loader)), desc=f'Epoch {epoch_idx + args.local_trained_epoch + 1}',
                             unit='batch')
        model.eval()

        y_true_all = []  # true labels
        y_pred_all = []  # top 1 predicted labels
        y_pred2_all = []  # top 2 predicted labels
        y_pred3_all = []  # top 3 predicted labels
        for val_batch_id in val_pbar:
            val_datas = next(val_data_loader_iterator)
            for key, value in val_datas.items():
                val_datas = {key: value if key == 'ques_id' else value.to(args.device) for key, value in
                             val_datas.items()}
            model.zero_grad()
            with torch.no_grad():
                output = model(input_ids=val_datas['ques_input'],
                               labels=val_datas['ques_meta_labels'],
                               attention_mask=val_datas['attention_mask']
                               )
            confidence_scores = output[1]
            meta_labels = val_datas['ques_meta_labels'].cpu().numpy()
            y_true_all.extend(meta_labels)

            for i in range(len(meta_labels[:])):
                top_k_logits_pred, top_k_indices_pred = torch.topk(confidence_scores[i], k=3)

                y_pred = [0 for _ in range(len(metalabel_key_index_dict))]
                y_pred[top_k_indices_pred[0]] = 1
                y_pred_all.append(y_pred)

                y_pred2 = [0 for _ in range(len(metalabel_key_index_dict))]
                for idx2 in top_k_indices_pred[:2]:
                    y_pred2[idx2] = 1
                y_pred2_all.append(y_pred2)

                y_pred3 = [0 for _ in range(len(metalabel_key_index_dict))]
                for idx3 in top_k_indices_pred:
                    y_pred3[idx3] = 1
                y_pred3_all.append(y_pred3)

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        y_pred2_all = np.array(y_pred2_all)
        y_pred3_all = np.array(y_pred3_all)
        p1 = round(precision_score(y_true_all, y_pred_all, average='micro', zero_division=1), 4)
        p2 = round(precision_score(y_true_all, y_pred2_all, average='micro', zero_division=1), 4)
        p3 = round(precision_score(y_true_all, y_pred3_all, average='micro', zero_division=1), 4)
        micro_f1 = round(f1_score(y_true_all, y_pred_all, average='micro', zero_division=1), 4)
        macro_f1 = round(f1_score(y_true_all, y_pred_all, average='macro', zero_division=1), 4)
        print('MicroPr@1 {}'.format(precision_score(y_true_all, y_pred_all, average='micro', zero_division=1)))
        print('MicroPr@2 {}'.format(precision_score(y_true_all, y_pred2_all, average='micro', zero_division=1)))
        print('MicroPr@3 {}'.format(precision_score(y_true_all, y_pred3_all, average='micro', zero_division=1)))
        print('MicroF1@1 {}'.format(f1_score(y_true_all, y_pred_all, average='micro', zero_division=1)))
        print('MacroF1@1 {}'.format(f1_score(y_true_all, y_pred_all, average='macro', zero_division=1)))

        with open(out_result_path / 'Reranking_result.txt', 'a', encoding='utf-8') as f:
            f.write('Val epoch:{}\tPrecison@1:{}\tPrecison@2:{}\tPrecison@3:{}\tMicro-F1@1:{}\t Macro-F1@1:{}\n'.format(
                epoch_idx + args.local_trained_epoch + 1, p1, p2, p3, micro_f1, macro_f1))
            f.write('-' * 20 + '\n')
