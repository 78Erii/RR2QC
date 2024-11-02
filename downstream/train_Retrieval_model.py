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
from data.Math_Senior.senior_three_level_label_dict import label_key_index_dict


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
        # question_id, question_content and their leaf labels
        ques_id, content, labels = self.lines[idx][0], self.lines[idx][1], self.lines[idx][2:]
        # encode the question_content into token indices
        tokens = self.tokenizer.encode(content, add_special_tokens=True)[:-1]

        # make multi-hot label vectors
        ques_labels = [0.0 for _ in range(len(label_key_index_dict))]
        for one_label in labels:
            one_idx = label_key_index_dict[one_label]
            ques_labels[one_idx] = 1.0

        ques_input = {'input_ids': tokens[:self.seq_len]}
        ques_input = self.tokenizer.pad(ques_input, max_length=self.seq_len, padding='max_length')
        # model_input['input_ids'][-1] = 0  # for ReBERTa setting
        output = {'ques_id': ques_id,
                  'ques_input': ques_input['input_ids'],
                  'attention_mask': ques_input['attention_mask'],
                  'ques_labels': ques_labels,
                  }

        # transfer data to tensor
        dic_output = {key: value if key == 'ques_id' else torch.tensor(value) for key, value in output.items()}
        return dic_output


def search_local_trained(out_path):
    out_path = out_path / 'Retrieval_trained_model'
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


def Datas_Extend(datas: dict, tokenizer, max_len, k=4):
    random4_label_inputs = random.sample(label_inputs, k=k)
    for one_label_input in random4_label_inputs:
        id = one_label_input.split('\t')[0]
        datas['ques_id'].append(id)
        content = one_label_input.split('\t')[1]
        tokens = tokenizer.encode(content,
                                  add_special_tokens=True,
                                  padding='max_length',
                                  max_length=max_len,
                                  truncation=True,
                                  return_tensors='pt'  # return pytorch sensor
                                  )
        datas['ques_input'] = torch.cat((datas['ques_input'], tokens), dim=0)
        labels = one_label_input.split('\t')[2]
        labels_one_hot = [0.0 if i != label_key_index_dict[labels] else 1.0 for i in
                          range(len(label_key_index_dict))]
        labels_one_hot = torch.tensor(labels_one_hot)
        labels_one_hot = torch.unsqueeze(labels_one_hot, dim=0)
        datas['ques_labels'] = torch.cat((datas['ques_labels'], labels_one_hot), dim=0)
        attention_mask = tokens.ne(0).int()
        datas['attention_mask'] = torch.cat((datas['attention_mask'], attention_mask), dim=0)
    return datas


def create_optimizer_and_scheduler(num_training_steps, learning_rate, model):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,
                                  weight_decay=0.0)

    def linear_decay(current_step):
        return max(0, 1 - current_step / num_training_steps)

    scheduler = LambdaLR(optimizer, lr_lambda=linear_decay)
    return optimizer, scheduler


def set_seed(args):
    random.seed(args.seed)  # python
    np.random.seed(args.seed)  # numpy
    torch.manual_seed(args.seed)  # torch
    torch.cuda.manual_seed(args.seed)  # cuda
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def copy_local_class_center_to_model(model):
    with open(Path(args.data_path) / args.dataset / 'class_center_label_input.pkl', 'rb') as file:
        class_center = pickle.load(file)
    class_center.to(model.device)
    model.class_center.data.copy_(class_center)


def build_model(args):
    if args.local_trained_epoch:
        local_path = out_path / 'Retrieval_trained_model' / 'model{}'.format(args.local_trained_epoch)
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
            config.num_labels = len(label_key_index_dict)
            config.lamda = args.lamda
            model = RetrievalRerankingToQuestionClassification(config=config)
            model.load_local_states(args.base_encoder_path)
            model.bert.resize_token_embeddings(len(tokenizer))
        elif args.base_model_name == 'TextCNN':
            model = TextCNNModel()
            config = BertConfig.from_pretrained(args.base_encoder_path)
            config.num_labels = len(label_key_index_dict)
            base_model = RetrievalRerankingToQuestionClassification(config=config)
            base_model.load_local_states(args.base_encoder_path)
            base_model.bert.resize_token_embeddings(len(tokenizer))
            pretrained_embeddings = base_model.bert.embeddings.word_embeddings.weight
            model.word_embedding = torch.nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        if args.is_use_CCL:
            copy_local_class_center_to_model(model)
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
    out_model_path = out_path / 'Retrieval_trained_model'
    out_model_path.mkdir(exist_ok=True, parents=True)
    out_result_path = out_path / 'Retrieval_result'
    out_result_path.mkdir(exist_ok=True, parents=True)
    args.is_init_CCL = True
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
    if args.is_init_CCL:
        args.batch_size -= 4
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0,
                                   shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)

    args.local_trained_epoch = search_local_trained(out_path)
    print('Already trained {} epochs'.format(args.local_trained_epoch))

    print('Building Retrieval model')
    model = build_model(args)

    if args.is_init_CCL:
        with open(Path(args.data_path) / args.dataset / 'only_label_input.txt', 'r',
                  encoding='utf-8') as f:
            label_inputs = [line.strip() for line in f.readlines() if line != '\n']
    if not args.local_trained_epoch:
        with open(out_result_path / 'Retrieval_result.txt', 'w', encoding='utf-8') as f:
            f.truncate(0)

    # freeze half of encoder layers
    freeze_encoder_layer(args, model)

    total_steps = len(train_data_loader) * args.epochs
    optimizer, scheduler = create_optimizer_and_scheduler(total_steps, args.lr, model)

    if not args.local_trained_epoch:
        with open(out_result_path / 'epoch_max_p1.txt', 'w', encoding='utf-8') as f:
            f.write('train0\t0')
            max_p1 = 0
    else:
        with open(out_result_path / 'epoch_max_p1.txt', 'r', encoding='utf-8') as f:
            line = f.readline()
            max_p1 = float(re.findall('train.*\t(.*)', line)[0])

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
            if args.is_init_CCL:
                train_datas = Datas_Extend(train_datas, tokenizer, args.max_len, k=4)  # train with label_inputs
            train_datas = {key: value if key == 'ques_id' else value.to(args.device) for key, value in
                           train_datas.items()}
            output = model(input_ids=train_datas['ques_input'],
                           labels=train_datas['ques_labels'],
                           attention_mask=train_datas['attention_mask'],
                           use_CCL=args.is_use_CCL
                           )  # 正向传播
            loss = output[0]
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            if args.is_use_CCL:
                loss_parts = output[3]
                total_BCE_loss += loss_parts[0].item()
                total_CCL_loss += loss_parts[1].item()
                train_pbar.set_postfix(loss_all=loss.item(), loss_BCE=loss_parts[0].item(),
                                       loss_CCL=loss_parts[1].item(),
                                       lr=current_lr)
            else:
                train_pbar.set_postfix(loss_all=loss.item(), lr=current_lr)

        avg_train_loss = total_train_loss / len(train_data_loader)  # average loss per epoch
        avg_CCL_loss = total_CCL_loss / len(train_data_loader)
        avg_BCE_loss = total_BCE_loss / len(train_data_loader)

        with open(out_result_path / 'Retrieval_result.txt', 'a', encoding='utf-8') as f:
            f.write('Epoch : {}, Train Loss : {}\t BCE Loss : {} \t CCL Loss : {}'.format(
                epoch_idx + args.local_trained_epoch + 1, avg_train_loss, avg_BCE_loss, avg_CCL_loss))
            f.write('\n')

        # save model states per epoch
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
                               labels=val_datas['ques_labels'],
                               attention_mask=val_datas['attention_mask']
                               )
            confidence_scores = output[1]
            sigmoid = torch.nn.Sigmoid()
            norm_scores = sigmoid(confidence_scores)
            labels = val_datas['ques_labels'].cpu().numpy()
            y_true_all.extend(labels)

            for i in range(len(labels[:])):
                top_k_logits_pred, top_k_indices_pred = torch.topk(norm_scores[i], k=3)

                y_pred = [0 for _ in range(len(label_key_index_dict))]
                y_pred[top_k_indices_pred[0]] = 1
                y_pred_all.append(y_pred)

                y_pred2 = [0 for _ in range(len(label_key_index_dict))]
                for idx2 in top_k_indices_pred[:2]:
                    y_pred2[idx2] = 1
                y_pred2_all.append(y_pred2)

                y_pred3 = [0 for _ in range(len(label_key_index_dict))]
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

        if float(p1) > max_p1:
            max_p1 = p1
            with open(out_result_path / 'epoch_max_p1.txt', 'w', encoding='utf-8') as f:
                f.write('train{}\t{}'.format(epoch_idx + args.local_trained_epoch + 1, max_p1))
            model.save_pretrained(out_result_path / 'best_model')

        print('MicroPr@1 {}'.format(precision_score(y_true_all, y_pred_all, average='micro', zero_division=1)))
        print('MicroPr@2 {}'.format(precision_score(y_true_all, y_pred2_all, average='micro', zero_division=1)))
        print('MicroPr@3 {}'.format(precision_score(y_true_all, y_pred3_all, average='micro', zero_division=1)))
        print('MicroF1@1 {}'.format(f1_score(y_true_all, y_pred_all, average='micro', zero_division=1)))
        print('MacroF1@1 {}'.format(f1_score(y_true_all, y_pred_all, average='macro', zero_division=1)))

        with open(out_result_path / 'Retrieval_result.txt', 'a', encoding='utf-8') as f:
            f.write('Val epoch:{}\tPrecison@1:{}\tPrecison@2:{}\tPrecison@3:{}\tMicro-F1@1:{}\t Macro-F1@1:{}\n'.format(
                epoch_idx + args.local_trained_epoch + 1, p1, p2, p3, micro_f1, macro_f1))
            f.write('-' * 20 + '\n')
