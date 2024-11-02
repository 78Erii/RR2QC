import os
import re
import tqdm
import torch
import pickle
import numpy as np
from transformers import BertTokenizer, BertConfig, RobertaConfig
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, f1_score
# necessary setting
from parameters import parse_args
from pathlib import Path
from model.model_RR2QC import RetrievalRerankingToQuestionClassification
from model.TextCNN import TextCNNModel
# load label dict
from data.Math_Senior.senior_three_level_label_dict import label_key_index_dict

label_index_key_dict = {value: key for key, value in label_key_index_dict.items()}


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


def build_model(args):
    if 'best_model' in os.listdir(out_result_path):
        local_path = Path(out_result_path / 'best_model')
        config = BertConfig.from_pretrained(local_path)
        model = RetrievalRerankingToQuestionClassification(config=config)
        model.load_local_states(local_path)
    elif args.local_trained_epoch:
        local_path = out_path / 'Retrieval_trained_model' / 'model{}'.format(args.local_trained_epoch)
        if args.base_model_name == 'BERT' or 'RoBERTa':
            config = BertConfig.from_pretrained(local_path)
            model = RetrievalRerankingToQuestionClassification(config=config)
            model.load_local_states(local_path)
        elif args.base_model_name == 'TextCNN':
            model = TextCNNModel()
            checkpoint = torch.load(local_path)
            model.load_state_dict(checkpoint)
    model.to(args.device)
    return model


if __name__ == '__main__':
    args = parse_args()
    out_path = Path(args.out_path) / f'{args.name}-{args.dataset}'
    out_model_path = out_path / 'Retrieval_trained_model'
    out_result_path = out_path / 'Retrieval_result'
    out_label_path = out_path / 'label'
    out_label_path.mkdir(exist_ok=True, parents=True)

    print('Loading tokenizer')
    tokenizers_path = Path(args.data_path) / args.dataset / 'Custom_tokenizer'
    tokenizer = BertTokenizer.from_pretrained(tokenizers_path)
    print('vocab Size:', len(tokenizer))

    test_data_path = Path(args.data_path) / args.dataset / args.test_set

    args.is_init_CCL = False
    if args.is_init_CCL:
        test_data_path = Path(args.data_path) / args.dataset / 'only_label_input.txt'
    test_dataset = RR2QC_Dataset(test_data_path, tokenizer, seq_len=args.max_len, data_lines=None)

    print('Creating Dataloader')
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)

    args.local_trained_epoch = search_local_trained(out_path)
    print('Already trained {} epochs'.format(args.local_trained_epoch))

    print('Building Retrieval model')
    model = build_model(args)

    prediction_path = out_result_path / 'prediction_path.txt'
    with open(prediction_path, 'w', encoding='utf-8') as f:
        f.truncate(0)
    True_labels = {}
    Retrieval_labels_with_scores = {}

    class_center = torch.empty((1, 768), dtype=torch.float, device=args.device)
    if args.base_model_name == 'RoBERTa':
        class_center = torch.empty((1, 1024), dtype=torch.float, device=args.device)

    model.eval()

    y_true_all = []  # true labels
    y_pred_all = []  # top 1 predicted labels
    y_pred2_all = []  # top 2 predicted labels
    y_pred3_all = []  # top 3 predicted labels

    '''val'''
    test_data_loader_iterator = iter(test_data_loader)
    test_pbar = tqdm.tqdm(range(len(test_data_loader)), unit='batch')

    '''val'''
    for test_idx in test_pbar:
        test_datas = next(test_data_loader_iterator)
        test_datas = {key: value if key == 'ques_id' else value.to(args.device) for key, value in
                      test_datas.items()}
        ques_id = test_datas['ques_id']
        input_ids = test_datas['ques_input']
        with torch.no_grad():
            output = model(input_ids=test_datas['ques_input'],
                           labels=test_datas['ques_labels'],
                           attention_mask=test_datas['attention_mask'],
                           )
        if args.is_init_CCL:
            class_center = torch.cat((class_center, output[2]), dim=0)

        confidence_scores = output[1]
        sigmoid = torch.nn.Sigmoid()
        norm_scores = sigmoid(confidence_scores)
        labels = test_datas['ques_labels'].cpu().numpy()

        y_true_all.extend(labels)

        for i in range(len(labels[:])):
            k = np.count_nonzero(labels[i] == 1)  # get the num of true labels per question
            true_label_indices = np.argsort(labels[i])[-k:]  # true label indices
            true_label_names = [label_index_key_dict[i] for i in true_label_indices]  # true label name
            True_labels[ques_id[i]] = true_label_names  # save to local

            # get top n pred label indices and logits
            n = len(label_key_index_dict)
            top_n_logits_pred, top_n_indices_pred = torch.topk(norm_scores[i], k=n)

            y_pred = [0 for _ in range(len(label_key_index_dict))]
            y_pred[top_n_indices_pred[0]] = 1
            y_pred_all.append(y_pred)

            y_pred2 = [0 for _ in range(len(label_key_index_dict))]
            for idx2 in top_n_indices_pred[:2]:
                y_pred2[idx2] = 1
            y_pred2_all.append(y_pred2)

            y_pred3 = [0 for _ in range(len(label_key_index_dict))]
            for idx3 in top_n_indices_pred[:3]:
                y_pred3[idx3] = 1
            y_pred3_all.append(y_pred3)

            with torch.no_grad():
                top_n_logits_pred = top_n_logits_pred.cpu().numpy()
                top_n_indices_pred = top_n_indices_pred.cpu().numpy()

            # record predicted result
            with (open(prediction_path, 'a', encoding='utf-8') as f):
                ques_content = tokenizer.decode(input_ids[i], skip_special_tokens=True).replace(' ', '')
                top_n_labelname_pred = [label_index_key_dict[i] for i in top_n_indices_pred]
                top_n_labels_with_scores = [top_n_labelname_pred[i] + '\t' + str('%.3f' % top_n_logits_pred[i]) for
                                            i in
                                            range(n)]

                Retrieval_labels_with_scores[ques_id[i]] = top_n_labels_with_scores  # record label sequence per question

                result = ques_id[i] + '\t' + ques_content + '\n' + 'True: ' + ' '.join(
                    true_label_names) + '\t' + 'Pred@3: ' + '\t'.join(top_n_labels_with_scores[:3]) + '\n'
                f.write(result) # write predicted result to local file

    if args.is_init_CCL:
        class_center_clone = class_center[1:].clone().cpu()
        with open(Path(args.data_path) / args.dataset / 'class_center_label_input.pkl', 'wb') as file:
            pickle.dump(class_center_clone, file)  # get the class_center_vector
    else:
        # save ture label to local
        with open(out_label_path / 'True_labels.pkl', 'wb') as file:
            pickle.dump(True_labels, file)
        # save Retrieval label sequence to local
        with open(out_label_path / 'Retrieval_labels_with_scores', 'wb') as file:
            pickle.dump(Retrieval_labels_with_scores, file)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    print('Precison@1 {}'.format(precision_score(y_true_all, y_pred_all, average='micro', zero_division=1)))
    print('Precison@2 {}'.format(precision_score(y_true_all, y_pred2_all, average='micro', zero_division=1)))
    print('Precison@3 {}'.format(precision_score(y_true_all, y_pred3_all, average='micro', zero_division=1)))
    print('MicroF1@1 {}'.format(f1_score(y_true_all, y_pred_all, average='micro', zero_division=1)))
    print('MacroF1@1 {}'.format(f1_score(y_true_all, y_pred_all, average='macro', zero_division=1)))
