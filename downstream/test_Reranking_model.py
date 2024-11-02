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
from pathlib import Path
from parameters import parse_args
from model.model_RR2QC import RetrievalRerankingToQuestionClassification
from model.TextCNN import TextCNNModel
# load label dict
from data.Math_Senior.senior_three_level_label_dict import label2metalabel, metalabel_key_index_dict

metalabel_index_key_dict = {value: key for key, value in metalabel_key_index_dict.items()}


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


def build_model(loop):
    local_path = out_path / 'Reranking_trained_model' / 'model{}'.format(loop)
    config = BertConfig.from_pretrained(local_path)
    model = RetrievalRerankingToQuestionClassification(config=config)
    if args.base_model_name == 'BERT' or 'RoBERTa':
        model.load_local_states(local_path)
    elif args.base_model_name == 'TextCNN':
        model = TextCNNModel()
        checkpoint = torch.load(local_path)
        model.load_state_dict(checkpoint)
    return model


if __name__ == '__main__':
    args = parse_args()
    out_path = Path(args.out_path) / f'{args.name}-{args.dataset}'
    out_model_path = out_path / 'Reranking_trained_model'
    out_result_path = out_path / 'Reranking_result'
    out_metalabel_path = out_path / 'metalabel'
    out_metalabel_path.mkdir(exist_ok=True, parents=True)

    print('Loading tokenizer')
    tokenizers_path = Path(args.data_path) / args.dataset / 'Custom_tokenizer'
    tokenizer = BertTokenizer.from_pretrained(tokenizers_path)
    print('tokenizer Size:', len(tokenizer))

    test_data_path = Path(args.data_path) / args.dataset / args.test_set
    test_dataset = RR2QC_Dataset(test_data_path, tokenizer, seq_len=args.max_len, data_lines=None)

    print('Creating Dataloader')
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)

    args.local_trained_epoch = search_local_trained(out_path)
    print('Already trained {} epochs'.format(args.local_trained_epoch))

    print('Building Reranking model')

    # for every saved Reranking_model states, get one unique metalabel sequence(Reranking_metalabels_with_scores.pkl)
    for loop in range(1, args.local_trained_epoch + 1):
        model = build_model(loop)
        model.to(args.device)

        Reranking_metalabels_with_scores = {}

        model.eval()

        y_true_all = []  # true labels
        y_pred_all = []  # top 1 predicted labels
        y_pred2_all = []  # top 2 predicted labels
        y_pred3_all = []  # top 3 predicted labels

        test_data_loader_iterator = iter(test_data_loader)
        test_pbar = tqdm.tqdm(range(len(test_data_loader)), unit='batch')

        for test_idx in test_pbar:
            test_datas = next(test_data_loader_iterator)
            test_datas = {key: value if key == 'ques_id' else value.to(args.device) for key, value in
                          test_datas.items()}
            ques_id = test_datas['ques_id']
            input_ids = test_datas['ques_input']
            with torch.no_grad():
                output = model(input_ids=test_datas['ques_input'],
                               labels=test_datas['ques_meta_labels'],
                               attention_mask=test_datas['attention_mask'],
                               )
            confidence_scores = output[1]
            sigmoid = torch.nn.Sigmoid()
            norm_score = sigmoid(confidence_scores)
            labels = test_datas['ques_meta_labels'].cpu().numpy()

            y_true_all.extend(labels)

            for i in range(len(labels[:])):
                # get top n pred label indices and logits
                n = len(metalabel_key_index_dict)
                top_n_logits_pred, top_n_indices_pred = torch.topk(norm_score[i], k=n)

                y_pred = [0 for _ in range(len(metalabel_key_index_dict))]
                y_pred[top_n_indices_pred[0]] = 1
                y_pred_all.append(y_pred)

                y_pred2 = [0 for _ in range(len(metalabel_key_index_dict))]
                for idx2 in top_n_indices_pred[:2]:
                    y_pred2[idx2] = 1
                y_pred2_all.append(y_pred2)

                y_pred3 = [0 for _ in range(len(metalabel_key_index_dict))]
                for idx3 in top_n_indices_pred[:3]:
                    y_pred3[idx3] = 1
                y_pred3_all.append(y_pred3)

                with torch.no_grad():
                    top_n_logits_pred = top_n_logits_pred.cpu().numpy()
                    top_n_indices_pred = top_n_indices_pred.cpu().numpy()
                top_n_labelname_pred = [metalabel_index_key_dict[i] for i in top_n_indices_pred]
                top_n_labels_with_scores = [top_n_labelname_pred[i] + '\t' + str('%.3f' % top_n_logits_pred[i]) for
                                            i in
                                            range(n)]
                Reranking_metalabels_with_scores[ques_id[i]] = top_n_labels_with_scores  # record metalabel sequence                # write pred result

        # save Reranking label sequence to local
        with open(out_metalabel_path / 'Reranking_metalabels_with_scores_loop{}'.format(loop), 'wb') as file:
            pickle.dump(Reranking_metalabels_with_scores, file)

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        print('Precison@1 {}'.format(precision_score(y_true_all, y_pred_all, average='micro', zero_division=1)))
        print('Precison@2 {}'.format(precision_score(y_true_all, y_pred2_all, average='micro', zero_division=1)))
        print('Precison@3 {}'.format(precision_score(y_true_all, y_pred3_all, average='micro', zero_division=1)))
        print('MicroF1@1 {}'.format(f1_score(y_true_all, y_pred_all, average='micro', zero_division=1)))
        print('MacroF1@1 {}'.format(f1_score(y_true_all, y_pred_all, average='macro', zero_division=1)))
