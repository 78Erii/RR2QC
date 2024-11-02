import tqdm
import torch
import random
from torch.utils.data import Dataset
from data.Math_Junior.four_level_label_dict import label_key_index_dict


class RCPTDataset(Dataset):
    def __init__(self, data_path, tokenizer, seq_len, args, data_lines=None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data_lines = data_lines
        self.data_path = data_path
        self.args = args

        with open(data_path, "r", encoding='utf-8') as f:
            self.lines = [line[:-1].split("\t") for line in
                          tqdm.tqdm(f, desc="Loading Dataset", total=data_lines)]
            self.data_lines = len(self.lines)

    def __len__(self):
        return self.data_lines

    def __getitem__(self, item):
        ques_id, text, labels = self.get_corpus_line(item)
        raw_content = self.StrToVec(text)
        raw_content = [101] + raw_content
        mlm_content, mlm_labels = self.random_word(text)
        mlm_content = [101] + mlm_content
        mlm_labels = [-100] + mlm_labels
        assert len(raw_content) == len(mlm_content)

        ques_labels = [0.0 for _ in range(len(label_key_index_dict))]
        for labels_one in labels:
            labels_one_idx = label_key_index_dict[labels_one]
            ques_labels[labels_one_idx] = 1.0

        raw_input_dict = {'input_ids': raw_content[:self.seq_len]}
        mlm_input_dict = {'input_ids': mlm_content[:self.seq_len]}
        mlm_labels = mlm_labels[:self.seq_len]

        raw_input = self.tokenizer.pad(raw_input_dict, max_length=self.seq_len, padding='max_length')
        attention_mask = raw_input['attention_mask']
        mlm_input = self.tokenizer.pad(mlm_input_dict, max_length=self.seq_len, padding='max_length')
        mlm_labels_padding = [-100 for _ in range(self.seq_len - len(mlm_labels))]
        mlm_labels.extend(mlm_labels_padding)

        output = {"ques_id": ques_id,
                  "raw_input": raw_input['input_ids'],
                  "mlm_input": mlm_input['input_ids'],
                  "mlm_labels": mlm_labels,
                  "attention_mask": attention_mask,
                  "ques_labels": ques_labels}

        dic_output = {}
        for key, value in output.items():
            if key == "ques_id":
                dic_output[key] = value
            elif key == 'ques_labels':
                dic_output[key] = value
            else:
                dic_output[key] = torch.tensor(value)
        return dic_output

    def StrToVec(self, content):
        content = content.split()
        tokens = [self.tokenizer.convert_tokens_to_ids(word) for word in content]
        return tokens

    def get_corpus_line(self, item):
        label1 = []
        label2 = []
        label3 = []
        label4 = []
        tags = self.lines[item][2:]
        for id in range(len(tags)):
            if (id + 1) % 4 == 0:
                label1.append(tags[id - 3])
                label2.append(tags[id - 2])
                label3.append(tags[id - 1])
                label4.append(tags[id])
        return self.lines[item][0], self.lines[item][1], label4

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if tokens[i] == '[PAD]':
                output_label.append(-100)
                continue
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = '[MASK]'

                # 10% randomly change token to random token
                elif prob < 0.9:

                    tokens[i] = random.sample(self.args.vocab_list, 1)

                # 10% randomly change token to current token
                output_label.append(self.tokenizer.convert_tokens_to_ids(token))

            else:
                output_label.append(-100)

        process_tokens = []
        for token_after in tokens:
            index = self.tokenizer.convert_tokens_to_ids(token_after)
            if type(index) is list:
                process_tokens.extend(index)
            else:
                process_tokens.append(index)

        return process_tokens, output_label
