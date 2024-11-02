import os
import re
import torch
import random
import numpy as np
from pathlib import Path
from parameters import parse_args
from transformers import BertTokenizer, BertConfig
from pretrain.model.model_RPCT import RCPTMoCo
from dataset import RCPTDataset
from train import train
from data.Math_Junior.four_level_label_dict import label_key_index_dict


def set_seed(args):
    random.seed(args.seed)  # py
    np.random.seed(args.seed)  # numpy
    torch.manual_seed(args.seed)  # torch
    torch.cuda.manual_seed(args.seed)  # cuda
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def search_local_trained(out_path):
    model_nums = os.listdir(out_path)
    nums = []
    for i in range(len(model_nums)):
        num = int(re.findall('cl-checkpoint-(\d+).pth', model_nums[i])[0])
        nums.append(num)
    if nums:
        max_nums = max(nums)
    else:
        max_nums = 0
    return max_nums


if __name__ == "__main__":
    args = parse_args()
    out_model_path = Path(args.output) / 'model'
    out_model_path.mkdir(exist_ok=True, parents=True)
    result = Path(args.output) / 'result.txt'
    args.local_trained_epoch = search_local_trained(out_model_path)
    args.class_center_path = Path(args.data) / f"{args.dataset}/class_center.pkl"

    set_seed(args)

    data_path = Path(args.data) / args.dataset / args.train_set

    print('Loading tokenizer')
    tokenizers_path = Path(args.data) / args.dataset / 'Custom_tokenizer'
    tokenizer = BertTokenizer.from_pretrained(tokenizers_path)

    vocab_list = tokenizers_path / 'vocab_latex.txt'
    with open(vocab_list, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    vocab_list = [line.strip() for line in lines]
    args.vocab_list = vocab_list

    print("begin training")
    print("loading model")

    config = BertConfig.from_pretrained(args.base_encoder_path)
    config.num_labels = len(label_key_index_dict)
    model = RCPTMoCo(bert_type=args.base_encoder_path, d_rep=args.ques_dim, project=args.project, K=args.queue_size,
                     m=args.momentum, config=config)
    # model.encoder_q.encoder.resize_token_embeddings(len(tokenizer))
    # model.encoder_k.encoder.resize_token_embeddings(len(tokenizer))
    # model.cls.resize_cls_vocab(len(tokenizer))
    if args.local_trained_epoch:
        model.load_model(os.path.join(out_model_path, 'cl-checkpoint-{}.pth'.format(args.local_trained_epoch)))

    model.to(args.device)

    print("loading data")
    train_dataset = RCPTDataset(data_path, tokenizer, seq_len=512, args=args)
    train(model, tokenizer, train_dataset, out_model_path, result, args)
