import os
import sys
import torch
import logging
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument(
        "--data",
        type=str,
        default=os.getenv("AMLT_DATA_DIR", "data"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.getenv("AMLT_OUTPUT_DIR", "output"),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Math_Junior",
        choices=["Math_Junior", "Math_Senior", "Physics_Senior", "DA_20K", "DA_20K_AUG"])

    parser.add_argument("--train_set", type=str, default="train.txt")
    parser.add_argument("--name", type=str, default="RCPT", help="job name.")

    parser.add_argument("--lr", type=float, default=1e-4)  # 学习率
    parser.add_argument("--batch_size", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local_trained_epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=23)  # 随机种子
    parser.add_argument('--vocab_list', type=str, default=[])
    parser.add_argument("--base_model_name", type=str, default="BERT", choices=["BERT", "RoBERTa", "TextCNN"])
    parser.add_argument("--base_encoder_path", type=str, default=r"E:\nxb\works\实验结果\20231127_BERT预训练权重\train_model\model30",
                        choices=["model/bert-base-chinese", "model/RoBERTa_zh_Large_PyTorch", r"E:\nxb\works\实验结果\20231127_BERT预训练权重\train_model\model30"],
                        help="download your model file into this path")
    parser.add_argument('--class_center_path', type=str, default=[])

    # train
    parser.add_argument("--project", type=str2bool, default=True,
                        help="whether to add a projector after transformer")
    parser.add_argument("--ques_dim", type=int, default=128,
                        help="the dimension of the question representation")
    parser.add_argument("--queue_size", type=int, default=165, help="the size of the memory queue")
    parser.add_argument("--Threshold", type=int, default=0.13, help="")
    parser.add_argument("--momentum", type=float, default=0.95,
                        help="moco momentum of updating key encoder")  # 越小的动量项越不容易归零
    parser.add_argument("--temperature", type=float, default=0.07, help="softmax temperature of Moco")

    parser.add_argument("--Jaccard", type=str2bool, default=True, help="")

    # ranking loss
    parser.add_argument("--rank", type=str2bool, default=True, help="")
    parser.add_argument('--min_tau', default=0.1, type=float, help='min temperature parameter in SimCLR')
    parser.add_argument('--max_tau', default=0.6, type=float, help='max temperature parameter in SimCLR')
    parser.add_argument('--do_sum_in_log', type=str2bool, default=False)  # 在log里求和

    args = parser.parse_args()
    return args
