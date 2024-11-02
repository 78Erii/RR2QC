import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument(
        "--data_path",
        type=str,
        default='data',
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="output",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Math_Senior",
        choices=["Math_Junior", "Math_Senior", "Physics_Senior", "DA_20K", "DA_20K_AUG"])

    parser.add_argument("--train_set", type=str, default="train.txt")
    parser.add_argument("--val_set", type=str, default="val.txt")
    parser.add_argument("--test_set", type=str, default="test.txt")

    # Train configuration
    parser.add_argument("--name", type=str, default="Retrieval", help="job name.")
    parser.add_argument("--base_model_name", type=str, default="BERT", choices=["BERT", "RoBERTa", "TextCNN"])
    parser.add_argument("--base_encoder_path", type=str, default='model/math_senior_model_pretrain',
                        choices=["model/model94", "model/RoBERTa_zh_Large_PyTorch"],
                        help="download the model file from huggingface or github and copy it to this path")

    parser.add_argument("--lr", type=float, default=2e-5, help='learning rate')
    parser.add_argument("--batch_size", type=int, default=32,
                        help="for a GPU below 24GB of memory, recommended to set the batch size to less than 32")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--local_trained_epoch", type=int, default=0, help='local checkpoint_num of max trained epoch')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=23, help='random seed')
    parser.add_argument("--lamda", type=float, default=0.1, help='a hyperparameter controlling loss_CCL and loss_BCE')
    parser.add_argument("--is_init_CCL", type=str, default=False,
                        help='train the Retrieval model with (batch_size+4) samples, '
                             'which 4 samples consist of label text and one-hot label according to data_path/only_label_input.txt')
    parser.add_argument("--is_use_CCL", type=str, default=False, help='After getting the class_center_vector(class_center_label_input.pkl),'
                                                                      'you can train the Retrieval model again with Class Center Learning task.')

    args = parser.parse_args()
    return args
