from transformers import PreTrainedTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch
import logging
from tqdm import tqdm
from torch.nn import functional as F
import os
import pickle
from pathlib import Path
from pretrain.loss.knowledge_hierarchy_distance import KHDistance
from pretrain.loss.RankingContrastiveLoss import ranking_contrastive_loss


def train(model, tokenizer: PreTrainedTokenizer, train_dataset, out_model_path, result, args):
    def collate(examples):
        ques_t, labels, math_id, mlm_input, mlm_labels, attention_mask = [], [], [], [], [], []
        for i in examples:
            ques_t.append(i["raw_input"])  # 目标题目的id
            labels.append(i["ques_labels"])
            math_id.append(i["ques_id"])
            mlm_input.append(i["mlm_input"])
            mlm_labels.append(i["mlm_labels"])
            attention_mask.append(i["attention_mask"])
        return (pad_sequence(ques_t, batch_first=True, padding_value=tokenizer.pad_token_id),
                torch.tensor(labels), math_id,
                pad_sequence(mlm_input, batch_first=True, padding_value=tokenizer.pad_token_id),
                pad_sequence(mlm_labels, batch_first=True, padding_value=tokenizer.pad_token_id),
                pad_sequence(attention_mask, batch_first=True, padding_value=tokenizer.pad_token_id))

    train_loader = DataLoader(
        train_dataset,
        # sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size,
        drop_last=True,  # drop_last=False
        pin_memory=False,
        collate_fn=collate,
        shuffle=True
    )

    # Prepare optimizer and schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  #

    # loss
    know_path = Path(args.data) / args.dataset / 'label_tree_index.csv'
    train_data_path = Path(args.data) / args.dataset / args.train_set
    if args.rank:
        if 'KHD_{}.pkl'.format(args.dataset) in os.listdir('loss'):
            with open('loss/KHD_{}.pkl'.format(args.dataset), 'rb') as f:
                KHD = pickle.load(f)
        else:
            KHD = KHDistance(know_path, train_data_path)
            with open('loss/KHD_{}.pkl'.format(args.dataset), 'wb') as f:
                pickle.dump(KHD, f)
        criterion = ranking_contrastive_loss(KHD, args)
    else:
        criterion = F.cross_entropy

    # Train!
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", args.epochs)
    logging.info("  Instantaneous batch size = %d", args.batch_size)

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        loss_RPCT_all = 0.0
        loss_mlm_all = 0.0
        data_loader_iterator = iter(train_loader)
        epoch_iterator = tqdm(range(len(train_loader)), desc=f'Epoch {epoch + args.local_trained_epoch + 1}',
                              unit="batch")

        for step in epoch_iterator:  # 调用collate函数
            batch = next(data_loader_iterator)
            ques_t, labels, math_id, mlm_input, mlm_labels, attention_mask = batch
            ques_t = ques_t.to(args.device)
            labels = labels.to(args.device)
            mlm_input = mlm_input.to(args.device)
            mlm_labels = mlm_labels.to(args.device)
            attention_mask = attention_mask.to(args.device)
            l_pos, l_neg, labels, labels_queue, ptr, loss_mlm = (
                model(target_input=ques_t,
                      labels=labels,
                      mlm_input=mlm_input,
                      mlm_labels=mlm_labels,
                      attention_mask=attention_mask))
            if args.rank:  # 是否使用排序的loss？
                loss = criterion(l_pos, l_neg, labels, labels_queue)
            else:
                logits = torch.cat([l_pos, l_neg], dim=1)
                logits /= args.min_tau
                labels = torch.zeros(logits.shape[0], dtype=torch.long).to(l_pos.device)
                loss = criterion(logits, labels)
            loss_all = 0.05 * loss + loss_mlm
            # loss_all = loss
            loss_all.backward()
            try:
                loss_RPCT_all += loss.item()
                loss_mlm_all += loss_mlm.item()
            except:
                print(step)
                print(math_id)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            epoch_iterator.set_postfix(loss_all=loss_all.item(), loss_RPCT=loss.item(), loss_mlm=loss_mlm.item())

        ckpt_path = os.path.join(out_model_path,
                                 "{}-{}-{}.pth".format('cl', 'checkpoint', epoch + args.local_trained_epoch + 1))
        # save_model(ckpt_path, model, args)
        model.save_model(ckpt_path)
        avg_RPCT_loss = loss_RPCT_all / len(train_loader)
        avg_llm_loss = loss_mlm_all / len(train_loader)
        print(avg_RPCT_loss)
        print(avg_llm_loss)
        with open(result, 'a+', encoding='utf-8') as f:
            f.write("Epoch : {}, RCPT avgLoss : {}, mlm avgLoss : {}\n".format(epoch + args.local_trained_epoch + 1,
                                                                               avg_RPCT_loss, avg_llm_loss))


def save_model(ckpt_path, model, args):
    os.makedirs(ckpt_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_path, "model.pt"))
    torch.save(args, os.path.join(ckpt_path, "training_args.bin"))
    model.encoder_q.encoder.save_pretrained(ckpt_path)  # 保存bin文件

    logging.info("[CL] Saving model checkpoint to %s", ckpt_path)
    logging.info("[CL] Saving optimizer and scheduler states to %s", ckpt_path)
