import pickle

import torch.nn as nn
import torch
import pandas as pd
import torch.nn.functional as F
import logging
from tqdm import tqdm
import numpy as np
import math


class ranking_contrastive_loss(object):
    def __init__(self, KHD, args):
        self.min_tau = args.min_tau
        self.max_tau = args.max_tau
        self.do_sum_in_log = args.do_sum_in_log
        self.args = args
        self.cross_entropy = nn.CrossEntropyLoss()
        # KHD Kwargs
        self.know_df = KHD.know_df
        self.label_paths = KHD.label_paths
        self.lines = KHD.lines
        self.label_group_distance_cache = KHD.label_group_distance_cache
        self.rank_list = KHD.rank_list
        self.Threshold = args.Threshold
        self.class_center = nn.Parameter((torch.empty(len(self.label_paths), 768)), requires_grad=True)
        self.init_from_local_class_center(args)

    def init_from_local_class_center(self, args):
        with open(args.class_center_path, 'rb') as f:
            class_center = pickle.load(f)
        self.class_center.data.copy_(class_center)
        self.class_center.to(self.args.device)

    def Compute_label_Similarity(self, Y_t, Y_p):
        len_Yt = len(Y_t)
        len_Yp = len(Y_p)

        similarity = 0.0
        for y_i in Y_t:
            vector1 = self.class_center[y_i]
            for y_j in Y_p:
                vector2 = self.class_center[y_j]
                s1 = torch.dot(vector1, vector2)  # 点积为负数
                norm1 = torch.norm(vector1)
                norm2 = torch.norm(vector2)
                cosine_similarity = s1 / (norm1 * norm2)
                similarity = similarity + (cosine_similarity + 1) / 2
        similarity /= (len_Yt * len_Yp)
        return similarity

    def sum_in_log(self, l_pos, l_neg, tau):
        logits = torch.cat([l_pos, l_neg], dim=1) / tau
        logits = F.softmax(logits, dim=1)  # make -inf 0.0  # 有nan出现了
        sum_pos = logits[:, 0:l_pos.shape[1]].sum(1)
        sum_pos = sum_pos[sum_pos > 1e-7]  #
        if len(sum_pos) > 0:
            loss = - torch.log(sum_pos).mean()
        else:
            loss = torch.tensor([0.0]).to(self.args.device)  # 要不要反传？
        return loss

    def sum_out_log(self, l_pos, l_neg, tau_pos, tau_neg, Weight_t2p):
        l_pos = l_pos / tau_pos
        l_neg = l_neg / tau_neg
        l_pos_exp = torch.exp(l_pos)
        l_pos_exp_weighted = l_pos_exp * Weight_t2p
        l_neg_exp_sum = torch.exp(l_neg).sum(dim=1).unsqueeze(1)
        all_scores = (l_pos_exp_weighted / (l_pos_exp_weighted + l_neg_exp_sum))
        all_scores = all_scores[all_scores > 1e-7]
        if len(all_scores) > 0:
            loss = - torch.log(all_scores).mean()
        else:
            # loss = - torch.log(all_scores).mean()
            loss = torch.tensor([0.0]).to(self.args.device)
        return loss

    def get_dynamic_tau(self, rank, max_dis_idx):
        rank_idx = self.rank_list.index(rank)
        d_taus = self.min_tau + (rank_idx / max_dis_idx) * (self.max_tau - self.min_tau)
        return d_taus

    def get_neg_dynamic_tau(self, dis_t2p, max_dis_idx):
        rank_array = torch.tensor(self.rank_list, device=self.args.device)  # 将 rank_list 转换为张量
        # 将 dis_t2p 平铺为一维张量
        flat_dis_t2p = dis_t2p.view(-1)  # 展平距离矩阵
        # 获取当前 batch 中的所有距离在 rank_array 中的索引
        dis_t2p_idx = torch.tensor([torch.where(rank_array == dis.item())[0].item() for dis in flat_dis_t2p],
                                   device=self.args.device)
        # 将 dis_t2p_idx 重新调整回原来的形状
        dis_t2p_idx = dis_t2p_idx.view(dis_t2p.shape)

        d_taus = self.min_tau + (dis_t2p_idx / max_dis_idx) * (self.max_tau - self.min_tau)
        return d_taus

    def __call__(self, l_pos_raw, l_neg_raw, labels, label_queue):
        '''
        l_pos: (batch size, 1) the similarity of target question and masked question in current batch
        l_neg: (bz, K) the similarity of target question and queue in whole queue
        labels: (bz, number_labels) the multi-hot vectors of leaf label groups in current batch
        label_queue(K, number_labels) the multi-hot vectors of leaf label groups in whole queue
        '''
        max_dis_idx = math.ceil(len(self.rank_list) * self.Threshold)  # len(torch.where(l_neg_raw > 0)[0])越来越低
        self.rank_list_cur = self.rank_list[
                             :max_dis_idx]  # max_dis_idx应该是有效长度：一个批次内所有可能距离的最大距离。但是当8为最大值时，已经没有负样本，此时计算loss为0
        max_dis = self.rank_list_cur[-1]  # 有效距离应该动态调整，以适应缩放的温度
        print(len(torch.where(l_neg_raw > 0)[0]), max_dis)

        BZ = l_pos_raw.shape[0]
        K = l_neg_raw.shape[1]
        dis_t2p = torch.zeros((BZ, K), device=self.args.device)
        Weight_t2p = torch.ones((BZ, K), device=self.args.device)
        for i in range(BZ):
            cur_indices = torch.nonzero(labels[i], as_tuple=False).cpu().numpy()
            target_label_group_indices = tuple(sorted(map(int, cur_indices)))
            for j in range(K):
                k_multi_hot = label_queue[j]
                k_indices = torch.nonzero(k_multi_hot, as_tuple=False)
                if k_indices.numel() != 0:  # 如果元素总数不为0(排除为入队的初始化k)
                    k_indices = k_indices[0].cpu().numpy()
                    label_group_indices = tuple(sorted(map(int, k_indices)))
                    label_pair = (target_label_group_indices,
                                  label_group_indices) if target_label_group_indices < label_group_indices else (
                        label_group_indices, target_label_group_indices)
                    if target_label_group_indices == label_group_indices:
                        dis = 0
                    else:
                        if label_pair in self.label_group_distance_cache.keys():
                            dis = self.label_group_distance_cache[label_pair]
                        else:
                            dis = max_dis  # 可以直接计算
                    if dis < max_dis:
                        Weight = self.Compute_label_Similarity(target_label_group_indices, label_group_indices)
                        Weight_t2p[i][j] = Weight
                else:
                    dis = max_dis  # 有效最大距离
                dis_t2p[i][j] = dis

        #

        res = {}
        rank_count = {}
        for i in range(max_dis_idx - 1):  # 最远的以及有效最远的不优化
            rank = self.rank_list_cur[i]

            mask_cur = (dis_t2p == rank)
            count_pos = len(torch.where(mask_cur == True)[0])
            rank_count.update({rank: count_pos})

            mask_far = (dis_t2p > rank)
            l_pos_cur = l_neg_raw.clone()
            l_neg_cur = l_neg_raw.clone()  # 全变成了负数
            l_pos_cur[~mask_cur] = -float("inf")
            l_neg_cur[~mask_far] = -float("inf")

            # l_pos_cur = l_pos_cur * Weight_t2p  # 有负数，导致-float("inf")变成正无穷
            if torch.all(l_pos_cur < 2e-7):
                continue
            taus_pos = self.get_dynamic_tau(rank, max_dis_idx)
            taus_neg = self.get_neg_dynamic_tau(dis_t2p, max_dis_idx)

            if self.do_sum_in_log:
                loss = self.sum_in_log(l_pos_cur, l_neg_cur, taus_pos)
            else:
                loss = self.sum_out_log(l_pos_cur, l_neg_cur, taus_pos, taus_neg, Weight_t2p)

            res['rank_' + str(rank)] = loss
        print('正样本对数' + str({'rank_' + str(key): value for key, value in rank_count.items() if value != 0}))
        return self.criterion(res)

    def criterion(self, outputs):
        loss = 0.0
        count = 0
        # legal_key = []
        for key, val in outputs.items():
            if val != 0:
                loss = loss + val
                count += 1
                # legal_key.append(key)
        if count:
            loss = loss / float(count)
        else:
            loss = torch.tensor([0.0]).to(self.args.device)
        # print('有效loss: '+str(legal_key))
        return loss
