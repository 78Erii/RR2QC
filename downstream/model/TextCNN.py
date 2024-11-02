# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

'''Convolutional Neural Networks for Sentence Classification'''


class TextCNNModel(nn.Module):
    def __init__(self):
        super(TextCNNModel, self).__init__()
        self.num_label4 = 426
        self.word_embedding = nn.Embedding(22080, 768, padding_idx=0)
        # self.embedding = nn.Embedding(22080, 768, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 256, (k, 768)) for k in (2, 3, 4)])  # main
        self.dropout = nn.Dropout(0.1)
        self.LayerNorm = nn.LayerNorm(768, eps=1e-12)
        self.fc = nn.Linear(256 * len((2, 3, 4)), self.num_label4)
        self.class_center = nn.Parameter((torch.empty(self.num_label4, 768)), requires_grad=True)
        # torch.nn.init.kaiming_uniform_(self.class_center)
        self._init_weights()

    def _init_weights(self):
        init.kaiming_uniform_(self.word_embedding.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.keyword_embedding.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.class_center)
        for conv in self.convs:
            init.kaiming_uniform_(conv.weight, a=math.sqrt(5))
            init.constant_(conv.bias, 0)
        init.kaiming_uniform_(self.fc.weight, a=math.sqrt(5))
        init.constant_(self.fc.bias, 0)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_ids, keyword_ids):
        embedding = self.word_embedding(input_ids)
        out = self.LayerNorm(embedding)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        logit = self.fc(out)
        return logit, out
