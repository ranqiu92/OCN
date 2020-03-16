# coding=utf-8

import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from util import sequence_mask, split_bert_sequence


class AttentivePooling(nn.Module):

    def __init__(self, input_size):
        super(AttentivePooling, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, input, mask):
        bsz, length, size = input.size()
        score = self.fc(input.contiguous().view(-1, size)).view(bsz, length)
        score = score.masked_fill((1 - mask), -float('inf'))
        prob = F.softmax(score, dim=-1)
        attn = torch.bmm(prob.unsqueeze(1), input)
        return attn


class TriLinear(nn.Module):

    def __init__(self, input_size):
        super(TriLinear, self).__init__()
        self.w1 = nn.Parameter(torch.FloatTensor(1, input_size))
        self.w2 = nn.Parameter(torch.FloatTensor(1, input_size))
        self.w3 = nn.Parameter(torch.FloatTensor(1, input_size))

        self.init_param()

    def forward(self, query, key):
        ndim = query.dim()
        q_logit = F.linear(query, self.w1)
        k_logit = F.linear(key, self.w2)

        shape = [1] * (ndim - 1) + [-1]
        dot_k = self.w3.view(shape) * key
        dot_logit = torch.matmul(query, torch.transpose(dot_k, -1, -2))

        logit = q_logit + torch.transpose(k_logit, -1, -2) + dot_logit
        return logit

    def init_param(self):
        init.normal_(self.w1, 0., 0.02)
        init.normal_(self.w2, 0., 0.02)
        init.normal_(self.w3, 0., 0.02)


class Attention(nn.Module):

    def __init__(self, sim):
        super(Attention, self).__init__()
        self.sim = sim

    def forward(self, query, key, value, query_mask=None, key_mask=None):
        ndim = query.dim()
        logit = self.sim(query, key)
        if query_mask is not None and key_mask is not None:
            mask = query_mask.unsqueeze(ndim - 1) * key_mask.unsqueeze(ndim - 2)
            logit = logit.masked_fill((1 - mask), -float('inf'))

        attn_weight = F.softmax(logit, dim=-1)
        if query_mask is not None and key_mask is not None:
            attn_weight = attn_weight.masked_fill((1-mask), 0.)

        attn = torch.matmul(attn_weight, value)

        kq_weight = F.softmax(logit, dim=1)
        if query_mask is not None and key_mask is not None:
            kq_weight = kq_weight.masked_fill((1 - mask), 0.)

        co_weight = torch.matmul(attn_weight, torch.transpose(kq_weight, -1, -2))
        co_attn = torch.matmul(co_weight, query)

        return (attn, attn_weight), (co_attn, co_weight)


class OCN(BertPreTrainedModel):

    def __init__(self, config, num_labels, max_doc_len, max_query_len, max_option_len):
        super(OCN, self).__init__(config)
        self.bert = BertModel(config)

        self.attn_sim = TriLinear(config.hidden_size)
        self.attention = Attention(sim=self.attn_sim)
        self.attn_fc = nn.Linear(config.hidden_size * 3, config.hidden_size, bias=True)

        self.opt_attn_sim = TriLinear(config.hidden_size)
        self.opt_attention = Attention(sim=self.opt_attn_sim)
        self.comp_fc = nn.Linear(config.hidden_size * 7, config.hidden_size, bias=True)

        self.query_attentive_pooling = AttentivePooling(input_size=config.hidden_size)
        self.gate_fc = nn.Linear(config.hidden_size * 3, config.hidden_size, bias=True)

        self.opt_selfattn_sim = TriLinear(config.hidden_size)
        self.opt_self_attention = Attention(sim=self.opt_selfattn_sim)
        self.opt_selfattn_fc = nn.Linear(config.hidden_size * 4, config.hidden_size, bias=True)

        self.score_fc = nn.Linear(config.hidden_size, 1)
        self.num_labels = num_labels
        self.hidden_size = config.hidden_size

        self.max_doc_len = max_doc_len
        self.max_query_len = max_query_len
        self.max_option_len = max_option_len

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, query_len, opt_len, labels=None):
        bsz = input_ids.size(0)
        input_ids = input_ids.view(bsz * self.num_labels, -1)
        token_type_ids = token_type_ids.view(bsz * self.num_labels, -1)
        attention_mask = attention_mask.view(bsz * self.num_labels, -1)

        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        last_layer = all_encoder_layers[-1]

        doc_enc = last_layer[:, 0 : self.max_doc_len + 2, :]
        opt_enc = last_layer[:, self.max_doc_len + 2:, :]

        opt_len = opt_len.view(-1)
        query_len = query_len.view(-1)
        _, query_enc, _ = split_bert_sequence(opt_enc, query_len, self.max_query_len, opt_len, self.max_option_len, has_cls=False)

        query_total_len = query_enc.size(1)
        query_mask = sequence_mask(query_len, query_total_len)
        query_attn = self.query_attentive_pooling(query_enc, query_mask)

        opt_total_len = opt_enc.size(1)
        opt_mask = attention_mask[:, self.max_doc_len + 2:] > 0
        doc_mask = attention_mask[:, 0 : self.max_doc_len + 2] > 0

        opt_mask = opt_mask.view(bsz, self.num_labels, opt_total_len)
        opt_enc = opt_enc.view(bsz, self.num_labels, opt_total_len, self.hidden_size)

        # Option Comparison
        correlation_list = []
        for i in range(self.num_labels):
            cur_opt = opt_enc[:, i, :, :]
            cur_mask = opt_mask[:, i, :]

            comp_info = []
            for j in range(self.num_labels):
                if j == i:
                    continue

                tmp_opt = opt_enc[:, j, :, :]
                tmp_mask = opt_mask[:, j, :]

                (attn, _), _ = self.opt_attention(cur_opt, tmp_opt, tmp_opt, cur_mask, tmp_mask)
                comp_info.append(cur_opt * attn)
                comp_info.append(cur_opt - attn)

            correlation = torch.tanh(self.comp_fc(torch.cat([cur_opt] + comp_info, dim=-1)))
            correlation_list.append(correlation)

        correlation_list = [correlation.unsqueeze(1) for correlation in correlation_list]
        opt_correlation = torch.cat(correlation_list, dim=1)

        opt_mask = opt_mask.view(bsz * self.num_labels, opt_total_len)
        opt_enc = opt_enc.view(bsz * self.num_labels, opt_total_len, self.hidden_size)
        opt_correlation = opt_correlation.contiguous().view(bsz * self.num_labels, opt_total_len, self.hidden_size)
        gate = torch.sigmoid(self.gate_fc(torch.cat((opt_enc, opt_correlation, query_attn.expand_as(opt_enc)), -1)))
        option = opt_enc * gate + opt_correlation * (1.0 - gate)

        (attn, _), (coattn, _) = self.attention(option, doc_enc, doc_enc, opt_mask, doc_mask)
        fusion = self.attn_fc(torch.cat((option, attn, coattn), -1))
        fusion = F.relu(fusion)

        (attn, _), _ = self.opt_self_attention(fusion, fusion, fusion, opt_mask, opt_mask)
        fusion = self.opt_selfattn_fc(torch.cat((fusion, attn, fusion * attn, fusion - attn), -1))
        fusion = F.relu(fusion)

        fusion = fusion.masked_fill(
            (1 - opt_mask).unsqueeze(-1).expand(bsz * self.num_labels, opt_total_len, self.hidden_size),
            -float('inf'))
        fusion, _ = fusion.max(dim=1)

        logits = self.score_fc(fusion).view(bsz, self.num_labels)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits
