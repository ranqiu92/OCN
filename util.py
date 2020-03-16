# coding=utf-8

import torch


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def split_bert_sequence(seq, seq1_lengths, max_seq1_length, seq2_lengths, max_seq2_length, pad = 0, has_cls = True):
    if has_cls:
        cls = seq[:, 0, :]
    else:
        cls = None
    begin_index = 1 if has_cls else 0
    seq1 = seq[:, begin_index: max_seq1_length + begin_index, :]
    seq1_mask = sequence_mask(seq1_lengths, max_seq1_length)
    seq1 = seq1.float().masked_fill(
        (1 - seq1_mask).unsqueeze(2),
        float(pad),
    ).type_as(seq1)
    seq_range = torch.arange(0, max_seq2_length).long().unsqueeze(0)
    if seq1_lengths.is_cuda:
        seq_range = seq_range.cuda()
    seq_index = seq_range + seq1_lengths.unsqueeze(1) + 1 + begin_index
    batch_size, index_len = seq_index.size()
    dim = seq.size()[2]
    seq2 = torch.gather(seq, dim=1, index=seq_index.unsqueeze(2).expand(batch_size, index_len, dim))
    seq2_mask = sequence_mask(seq2_lengths, max_seq2_length)
    seq2 = seq2.float().masked_fill(
        (1 - seq2_mask).unsqueeze(2),
        float(pad),
    ).type_as(seq2)
    return cls, seq1, seq2
