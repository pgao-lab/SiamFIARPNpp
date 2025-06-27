# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'



# def get_cls_loss(pred, label, select):
#     if len(select.size()) == 0:
#         return 0
#     pred = torch.index_select(pred, 0, select)
#     label = torch.index_select(label, 0, select)
#     return F.nll_loss(pred, label)
#
# def select_cross_entropy_loss(pred, label):
#     pred = pred.view(-1, 2)
#     label = label.view(-1)
#     pos = label.data.eq(1).nonzero().squeeze().cuda()
#     neg = label.data.eq(0).nonzero().squeeze().cuda()
#     loss_pos = get_cls_loss(pred, label, pos)
#     loss_neg = get_cls_loss(pred, label, neg)
#     return loss_pos * 0.5 + loss_neg * 0.5


def select_cross_entropy_loss(pred, label, weight=None):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()

    if weight is not None:
        weight_pos = weight[pos]
        weight_neg = weight[neg]
    else:
        weight_pos = None
        weight_neg = None
    loss_pos = get_cls_loss(pred, label, pos, weight_pos)
    loss_neg = get_cls_loss(pred, label, neg, weight_neg)
    return loss_pos * 0.5 + loss_neg * 0.5

def get_cls_loss(pred, label, select, weight=None):
    if len(select.size()) == 0:
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)

    if weight is not None:
        weight = torch.index_select(weight, 0, select)  # 获取对应样本的权重
        return F.nll_loss(pred, label, weight=weight)
    return F.nll_loss(pred, label)


# def weight_l1_loss(pred_loc, label_loc, loss_weight):
#     b, _, sh, sw = pred_loc.size()
#     pred_loc = pred_loc.view(b, 4, -1, sh, sw)
#     diff = (pred_loc - label_loc).abs()
#     diff = diff.sum(dim=1).view(b, -1, sh, sw)
#     loss = diff * loss_weight
#     return loss.sum().div(b)

def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    # print("pred_loc",pred_loc.shape)
    # print("label_loc",label_loc.shape)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)

    diff = torch.clamp(diff, min=0.0, max=1e6)  # Clip to reasonable range
    loss = diff * loss_weight
    return loss.sum().div(b)
