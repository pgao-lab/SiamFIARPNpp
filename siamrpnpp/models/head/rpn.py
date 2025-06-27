# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from siamrpnpp.core.xcorr import xcorr_fast, xcorr_depthwise
from siamrpnpp.models.init_weight import init_weights



class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

class UPChannelRPN(RPN):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=256):  # out_channels没有用到
        super(UPChannelRPN, self).__init__()

        cls_output = 2 * anchor_num

        loc_output = 4 * anchor_num

        self.template_cls_conv = nn.Conv2d(in_channels,
                                           in_channels * cls_output, kernel_size=3)

        self.template_loc_conv = nn.Conv2d(in_channels,
                                           in_channels * loc_output, kernel_size=3)

        self.search_cls_conv = nn.Conv2d(in_channels,
                                         in_channels, kernel_size=3)

        self.search_loc_conv = nn.Conv2d(in_channels,
                                         in_channels, kernel_size=3)

        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)

    def forward(self, z_f, x_f):

        cls_kernel = self.template_cls_conv(z_f)  # 11111

        loc_kernel = self.template_loc_conv(z_f)  # 22222

        cls_feature = self.search_cls_conv(x_f)  # 11111

        loc_feature = self.search_loc_conv(x_f)  # 22222

        cls = xcorr_fast(cls_feature, cls_kernel)  # 11111

        loc = self.loc_adjust(xcorr_fast(loc_feature, loc_kernel))  # 22222

        return cls, loc

class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, hidden_kernel_size=5):
        super(DepthwiseXCorr, self).__init__()

        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1)
        )

    def forward(self, kernel, search):

        kernel = self.conv_kernel(kernel)

        search = self.conv_search(search)

        feature = xcorr_depthwise(search, kernel)

        out = self.head(feature)

        return out

class DepthwiseRPN(RPN):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=256):
        super(DepthwiseRPN, self).__init__()

        self.cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num)

        self.loc = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num)

    def forward(self, z_f, x_f):

        cls = self.cls(z_f, x_f)

        loc = self.loc(z_f, x_f)
        return cls, loc

class MultiRPN(RPN):

    def __init__(self, anchor_num, in_channels, weighted=False):
        super(MultiRPN, self).__init__()
        self.weighted = weighted

        for i in range(len(in_channels)):

            self.add_module('rpn' + str(i + 2),
                            DepthwiseRPN(anchor_num, in_channels[i], in_channels[i]))

        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))


    def forward(self, z_fs, x_fs):

        cls = []
        loc = []

        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):

            rpn = getattr(self, 'rpn' + str(idx))

            c, l = rpn(z_f, x_f)
            cls.append(c)
            loc.append(l)

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)

        def avg(lst):

            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):

            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight)
        else:
            return avg(cls), avg(loc)

class UPChannelFIARPN(RPN):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=0,feat_channels=256,stacked_convs=3):  # out_channels没有用到
        super(UPChannelFIARPN, self).__init__()

        self.stacked_convs = stacked_convs

        cls_output = 2 * anchor_num
        loc_output = 4 * anchor_num

        self.template_cls_conv = nn.Conv2d(in_channels,
                                           in_channels * cls_output, kernel_size=3)

        self.template_loc_conv = nn.Conv2d(in_channels,
                                           in_channels * loc_output, kernel_size=3)
        self.search_cls_conv = nn.Conv2d(in_channels,
                                         in_channels, kernel_size=3)
        self.search_loc_conv = nn.Conv2d(in_channels,
                                         in_channels, kernel_size=3)
        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)

        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels

            cls_conv = nn.Conv2d(
                chn,
                self.feat_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )

            self.cls_convs.append(nn.Sequential(cls_conv))

            reg_conv = nn.Conv2d(
                chn,
                self.feat_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )

            self.reg_convs.append(nn.Sequential(reg_conv))

            self.dcn = nn.Conv2d(
                in_channels=2 * self.feat_channels,
                out_channels=2 * self.feat_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )

    def fian(self, input_feat):

        cls_feat = input_feat
        reg_feat = input_feat

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)

        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        feat = torch.cat((cls_feat, reg_feat), dim=1)

        if feat.shape[2] < 3:
            feat = F.pad(feat, (0, 0, 0, 3 - feat.shape[2]), mode='constant', value=0)
        if feat.shape[3] < 3:
            feat = F.pad(feat, (0, 3 - feat.shape[3]), mode='constant', value=0)

        feat = self.dcn(feat)

        cls_feat = feat[:, :cls_feat.shape[1], ...]
        reg_feat = feat[:, cls_feat.shape[1]:, ...]
        return cls_feat, reg_feat


    def forward(self, z_f, x_f):

        cls_kernel_feat, loc_kernel_feat = self.fian(z_f)

        cls_feature_feat, loc_feature_feat = self.fian(x_f)

        cls_kernel = self.template_cls_conv(cls_kernel_feat)

        loc_kernel = self.template_loc_conv(loc_kernel_feat)

        cls_feature = self.search_cls_conv(cls_feature_feat)
        loc_feature = self.search_loc_conv(loc_feature_feat)

        cls = xcorr_fast(cls_feature, cls_kernel)

        loc = self.loc_adjust(xcorr_fast(loc_feature, loc_kernel))  # 22222

        return cls, loc

class DepthwiseRPNFIA(RPN):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=256,feat_channels=256,stacked_convs=3):
        super(DepthwiseRPNFIA, self).__init__()

        self.cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num)

        self.loc = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num)

        self.in_channels = in_channels
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels

            cls_conv = nn.Conv2d(
                chn,
                self.feat_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )

            self.cls_convs.append(nn.Sequential(cls_conv))

            reg_conv = nn.Conv2d(
                chn,
                self.feat_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )

            self.reg_convs.append(nn.Sequential(reg_conv))

            self.dcn = nn.Conv2d(
                in_channels=2 * self.feat_channels,
                out_channels=2 * self.feat_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )

    def fian(self, input_feat):

        cls_feat = input_feat
        reg_feat = input_feat

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)

        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        feat = torch.cat((cls_feat, reg_feat), dim=1)

        if feat.shape[2] < 3:
            feat = F.pad(feat, (0, 0, 0, 3 - feat.shape[2]), mode='constant', value=0)
        if feat.shape[3] < 3:
            feat = F.pad(feat, (0, 3 - feat.shape[3]), mode='constant', value=0)

        feat = self.dcn(feat)

        cls_feat = feat[:, :cls_feat.shape[1], ...]
        reg_feat = feat[:, cls_feat.shape[1]:, ...]
        return cls_feat, reg_feat

    def forward(self, z_f, x_f):

        cls_kernel_feat, loc_kernel_feat = self.fian(z_f)
        cls_feature_feat, loc_feature_feat = self.fian(x_f)

        cls = self.cls(cls_kernel_feat, cls_feature_feat)
        loc = self.loc(loc_kernel_feat, loc_feature_feat)

        return cls, loc

class MultiRPNFIA(RPN):
    def __init__(self, anchor_num, in_channels, weighted=False):
        super(MultiRPNFIA, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('rpn' + str(i + 2),
                            DepthwiseRPNFIA(anchor_num, in_channels[i], in_channels[i]))
        if self.weighted:

            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)), requires_grad=True)
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)), requires_grad=True)

    def forward(self, z_fs, x_fs):
        cls = []
        loc = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            rpn = getattr(self, 'rpn' + str(idx))
            c, l = rpn(z_f, x_f)
            cls.append(c)
            loc.append(l)

        if self.weighted:

            cls_weight = F.softmax(self.cls_weight, dim=0)
            loc_weight = F.softmax(self.loc_weight, dim=0)

            cls_out = self.weighted_avg(cls, cls_weight)
            loc_out = self.weighted_avg(loc, loc_weight)
            return cls_out, loc_out
        else:
            return self.avg(cls), self.avg(loc)

    def weighted_avg(self, lst, weight):
        return sum([l * w for l, w in zip(lst, weight)])

    def avg(self, lst):

        return sum(lst) / len(lst)
