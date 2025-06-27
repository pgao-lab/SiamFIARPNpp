# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn

class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l = 4
            r = l + 7
            x = x[:, :, l:r, l:r]
        return x


class AdjustAllLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustAllLayer, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0], out_channels[0])
        else:
            for i in range(self.num):
                self.add_module('downsample'+str(i+2),
                                AdjustLayer(in_channels[i], out_channels[i]))

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample'+str(i+2))
                out.append(adj_layer(features[i]))
            return out

class GoogLeNetAdjustLayer(nn.Module):
    '''
    with mask: F.interpolate
    '''
    def __init__(self, in_channels, out_channels, crop_pad=0, kernel=1):
        super(GoogLeNetAdjustLayer, self).__init__()
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel),
            nn.BatchNorm2d(out_channels, eps=0.001),
        )
        self.crop_pad = crop_pad

    def forward(self, x):
        x = self.channel_reduce(x)

        if x.shape[-1] > 25 and self.crop_pad > 0:
            crop_pad = self.crop_pad
            x = x[:, :, crop_pad:-crop_pad, crop_pad:-crop_pad]

        return x

