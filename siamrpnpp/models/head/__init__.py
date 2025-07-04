# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from siamrpnpp.models.head.mask import MaskCorr, Refine
from siamrpnpp.models.head.rpn import UPChannelRPN, DepthwiseRPN, MultiRPN,UPChannelFIARPN,MultiRPNFIA

RPNS = {
        'UPChannelRPN': UPChannelRPN,
        'DepthwiseRPN': DepthwiseRPN,
        'MultiRPN': MultiRPN,
        'UPChannelFIARPN': UPChannelFIARPN,
        'MultiRPNFIA':MultiRPNFIA
       }

MASKS = {
         'MaskCorr': MaskCorr,
        }

REFINE = {
          'Refine': Refine,
         }


def get_rpn_head(name, **kwargs):
    return RPNS[name](**kwargs)


def get_mask_head(name, **kwargs):
    return MASKS[name](**kwargs)


def get_refine_head(name):
    return REFINE[name]()
