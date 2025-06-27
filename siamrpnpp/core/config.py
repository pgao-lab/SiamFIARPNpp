# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = ""

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

# Anchor Target
# Positive anchor threshold
__C.TRAIN.THR_HIGH = 0.6

# Negative anchor threshold
__C.TRAIN.THR_LOW = 0.3

# Number of negative
__C.TRAIN.NEG_NUM = 16

# Number of positive
__C.TRAIN.POS_NUM = 16

# Number of anchors per images
__C.TRAIN.TOTAL_NUM = 64

# exemplar_size
__C.TRAIN.EXEMPLAR_SIZE = 127
# train_search_size
__C.TRAIN.SEARCH_SIZE = 255
# train_bash_size
__C.TRAIN.BASE_SIZE = 8
# train_output_size 需要与models中的输出一致
__C.TRAIN.OUTPUT_SIZE = 17 # 25 17
# train_resume恢复训练
__C.TRAIN.RESUME = ''
# train_log_dir 日志文件的路径
__C.TRAIN.LOG_DIR = ''
# snapshot_dir 模型保存的路劲
__C.TRAIN.SNAPSHOT_DIR = ''
# train_epoch 训练轮数
__C.TRAIN.EPOCH = 50
# 训练开始的轮数
__C.TRAIN.START_EPOCH = 0
# batch_size
__C.TRAIN.BATCH_SIZE = 512# 32
# num_works
__C.TRAIN.NUM_WORKERS = 16
# train_momenntum 用来加速梯度下降并增加梯度下降过程稳定性
__C.TRAIN.MOMENTUM = 0.9
# train_weight_decay 权重衰减因子
__C.TRAIN.WEIGHT_DECAY = 0.0001
# train_cls_weight 分类损失权重
__C.TRAIN.CLS_WEIGHT = 1.0
# train_loc_weight 定位损失权重
__C.TRAIN.LOC_WEIGHT = 1.2



# train_mast_weight
__C.TRAIN.MASK_WEIGHT = 1
# train_print_freg20个轮次 就会输出一次日志信息
__C.TRAIN.PRINT_FREQ = 20

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 0.005

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = True

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

__C.DATASET.TEMPLATE = CN()


__C.DATASET.TEMPLATE.SHIFT = 4

__C.DATASET.TEMPLATE.SCALE = 0.05

__C.DATASET.TEMPLATE.BLUR = 0.0

__C.DATASET.TEMPLATE.FLIP = 0.0

__C.DATASET.TEMPLATE.COLOR = 1.0

__C.DATASET.SEARCH = CN()

__C.DATASET.SEARCH.SHIFT = 64

__C.DATASET.SEARCH.SCALE = 0.18 

__C.DATASET.SEARCH.BLUR = 0.0  

__C.DATASET.SEARCH.FLIP = 0.0  
 
__C.DATASET.SEARCH.COLOR = 1.0 

# Sample Negative pair see [DaSiamRPN](https://arxiv.org/pdf/1808.06048)
# for detail discussion
__C.DATASET.NEG = 0.2

# improve tracking performance for otb100
__C.DATASET.GRAY = 0.0

__C.DATASET.NAMES = ('VID', 'YOUTUBEBB', 'COCO', 'DET', 'GOT')  #('VID', 'COCO', 'DET', 'YOUTUBEBB', 'GOT')

__C.DATASET.VID = CN()
__C.DATASET.VID.ROOT = 'path/to/VID_DATASET'
__C.DATASET.VID.ANNO = 'path/to/VID_DATASET.json'
__C.DATASET.VID.FRAME_RANGE = 100
__C.DATASET.VID.NUM_USE = 100000  # repeat until reach NUM_USE

__C.DATASET.GOT = CN()
__C.DATASET.GOT.ROOT = ''
__C.DATASET.GOT.ANNO = ''
__C.DATASET.GOT.FRAME_RANGE = 100
__C.DATASET.GOT.NUM_USE = 64000  # use all not repeat -1   64000

__C.DATASET.YOUTUBEBB = CN()
__C.DATASET.YOUTUBEBB.ROOT = ''
__C.DATASET.YOUTUBEBB.ANNO = ''
__C.DATASET.YOUTUBEBB.FRAME_RANGE = 3
__C.DATASET.YOUTUBEBB.NUM_USE = -1    # use all not repeat

__C.DATASET.COCO = CN()
__C.DATASET.COCO.ROOT = ''
__C.DATASET.COCO.ANNO = ''
__C.DATASET.COCO.FRAME_RANGE = 1
__C.DATASET.COCO.NUM_USE = -1

__C.DATASET.DET = CN()
__C.DATASET.DET.ROOT = ''
__C.DATASET.DET.ANNO = ''
__C.DATASET.DET.FRAME_RANGE = 1
__C.DATASET.DET.NUM_USE = -1

__C.DATASET.VIDEOS_PER_EPOCH = 64000 # #默认的数量600000, 64000, 32000

# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #

__C.BACKBONE = CN()

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
__C.BACKBONE.TYPE = 'googlenet'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = ''

# Train layers
__C.BACKBONE.TRAIN_LAYERS = []

# Crop_pad
__C.BACKBONE.CROP_PAD = 4

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Backbone offset
__C.BACKBONE.OFFSET = 13

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 10 #第10层开始训练

# Backbone stride
__C.BACKBONE.STRIDE = 8

# Train channel_layer
__C.BACKBONE.CHANNEL_REDUCE_LAYERS = []

# ------------------------------------------------------------------------ #
# Adjust layer options
# ------------------------------------------------------------------------ #
__C.ADJUST = CN()

# Adjust layer
__C.ADJUST.ADJUST = True

__C.ADJUST.KWARGS = CN(new_allowed=True)

# Adjust layer type
__C.ADJUST.TYPE = "GoogLeNetAdjustLayer" # AdjustAllLayer

# ------------------------------------------------------------------------ #
# RPN options
# ------------------------------------------------------------------------ #
__C.RPN = CN()

# RPN type
__C.RPN.TYPE = ''

__C.RPN.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# mask options
# ------------------------------------------------------------------------ #
__C.MASK = CN()

# Whether to use mask generate segmentation
__C.MASK.MASK = False

# Mask type
__C.MASK.TYPE = "MaskCorr"

__C.MASK.KWARGS = CN(new_allowed=True)

__C.REFINE = CN()

# Mask refine
__C.REFINE.REFINE = False

# Refine type
__C.REFINE.TYPE = "Refine"

# ------------------------------------------------------------------------ #
# Anchor options
# ------------------------------------------------------------------------ #
__C.ANCHOR = CN()

# Anchor stride
__C.ANCHOR.STRIDE = 8

# Anchor ratios
__C.ANCHOR.RATIOS = [0.33, 0.5, 1, 2, 3]

# Anchor scales
__C.ANCHOR.SCALES = [8]

# Anchor number
__C.ANCHOR.ANCHOR_NUM = len(__C.ANCHOR.RATIOS) * len(__C.ANCHOR.SCALES)


# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

__C.TRACK.TYPE = 'SiamRPNTracker'

# Scale penalty
__C.TRACK.PENALTY_K = 0.04

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.44

# Interpolation learning rate
__C.TRACK.LR = 0.4

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127

# Instance size
__C.TRACK.INSTANCE_SIZE = 255

# Base size
__C.TRACK.BASE_SIZE = 8

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5

# Long term lost search size
__C.TRACK.LOST_INSTANCE_SIZE = 831

# Long term confidence low
__C.TRACK.CONFIDENCE_LOW = 0.85

# Long term confidence high
__C.TRACK.CONFIDENCE_HIGH = 0.998

# Mask threshold
__C.TRACK.MASK_THERSHOLD = 0.30

# Mask output size
__C.TRACK.MASK_OUTPUT_SIZE = 127
