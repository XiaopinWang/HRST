# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Yanjie Li (leeyegy@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
# import timm
import math
from .Swin_transformer_block import swin224
from .hr_base import HRNET_base

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class HRST(nn.Module):

    def __init__(self, cfg, **kwargs):

        extra = cfg.MODEL.EXTRA

        super(HRST, self).__init__()

        print(cfg.MODEL)
        ##################################################
        self.pre_feature = HRNET_base(cfg,**kwargs)
        self.swin_transformer = swin224()

    def forward(self, x):
        x = self.pre_feature(x)
        x = self.swin_transformer(x)
        return x

    def init_weights(self, pretrained='', cfg=None):
        self.pre_feature.init_weights(pretrained)


def get_pose_net(cfg, is_train, **kwargs):
    model = HRST(cfg, **kwargs)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED,cfg)
    return model
