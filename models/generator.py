#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.downsampleby2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True) # padding='same' not supported by PyTorch
        self.upsampleby2   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2)
       		)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2)
       		)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2)
       		)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2)
       		)
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2)
       		)
        self.conv6 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2)
       		)
        self.conv7 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2)
       		)
        self.conv8 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2)
       		)
        self.conv9 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
            nn.LeakyReLU(negative_slope=0.2)
            )
        self.conv10= nn.Sequential(
            nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1), # padding='same' not supported by PyTorch
       		)

    def forward(self, input_img, input_variance):
        input_1 	= input_img
        input_2 	= input_variance
        input_cat	= torch.cat((input_1, input_2), 1)
        skips       = []
        #
        # ENCODER block -------------------------------------------
        # conv1
        feats   	= self.conv1(input_cat)
        skips.append(feats)
        feats   	= self.downsampleby2(feats)
        # conv2
        feats   	= self.conv2(feats)
        skips.append(feats)
        feats   = self.downsampleby2(feats)
        # conv3
        feats   = self.conv3(feats)
        skips.append(feats)
        feats   = self.downsampleby2(feats)
        # conv4
        feats   = self.conv4(feats)
        skips.append(feats)
        feats   = self.downsampleby2(feats)
        # conv5
        feats   = self.conv5(feats)
        # DECODER block -------------------------------------------
        # conv6
        feats   =  self.upsampleby2(feats)
        feats   =  torch.cat((feats, skips.pop()), 1)
        feats   =  self.conv6(feats)
        # conv7
        feats   =  self.upsampleby2(feats)
        feats   =  torch.cat((feats, skips.pop()), 1)
        feats   =  self.conv7(feats)
        # conv8
        feats   =  self.upsampleby2(feats)
        feats   =  torch.cat((feats, skips.pop()), 1)
        feats   =  self.conv8(feats)
        # conv9
        feats   =  self.upsampleby2(feats)
        feats   =  torch.cat((feats, skips.pop()), 1)
        feats   =  self.conv9(feats)
        #
        residual= self.conv10(feats)
        return input_1 + residual







