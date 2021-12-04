import os
import sys
import numpy as np
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    def __init__(self,in_chl,out_chl,ksize,stride,with_bn=True,activate='lrelu'):
        super(BaseConv, self).__init__()

        pad = (ksize - 1) // 2

        self.conv = nn.Conv2d(in_channels=in_chl,
                              out_channels=out_chl,
                              kernel_size=ksize,
                              stride=stride,
                              padding=pad,
                              bias = False if with_bn else True
                              )
        self.bn = nn.BatchNorm2d(out_chl,eps = 1e-3) if with_bn else None
        self.relu = get_activation(activate) if activate is not None else None

    def forward(self,x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self,in_chl,out_chl,shortcut=True,activate='lrelu'):
        super(BasicBlock, self).__init__()
        down_chl  = in_chl // 2
        self.conv1 = BaseConv(in_chl,down_chl,ksize=1,stride=1,activate=activate)
        self.conv2 = BaseConv(down_chl,out_chl,ksize=3,stride=1,activate=activate)
        self.shortcut = shortcut and in_chl == out_chl

    def forward(self,x):
        conv1  =self.conv1(x)
        out = self.conv2(conv1)
        if self.shortcut:
           out = out + x
        return out



class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.dark0 = BaseConv(3, 32, 3, 1)
        self.dark1 = nn.Sequential(
            BaseConv(32, 64, 3, 2),
            BasicBlock(64, 64, shortcut=True)
        )
        self.dark2 = nn.Sequential(
            BaseConv(64, 128, 3, 2),
            BasicBlock(128, 128, True),
            BasicBlock(128, 128, True),
        )  # layer11
        self.dark3 = nn.Sequential(
            BaseConv(128, 256, 3, 2),
            BasicBlock(256, 256, True),
            BasicBlock(256, 256, True),
            BasicBlock(256, 256, True),
            BasicBlock(256, 256, True),
            BasicBlock(256, 256, True),
            BasicBlock(256, 256, True),
            BasicBlock(256, 256, True),
            BasicBlock(256, 256, True),
        )
        self.dark4 = nn.Sequential(
            BaseConv(256, 512, 3, 2),
            BasicBlock(512, 512, True),
            BasicBlock(512, 512, True),
            BasicBlock(512, 512, True),
            BasicBlock(512, 512, True),
            BasicBlock(512, 512, True),
            BasicBlock(512, 512, True),
            BasicBlock(512, 512, True),
            BasicBlock(512, 512, True),
        )
        self.dark5 = nn.Sequential(
            BaseConv(512, 1024, 3, 2),
            BasicBlock(1024, 1024, True),
            BasicBlock(1024, 1024, True),
            BasicBlock(1024, 1024, True),
            BasicBlock(1024, 1024, True),
        )

    def forward(self,x):
        out0 = self.dark0(x)
        out1 = self.dark1(out0)
        out2 = self.dark2(out1)
        out3 = self.dark3(out2)
        out4 = self.dark4(out3)
        out5 = self.dark5(out4)
        return out3,out4,out5


if __name__ == "__main__":
    x = torch.randn((1,3,416,416))
    model = Darknet53()
    out3,out4,out5 = model(x)
    print(out3.shape)
    print(out4.shape)
    print(out5.shape)
