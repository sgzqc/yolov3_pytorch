import os
import sys
import numpy as np
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
from darknet import BaseConv,Darknet53


class YOLOBlock(nn.Module):
    def __init__(self,in_chl,out_chl):
        super(YOLOBlock, self).__init__()
        self.conv1 = BaseConv(in_chl,out_chl,ksize=1,stride=1)
        self.conv2 = BaseConv(out_chl,out_chl*2,ksize=3,stride=1)
        self.conv3 = BaseConv(out_chl*2, out_chl, ksize=1, stride=1)
        self.conv4 = BaseConv(out_chl, out_chl * 2, ksize=3, stride=1)
        self.conv5 = BaseConv(out_chl*2,out_chl,ksize=1,stride=1)

    def forward(self,x):
        x1 = self.conv2(self.conv1(x))
        x2 = self.conv4(self.conv3(x1))
        x3 = self.conv5(x2)
        return x3


class YOLO_HEAD(nn.Module):
    def __init__(self,in_chl,mid_chl,out_chl):
        super(YOLO_HEAD, self).__init__()
        self.conv1 = BaseConv(in_chl,mid_chl,ksize=3,stride=1)
        self.conv2 = BaseConv(mid_chl, out_chl, ksize=1, stride=1,with_bn=False,activate=None)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class YOLOV3(nn.Module):
    def __init__(self,num_calss):
        super(YOLOV3, self).__init__()
        self.darknet =  Darknet53()
        # upsample
        self.feature1 = YOLOBlock(1024,512)
        self.out_head1 = YOLO_HEAD(512, 1024, 3*(num_calss+1+4))

        self.cbl1 = self._make_cbl(512,256,1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.feature2  = YOLOBlock(768,256)
        self.out_head2 = YOLO_HEAD(256, 512, 3*(num_calss+1+4))

        self.cbl2 = self._make_cbl(256, 128, 1)
        self.feature3  = YOLOBlock(384, 128)
        self.out_head3 = YOLO_HEAD(128, 256, 3*(num_calss+1+4))

        self.num_class = num_calss

        # anchor set
        self.yolo_anchors = [[[10, 13], [16, 30], [33, 23]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[116, 90], [156, 198], [373, 326]]]
        self.yolo_strides = [8, 16, 32]
        self.strides = np.array(self.yolo_strides)
        self.anchors = torch.from_numpy((np.array(self.yolo_anchors).T / self.strides).T)

    def _make_cbl(self, _in, _out, ks):
        return BaseConv(_in, _out, ks, stride=1, activate="lrelu")

    def forward(self,x):
        # backbone
        out3,out4,out5 = self.darknet(x)
        # 13X13X255
        feature1 = self.feature1(out5)
        out_large = self.out_head1(feature1)
        # 26X26X255
        cb1 = self.cbl1(feature1)
        up1 = self.upsample(cb1)
        x1_in = torch.cat([up1, out4], 1)
        feature2 = self.feature2(x1_in)
        out_medium = self.out_head2(feature2)
        # 52X52X255
        cb2 = self.cbl2(feature2)
        up2 = self.upsample(cb2)
        x2_in = torch.cat([up2, out3], 1)
        feature3 = self.feature3(x2_in)
        out_small = self.out_head3(feature3)

        return (out_large,out_medium,out_small)


    def predict(self,out_large,out_medium,out_small):
        """
        param: out_large  [1,255,13,13]
        param: out_medium [1,255,26,26]
        param: out_small  [1,255,52,52]
        """

        # transpose
        trans_large = out_large.permute((0,2,3,1))       # [1,13,13,255]
        trans_medium = out_medium.permute((0, 2, 3, 1))  # [1,26,26,255]
        trans_small = out_small.permute((0, 2, 3, 1))    # [1,52,52,255]
        # decode
        pred_small = self.decode(trans_small,i=0)    # [1,52,52,3,85]
        pred_media = self.decode(trans_medium,i=1)   # [1,26,26,3,85]
        pred_large = self.decode(trans_large,i=2)    # [1,13,13,3,85]

        # out
        n = pred_small.shape[0]
        c = pred_small.shape[-1]
        out_small = pred_small.view(n,-1,c)
        out_media = pred_media.view(n,-1,c)
        out_large = pred_large.view(n, -1, c)

        out_pred = torch.concat([out_small,out_media,out_large],dim=1)

        return out_pred


    def decode(self,conv_layer,i=0):
        """
        param: conv_layer nXhXwX255
        """
        n,h,w,c = conv_layer.shape
        conv_output = conv_layer.view(n,h,w,3,5+self.num_class)
        # divide output
        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]  # offset of center position
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]  # Prediction box length and width offset
        conv_raw_conf = conv_output[:, :, :, :, 4:5]  # confidence of the prediction box
        conv_raw_prob = conv_output[:, :, :, :, 5:]   # category probability of the prediction box

        # grid to 13X13 26X26 52X52
        yv, xv = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        yv_new = yv.unsqueeze(dim=-1)
        xv_new = xv.unsqueeze(dim=-1)
        xy_grid = torch.concat([xv_new,yv_new],dim=-1)
        # reshape and repeat
        xy_grid = xy_grid.view(1,h,w,1,2)         # (13,13,2)-->(1,13,13,1,2)
        xy_grid = xy_grid.repeat(n,1,1,3,1).float() # (1,13,13,1,2)--> (1,13,13,3,2)

        # Calculate teh center position  and h&w of the prediction box
        pred_xy = (torch.sigmoid(conv_raw_dxdy) + xy_grid)* self.strides[i]
        pred_wh = (torch.exp(conv_raw_dwdh) * self.anchors[i]) * self.strides[i]
        pred_xywh = torch.concat([pred_xy,pred_wh],dim=-1)
        # score and cls
        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)

        return torch.concat([pred_xywh,pred_conf,pred_prob],dim=-1)


if __name__ == "__main__":
    x = torch.randn((1,3,416,416))
    model = YOLOV3(80)
    cnt = 0
    out = model(x)
    out_large,out_medium,out_small= model(x)
    print(out_large.shape)
    print(out_medium.shape)
    print(out_small.shape)
