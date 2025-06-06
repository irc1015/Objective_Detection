#version ==> 0.0.1-2025.6
#Convolution modules

import math
from typing import List
import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "Index",
)

def autopad(k, p=None, d=1):
    '''
    :param k: kernel
    :param p: padding
    :param d: dilation
    :return: Pad to 'same' shape outputs
    '''
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    '''
    Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    '''
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        '''
        :param c1(int): Number of input channels.
        :param c2(int): Number of output channels.
        :param k(int): Kernel size.
        :param s(int): Stride size.
        :param p(int): Padding size.
        :param g(int): Group size.
        :param d(int): Dilation size.
        :param act(bool | nn.Module): Activation function.
        '''
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        #Apply convolution and activation without batch normalization
        return self.act(self.conv(x))

class Conv2(Conv):
    '''
    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    '''
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        '''
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        '''
        super().__init__()
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(k, p, d), groups=g, dilation=d, bias=False) # 1x1 conv

    def forward(self, x):
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        '''Fuse parallel convolutions'''
        w = torch.zeros_like(self.conv.weight.data)
        # [out_channels, in_channels, kernel_height, kernel_width] -> [out_channels, in_channels, 3, 3]
        i = [x // 2 for x in w.shape[2:]] # [1, 1]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__('cv2') #It is a delete relation operation
        self.forward = self.forward_fuse
        #switch a model from training mode to inference mode, skips the self.cv2

class LightConv(nn.Module):
    '''
    Light convolution module with 1x1 and depthwise convolutions.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    '''
    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        '''
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        '''
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        return self.conv2(self.conv1(x))

class DWConv(Conv):
    '''Depth-wise convolution module'''
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        '''
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        '''
        super().__init__(c1, c2, k, s, g=math.gcd(c1,c2), d=d, act=act)
        # math.gcd is greatest common divisor (GCD)
        # in_channels and out_channels must both be divisible by groups

class DWConvTranspose2d(nn.ConvTranspose2d):
    '''Depth-wise transposed convolution module'''
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        '''
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        '''
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))

class ConvTranspose(nn.Module):
    '''
    Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    '''
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        '''
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        '''
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        return self.act(self.conv_transpose(x))

class Focus(nn.Module):
    '''
    Focus module for concentrating feature information.
    Slices input tensor into 4 parts and concatenates them in the channel dimension.
    '''
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        '''
         Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        '''
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)

    def forward(self, x):
        '''
        Apply Focus operation and convolution to input tensor.
        input shape:(B, C, W, H) -> output shape:(B, 4C, W/2, H/2)

        example:[0, 1, 2, 3, 4, 5]
        ::2 is [0, 2, 4]
        1::2 is [1, 3, 5]
        '''
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), dim=1))

class GhostConv(nn.Module):



