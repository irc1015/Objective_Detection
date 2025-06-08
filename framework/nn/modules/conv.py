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
    '''Generates more features with fewer parameters by using cheap operations.'''
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        '''
         Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        '''
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), dim=1)

class RepConv(nn.Module):
    '''
    RepConv module with training and deploy modes.
    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).
    '''
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        '''
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        '''
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p, g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        '''Forward pass for deploy mode'''
        return self.act(self.conv(x))

    def forward(self, x):
        '''Forward pass for training mode'''
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        '''
        Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): Equivalent kernel
            (torch.Tensor): Equivalent bias
        '''
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid


    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        '''
        (1, 1, 1, 1) means:
            Add 1 element to the left, Add 1 element to the right.
            Add 1 element to the top, Add 1 element to the bottom.
        If kernel1x1 has shape [H, W], the output shape after padding will be [H+2, W+2].
        If kernel1x1 is a 4D tensor (e.g., [C_in, C_out, H, W]), the last two dimensions (height and width) will be padded, resulting in [C_in, C_out, H+2, W+2].
        '''
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, (1, 1, 1, 1))

    def _fuse_bn_tensor(self, branch):
        '''
        Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            kernel (torch.Tensor): Fused kernel.
            bias (torch.Tensor): Fused bias.
        '''
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(branch, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        '''Fuse convolutions for inference by creating a single equivalent convolution.'''
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True,).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')

class ChannelAttention(nn.Module):
    '''
    Channel-attention module for feature recalibration.
    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    '''
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))

class SpatialAttention(nn.Module):
    '''
    Spatial-attention module for feature recalibration.
    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    '''
    def __init__(self, kernel_size=7):
        '''
        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        '''
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(self.cv1(torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)))

class CBAM(nn.Module):
    '''
    Convolutional Block Attention Module.
    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    '''
    def __init__(self, c1, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels=c1)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))

class Concat(nn.Module):
    '''
    Concatenate a list of tensors along specified dimension.
    '''
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x: List[torch.Tensor]):
        return torch.cat(x, self.d)

class Index(nn.Module):
    '''
    Returns a particular index of the input.
    '''
    def __init__(self, index=0):
        super().__init__()
        self.index = index

    def forward(self, x: List[torch.Tensor]):
        return x[self.index]