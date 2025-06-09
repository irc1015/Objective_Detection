#version ==> 0.0.1-2025.6

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import uniform_

def _get_clones(module, n):
    '''
    Create a list of cloned modules from the given module.

    Args:
        module (nn.Module): The module to be cloned.
        n (int): Number of clones to create.

    Returns:
        (nn.ModuleList): A ModuleList containing n clones of the input module.

    Examples:
        >>> import torch.nn as nn
        >>> layer = nn.Linear(10, 10)
        >>> clones = _get_clones(layer, 3)
        >>> len(clones)
        3
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

def bias_init_with_prob(prior_prob=0.01):
    '''
    Initialize conv/fc bias value according to a given probability value.

    This function calculates the bias initialization value based on a prior probability using the inverse error function.
    It's commonly used in object detection models to initialize classification layers with a specific positive prediction
    probability.
    '''
    return float(-np.log(1 - prior_prob) / prior_prob)

def linear_init(module):
    '''
    Initialize the weights and biases of a linear module.

    This function initializes the weights of a linear module using a uniform distribution within bounds calculated
    from the input dimension. If the module has a bias, it is also initialized.

    conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
    print(conv.weight.shape)  # Output: torch.Size([16, 3, 3, 3])
    '''
    bound = 1 / math.sqrt(module.weight.shape[0])
    uniform_(module.weight, -bound, bound)
    if hasattr(module, 'bias') and module.bias is not None:
        uniform_(module.bias, -bound, bound)

def inverse_sigmoid(x, eps=1e-5):
    '''
    Calculate the inverse sigmoid function for a tensor.

    This function applies the inverse of the sigmoid function to a tensor, which is useful in various neural network
    operations, particularly in attention mechanisms and coordinate transformations.

    Args:
        x (torch.Tensor): Input tensor with values in range [0, 1].
        eps (float, optional): Small epsilon value to prevent numerical instability.

    Returns:
        (torch.Tensor): Tensor after applying the inverse sigmoid function.
    '''
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

def multi_scale_deformable_attn_pytorch(value: torch.Tensor,
                                        value_spatial_shapes: torch.Tensor,
                                        sampling_locations: torch.Tensor,
                                        attention_weights: torch.Tensor,) -> torch.Tensor:
    '''
    Implement multi-scale deformable attention in PyTorch.

    This function performs deformable attention across multiple feature map scales(value_spatial_shapes),
    allowing the model to attend todifferent spatial locations(sampling_locations) with learned offsets.

    Args:
        value (torch.Tensor): input feature maps across all scales. (bs, num_keys, num_heads, embed_dims).
            bs is the batch size,
            num_heads is the number of attention heads,
            embed_dims is the embedding dimension per head,
            num_keys is the sum of spatial elements (H_ * W_) across all scales.
        value_spatial_shapes (torch.Tensor): Spatial shapes of the value tensor with shape (num_levels, 2).
            (num_levels, (H_, W_))
        sampling_locations (torch.Tensor): The sampling locations with shape (bs, num_queries, num_heads, num_levels, num_points, 2).
            The last dimension (2) is for x and y coordinates.
        attention_weights (torch.Tensor): The attention weights with shape (bs, num_queries, num_heads, num_levels, num_points).
            Provides the weights for each sampled point, used to compute the weighted sum in the attention mechanism.


    The function returns a tensor of shape (bs, num_queries, num_heads * embed_dims), representing the attention output for each query across the batch.
    '''
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    #(bs, H_ * W_, num_heads, embed_dims) across num_keys
    sampling_grids = 2 * sampling_locations - 1
    '''
    The sampling_locations tensor contains coordinates in the range [0, 1], 
    but PyTorch’s F.grid_sample expects coordinates in the range [-1, 1]. The function scales them accordingly
    '''
    sampling_value_list = []

    for level, (H_, W_) in enumerate(value_spatial_shapes):

        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        '''
       - value_list[level]: Shape (bs, H_ * W_, num_heads, embed_dims)
       - .flatten(2): Flattens the spatial dimensions, shape (bs, H_ * W_, num_heads * embed_dims)
       - .transpose(1, 2): Swaps dimensions, shape (bs, num_heads * embed_dims, H_ * W_)
       - .reshape(bs * num_heads, embed_dims, H_, W_): Reshapes to a 4D tensor for F.grid_sample.
        '''

        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        '''
        - sampling_grids[:, :, :, level]: Shape (bs, num_queries, num_heads, num_points, 2)
        - .transpose(1, 2): Shape (bs, num_heads, num_queries, num_points, 2)
        - .flatten(0, 1): Shape (bs * num_heads, num_queries, num_points, 2).
        '''

        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
        '''
        - F.grid_sample performs bilinear interpolation to sample from value_l_ at the locations in sampling_grid_l_
        - Output shape: (bs * num_heads, embed_dims, num_queries, num_points).
        '''

    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    '''
    - Original shape: (bs, num_queries, num_heads, num_levels, num_points)
    - .transpose(1, 2): Shape (bs, num_heads, num_queries, num_levels, num_points)
    - Finally shape: (bs * num_heads, 1, num_queries, num_levels * num_points)
    '''

    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    '''
    - torch.stack(sampling_value_list, dim=-2): Shape (bs * num_heads, embed_dims, num_queries, num_levels, num_points).
    - .flatten(-2): Flattens the last two dimensions, shape (bs * num_heads, embed_dims, num_queries, num_levels * num_points).
        as same as the shape of attention_weights
    - *attention_weights: Element-wise multiplication with weights.
    - .sum(-1): Sums over the sampling points, shape (bs * num_heads, embed_dims, num_queries).
    - .view(bs, num_heads * embed_dims, num_queries): Reshapes for the final output.
    '''

    '''
    - .transpose(1, 2): Shape (bs, num_queries, num_heads * embed_dims)
    '''
    return output.transpose(1, 2).contiguous()
