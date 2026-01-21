# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

import math

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
    "SPDConv",
    "SPDConv_CA",
    "SmallObjectBlock",
    "DendriticConv2d",
    "PConv",
    "P3Shortcut",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
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
            if not hasattr(self, "id_tensor"):
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
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


class Index(nn.Module):
    """Returns a particular index of the input."""

    def __init__(self, c1, c2, index=0):
        """Returns a particular index of the input."""
        super().__init__()
        self.index = index

    def forward(self, x):
        """
        Forward pass.

        Expects a list of tensors as input.
        """
        return x[self.index]


class SPDConv(nn.Module):
    """
    Spatial-to-Depth Convolution (SPDConv) for downsampling.
    
    Preserves spatial information during downsampling by converting spatial dimensions
    to depth (channels). Better for small object detection compared to standard Conv downsampling.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        k (int): Kernel size (default: 3)
        s (int): Stride (default: 2, must be 2 for proper downsampling)
        p (int): Padding (auto-calculated if None)
        g (int): Groups (default: 1)
        act (bool): Activation (default: True)
    """

    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, act=True):
        """Initialize SPDConv module for spatial-to-depth downsampling."""
        super().__init__()
        if s != 2:
            raise ValueError(f"SPDConv stride must be 2, got {s}")
        
        # SPDConv: Split spatial -> concatenate to channels -> conv
        # Input: [B, C, H, W] -> Split into 4 parts -> [B, C*4, H/2, W/2] -> Conv to c2
        self.conv = Conv(c1 * 4, c2, k, 1, autopad(k, p), g, act=act)
        self.s = s

    def forward(self, x):
        """
        Forward pass: Split spatial dimensions into 4 parts and concatenate to channels.
        
        Input: [B, C, H, W]
        Output: [B, C2, H/2, W/2]
        """
        # Split spatial dimension into 4 parts (for stride=2 downsampling)
        # Equivalent to: top-left, top-right, bottom-left, bottom-right
        x = torch.cat(
            [x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1
        )
        # Apply convolution (stride=1, spatial size already reduced by split)
        return self.conv(x)


class SPDConv_CA(nn.Module):
    """
    Spatial-to-Depth Convolution with Channel Attention (SPDConv_CA).
    
    Combines SPDConv downsampling with Channel Attention for better feature selection.
    Preserves spatial information during downsampling while emphasizing important channels.
    Better for small object detection compared to standard Conv downsampling.
    
    Architecture:
    - SPDConv: Split spatial -> concatenate to channels -> conv
    - Channel Attention: Adaptive pooling -> FC -> Sigmoid -> Multiply
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        k (int): Kernel size (default: 3)
        s (int): Stride (default: 2, must be 2 for proper downsampling)
        p (int): Padding (auto-calculated if None)
        g (int): Groups (default: 1)
        act (bool): Activation (default: True)
    """
    
    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, act=True):
        """Initialize SPDConv_CA module."""
        super().__init__()
        if s != 2:
            raise ValueError(f"SPDConv_CA stride must be 2, got {s}")
        
        # SPDConv: Split spatial -> concatenate to channels -> conv
        # Input: [B, C, H, W] -> Split into 4 parts -> [B, C*4, H/2, W/2] -> Conv to c2
        self.conv = Conv(c1 * 4, c2, k, 1, autopad(k, p), g, act=act)
        
        # Channel Attention after convolution
        self.ca = ChannelAttention(c2)
        
        self.s = s
    
    def forward(self, x):
        """
        Forward pass: SPDConv + Channel Attention.
        
        Input: [B, C, H, W]
        Output: [B, C2, H/2, W/2]
        """
        # Split spatial dimension into 4 parts (for stride=2 downsampling)
        # Equivalent to: top-left, top-right, bottom-left, bottom-right
        x = torch.cat(
            [x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1
        )
        # Apply convolution (stride=1, spatial size already reduced by split)
        x = self.conv(x)
        # Apply channel attention
        x = self.ca(x)
        return x


class DendriticConv2d(nn.Module):
    """
    Dendritic Convolution: Multi-branch convolution with gating mechanism.
    
    Architecture:
    - Multiple parallel conv branches (B branches)
    - Each branch: Conv2d -> BN -> Activation
    - Gate network: Generates per-branch weights via pooling + MLP
    - Fuse: Weighted sum of branch outputs
    
    Drop-in replacement for Conv2d, especially effective for:
    - Noisy input (early backbone)
    - Feature fusion (neck)
    - Small object detection
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        k (int): Kernel size (default: 3)
        s (int): Stride (default: 1)
        p (int): Padding (auto-calculated if None)
        g (int): Groups (default: 1)
        branches (int): Number of parallel branches (default: 4)
        act (bool): Activation (default: True)
    """
    
    default_act = nn.SiLU()  # default activation
    
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, branches=4, act=True):
        """Initialize DendriticConv2d with multiple branches and gating."""
        super().__init__()
        self.branches = branches
        
        # Create B parallel branches: Conv -> BN -> Act
        self.branch_convs = nn.ModuleList()
        for _ in range(branches):
            branch = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(c2),
                self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
            )
            self.branch_convs.append(branch)
        
        # Gate network: Pooling -> MLP -> Sigmoid
        # Input: pooled features -> Output: (N, B, 1, 1) gate weights
        self.gate_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.gate_mlp = nn.Sequential(
            nn.Conv2d(c1, c1 // 4, 1, bias=False),  # Reduce channels
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // 4, branches, 1, bias=False),  # Output B gates
            nn.Sigmoid()  # Gate weights
        )
    
    def forward(self, x):
        """
        Forward pass through DendriticConv2d.
        
        Args:
            x: Input tensor [N, C1, H, W]
        
        Returns:
            Output tensor [N, C2, H', W'] where H', W' depend on stride
        """
        # Generate branch features
        branch_feats = [branch(x) for branch in self.branch_convs]  # List of [N, C2, H', W']
        
        # Generate gate weights: (N, B, 1, 1)
        pooled = self.gate_pool(x)  # [N, C1, 1, 1]
        gates = self.gate_mlp(pooled)  # [N, B, 1, 1]
        
        # Weighted sum: sum(gates[:, b] * branch_feats[b] for b in range(B))
        # gates[:, b] shape: [N, 1, 1, 1] -> broadcast to [N, C2, H', W']
        output = sum(gates[:, b:b+1, :, :] * branch_feats[b] for b in range(self.branches))
        
        return output


class PConv(nn.Module):
    """
    Partial Convolution (PConv) for selective feature extraction.
    
    Originally proposed for image inpainting, PConv applies convolution operations only
    over valid pixels in the input feature map, while maintaining identity mapping over
    invalid or irrelevant areas (occluded zones, background clutter, missing pixels).
    
    This design improves both the specificity and sparsity of feature representations,
    making it particularly effective for:
    - UAV imagery with occlusions, shadows, and motion blur
    - Road scenes with occlusions and image degradation
    - Small object detection in low-contrast regions
    - Ambiguous boundary regions where traditional convolution produces blurred responses
    
    Architecture:
    - Dynamic binary mask generation from input features
    - Mask-guided partial convolution (only valid regions)
    - Normalization to compensate for partial support
    - Optional identity-preserving for invalid regions
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        k (int): Kernel size (default: 3)
        s (int): Stride (default: 1)
        p (int): Padding (auto-calculated if None)
        g (int): Groups (default: 1)
        d (int): Dilation (default: 1)
        act (bool): Activation (default: True)
        use_identity (bool): Use identity mapping for invalid regions (default: True)
        mask_threshold (float): Threshold for dynamic mask generation (default: 0.1)
        eps (float): Epsilon for numerical stability (default: 1e-8)
    """
    
    default_act = nn.SiLU()  # default activation
    
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True, 
                 use_identity=True, mask_threshold=0.1, eps=1e-8):
        """Initialize PConv module."""
        super().__init__()
        self.use_identity = use_identity
        self.mask_threshold = mask_threshold
        self.eps = eps
        self.kernel_size = k
        self.stride = s
        self.padding = autopad(k, p, d) if p is None else p
        self.dilation = d
        
        # Standard convolution layer (will be masked during forward)
        # Use bias=False to match Conv pattern and avoid dtype mismatch in mixed precision
        self.conv = nn.Conv2d(c1, c2, k, s, self.padding, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        
        # Optional 1x1 projection for identity mapping when channel mismatch
        if use_identity and c1 != c2:
            self.identity_proj = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        else:
            self.identity_proj = None
    
    def build_mask_from_feature(self, x):
        """
        Generate dynamic binary mask from input features.
        
        Uses feature energy (mean absolute value) to identify valid regions.
        Regions with energy above threshold are considered valid.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            mask: Binary mask [B, 1, H, W] (1 = valid, 0 = invalid)
        """
        # Compute feature energy: mean absolute value across channels
        energy = torch.mean(torch.abs(x), dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Generate binary mask: 1 if energy > threshold, else 0
        # Use same dtype as input to avoid dtype mismatch
        threshold_tensor = torch.tensor(self.mask_threshold, dtype=x.dtype, device=x.device)
        mask = (energy > threshold_tensor).to(dtype=x.dtype)
        
        return mask
    
    def partial_conv2d(self, x, mask):
        """
        Apply partial convolution with mask guidance.
        
        Convolution is applied only to valid regions (where mask=1), with proper
        normalization to compensate for partial support.
        
        Args:
            x: Input tensor [B, C, H, W]
            mask: Binary mask [B, 1, H, W] (1 = valid, 0 = invalid)
            
        Returns:
            y: Output tensor [B, C2, H', W']
            mask_out: Updated mask [B, 1, H', W']
        """
        # Ensure mask has same dtype and device as input
        mask = mask.to(dtype=x.dtype, device=x.device)
        
        # Apply mask to input: zero out invalid regions
        x_masked = x * mask
        
        # Standard convolution on masked input
        y = self.conv(x_masked)
        
        # Compute valid pixel count for each output position
        # Use unfold to extract local windows from mask
        mask_padded = nn.functional.pad(mask, [self.padding] * 4, mode='constant', value=0)
        mask_unfold = nn.functional.unfold(
            mask_padded,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation
        )  # [B, k*k, H'*W']
        
        # Count valid pixels in each window
        valid_count = mask_unfold.sum(dim=1, keepdim=True)  # [B, 1, H'*W']
        valid_count = valid_count.view(y.shape[0], 1, y.shape[2], y.shape[3])  # [B, 1, H', W']
        
        # Normalization scale: kernel_size^2 / (valid_pixels + eps)
        # This compensates for partial support in the convolution window
        # Ensure scale has same dtype as y
        eps_tensor = torch.tensor(self.eps, dtype=y.dtype, device=y.device)
        scale = (self.kernel_size * self.kernel_size) / (valid_count.to(y.dtype) + eps_tensor)
        
        # Apply normalization
        y = y * scale
        
        # Update output mask: 1 if any valid pixels in window, else 0
        # Keep mask in same dtype as input for consistency
        mask_out = (valid_count > 0).to(dtype=x.dtype)
        
        return y, mask_out
    
    def forward(self, x, mask=None):
        """
        Forward pass through PConv.
        
        Args:
            x: Input tensor [B, C, H, W]
            mask: Optional external mask [B, 1, H, W]. If None, generated dynamically.
            
        Returns:
            Output tensor [B, C2, H', W']
        """
        # Generate mask if not provided
        if mask is None:
            mask = self.build_mask_from_feature(x)
        
        # Apply partial convolution
        y_pconv, mask_out = self.partial_conv2d(x, mask)
        
        # Apply batch normalization and activation
        y_pconv = self.act(self.bn(y_pconv))
        
        # Identity-preserving for invalid regions
        if self.use_identity:
            # Project input if channel mismatch
            if self.identity_proj is not None:
                x_proj = self.identity_proj(x)
            else:
                x_proj = x
            
            # Downsample/upsample x_proj to match output size if needed
            if x_proj.shape[2:] != y_pconv.shape[2:]:
                x_proj = nn.functional.interpolate(
                    x_proj, 
                    size=y_pconv.shape[2:], 
                    mode='nearest'
                )
            
            # Combine: PConv output for valid regions, identity for invalid regions
            y = y_pconv * mask_out + x_proj * (1 - mask_out)
        else:
            y = y_pconv
        
        return y


class P3Shortcut(nn.Module):
    """
    P3 Shortcut Connection for feature reuse from backbone to head.
    
    This module creates a lightweight shortcut from backbone P3 directly to Detect P3,
    preserving detailed information that might be lost during neck processing.
    
    Architecture:
    - Project backbone P3 with Conv1x1 to match head P3 channels
    - Element-wise addition with head P3 output
    - Preserves spatial details from backbone
    
    Args:
        c1 (int): Input channels from backbone P3
        c2 (int): Output channels (should match head P3 channels)
        act (bool): Activation (default: False, no activation for shortcut)
    """
    
    def __init__(self, c1, c2, act=False):
        """Initialize P3Shortcut module."""
        super().__init__()
        # Lightweight 1x1 projection to match channels
        self.proj = Conv(c1, c2, k=1, s=1, act=act)
    
    def forward(self, x):
        """
        Forward pass through P3Shortcut.
        
        Args:
            x: List of two tensors [backbone_p3, head_p3]
               - backbone_p3: [B, C1, H, W] from backbone P3
               - head_p3: [B, C2, H, W] from head P3
               
        Returns:
            Output tensor [B, C2, H, W] = head_p3 + proj(backbone_p3)
        """
        backbone_p3, head_p3 = x[0], x[1]
        
        # Project backbone P3 to match head P3 channels
        backbone_proj = self.proj(backbone_p3)
        
        # Element-wise addition
        output = head_p3 + backbone_proj
        
        return output