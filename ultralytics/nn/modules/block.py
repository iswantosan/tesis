# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "TorchVision",
    "SmallObjectBlock",
    "RFCBAM",
    "DySample",
    "DPCB",
    "BFB",
    "SOFP",
    "HRDE",
    "MDA",
    "DSOB",
    "EAE",
    "CIB2",
    "TEB",
    "FDB",
    "SACB",
    "FBSB",
    "FDEB",
    "DPRB",
    "CoordinateAttention",
    "SimAM",
    "ConvNeXtBlock",
    "EdgePriorBlock",
    "LocalContextMixer",
    "TinyObjectAlignment",
    "AntiFPGate",
    "BackgroundSuppressionGate",
    "EdgeLineEnhancement",
    "AggressiveBackgroundSuppression",
    "CrossScaleSuppression",
    "MultiScaleEdgeEnhancement",
    "BGSuppressP3",
    "LocalContrastGate",
    "ShapeGate",
    "FSNeck",
    "DIFuse",
    "SADHead",
    "AdaptiveFeatureFusion",
    "NoiseSuppression",
    "GlobalContextBlock",
    "LargeKernelConv",
    "BiFPN",
    "EMA",
    "EMA_Bottleneck",
    "C3_EMA",
    "EMA_Plus",
    "C3_EMA_Enhanced",
    "CrossLevelAttention",
    "PANPlus",
    "ECA",
    "LightAttention",
    "SPDDown",
    "SPD_A_Block",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class EMA(nn.Module):
    """Efficient Multi-scale Attention for YOLO - Lightweight"""

    def __init__(self, channels, factor=32):
        super().__init__()
        self.groups = max(1, channels // factor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 1D convolution for channel interaction
        self.ca = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)

        # Group normalization for stability
        self.gn = nn.GroupNorm(self.groups, channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape

        # Residual
        identity = x

        # Group normalization (applied directly, GroupNorm handles grouping internally)
        x = self.gn(x)

        # Channel attention
        y = self.avg_pool(x)  # [b, c, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [b, 1, c]
        y = self.ca(y)  # 1D conv
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # [b, c, 1, 1]

        return identity * y.expand_as(x)


class EMA_Bottleneck(nn.Module):
    """Bottleneck with EMA Attention"""

    def __init__(self, c1, c2, shortcut=True, g=1):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        self.ema = EMA(c2)  # EMA Attention

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        out = self.ema(out)  # Apply EMA attention
        return x + out if self.add else out


class C3_EMA(nn.Module):
    """C3 with EMA Attention in bottleneck - Enhanced version of C3k2"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)

        # Convolution layers
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)

        # EMA Attention-enhanced bottlenecks
        self.m = nn.Sequential(
            *[EMA_Bottleneck(c_, c_, shortcut, g) for _ in range(n)]
        )

    def forward(self, x):
        return self.cv3(
            torch.cat(
                (self.m(self.cv1(x)), self.cv2(x)),
                dim=1
            )
        )


class EMA_Plus(nn.Module):
    """Enhanced EMA dengan channel + spatial attention"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        ca_avg = self.channel_attention(self.avg_pool(x))
        ca_max = self.channel_attention(self.max_pool(x))
        ca = self.sigmoid(ca_avg + ca_max)

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa_input)

        return x * ca * sa


class C3_EMA_Enhanced(nn.Module):
    """Enhanced C3 dengan EMA attention + spatial attention"""

    def __init__(self, c1, c2, n=3, shortcut=True):
        super().__init__()
        c_ = c2 // 2

        # Dual-branch dengan attention
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c1, c_, 1)

        # Bottleneck sequence dengan EMA
        self.m = nn.Sequential(*[
            nn.Sequential(
                Bottleneck(c_, c_, shortcut),
                EMA_Plus(c_)  # Enhanced EMA
            ) for _ in range(n)
        ])

        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        x1 = self.m(self.cv1(x))
        x2 = self.cv2(x)
        return self.cv3(torch.cat([x1, x2], dim=1))


class CrossLevelAttention(nn.Module):
    """Attention antar pyramid levels"""

    def __init__(self, channels):
        super().__init__()
        self.attention_gen = nn.Sequential(
            Conv(channels * 3, channels, 1),
            nn.ReLU(),
            Conv(channels, 3, 1),  # 3 weights untuk P3, P4, P5
            nn.Softmax(dim=1)
        )

    def forward(self, p3, p4, p5):
        # Resize semua ke ukuran P4 (middle size)
        p3_resized = F.interpolate(p3, size=p4.shape[-2:], mode='nearest')
        p5_resized = F.interpolate(p5, size=p4.shape[-2:], mode='nearest')

        # Concatenate untuk global context
        combined = torch.cat([p3_resized, p4, p5_resized], dim=1)

        # Generate attention weights
        weights = self.attention_gen(combined)  # [B, 3, H, W]
        w1, w2, w3 = weights.chunk(3, dim=1)  # Masing-masing untuk P3, P4, P5

        # Apply cross-level attention
        p3_att = p3 * F.interpolate(w1, size=p3.shape[-2:], mode='nearest')
        p4_att = p4 * w2
        p5_att = p5 * F.interpolate(w3, size=p5.shape[-2:], mode='nearest')

        # Residual addition
        p3 = p3 + p3_att
        p4 = p4 + p4_att
        p5 = p5 + p5_att

        return p3, p4, p5


class PANPlus(nn.Module):
    """
    Enhanced Path Aggregation Network++
    Expected gain: +1.2-1.8% mAP
    """

    def __init__(self, channels=[256, 512, 1024], num_classes=80, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.nl = 3  # number of detection layers (P3, P4, P5)

        # Channel alignment (bi-directional)
        # channels list is [P5, P4, P3] in order of input
        # But we need to map correctly: P5=channels[0], P4=channels[1], P3=channels[2]
        self.align_p5 = Conv(channels[0], 256, 1)  # P5: channels[0]
        self.align_p4 = Conv(channels[1], 256, 1)  # P4: channels[1]
        self.align_p3 = Conv(channels[2], 256, 1)  # P3: channels[2]

        # =================== TOP-DOWN PATH ===================
        # P5 → P4
        self.top_p5_to_p4 = nn.Sequential(
            Conv(256, 256, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        # P4 → P3
        self.top_p4_to_p3 = nn.Sequential(
            Conv(256, 256, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        # =================== BOTTOM-UP PATH ===================
        # P3 → P4
        self.bottom_p3_to_p4 = nn.Sequential(
            Conv(256, 256, 3, 2)
        )

        # P4 → P5
        self.bottom_p4_to_p5 = nn.Sequential(
            Conv(256, 256, 3, 2)
        )

        # =================== FEATURE REFINEMENT ===================
        # Enhanced C3 blocks dengan residual connections
        self.refine_p3 = C3_EMA_Enhanced(256, 256, n=3)
        self.refine_p4 = C3_EMA_Enhanced(256, 256, n=3)
        self.refine_p5 = C3_EMA_Enhanced(256, 256, n=3)

        # =================== DETECTION HEADS ===================
        # Separate head untuk setiap level (shared weights)
        c2_reg = max((16, 256 // 4, self.reg_max * 4))
        c2_cls = max(256, min(self.num_classes, 100))

        self.cls_head = nn.ModuleList([
            nn.Sequential(
                Conv(256, 256, 3),
                Conv(256, 256, 3),
                nn.Conv2d(256, num_classes, 1)
            ) for _ in range(3)
        ])

        self.reg_head = nn.ModuleList([
            nn.Sequential(
                Conv(256, 256, 3),
                Conv(256, 256, 3),
                nn.Conv2d(256, 4 * reg_max, 1)
            ) for _ in range(3)
        ])

        # =================== ATTENTION FUSION ===================
        # Cross-level attention untuk adaptive fusion
        self.cross_attention = CrossLevelAttention(256)

    def forward(self, p5, p4, p3):
        """
        Input order: [P5, P4, P3] (same as Detect head input order)
        p5: [batch, channels[0], H5, W5] - typically [B, 1024, 20, 20]
        p4: [batch, channels[1], H4, W4] - typically [B, 512, 40, 40]
        p3: [batch, channels[2], H3, W3] - typically [B, 256, 80, 80]
        """
        # 1. Channel alignment
        p5 = self.align_p5(p5)  # [B, 256, H5, W5]
        p4 = self.align_p4(p4)  # [B, 256, H4, W4]
        p3 = self.align_p3(p3)  # [B, 256, H3, W3]

        # =================== TOP-DOWN AGGREGATION ===================
        # P5 → P4 (with residual)
        p5_to_p4 = self.top_p5_to_p4(p5)
        p4 = p4 + p5_to_p4  # Enhanced P4

        # P4 → P3 (with residual)
        p4_to_p3 = self.top_p4_to_p3(p4)
        p3 = p3 + p4_to_p3  # Enhanced P3

        # =================== BOTTOM-UP AGGREGATION ===================
        # P3 → P4 (with residual)
        p3_to_p4 = self.bottom_p3_to_p4(p3)
        p4 = p4 + p3_to_p4  # Re-enhanced P4

        # P4 → P5 (with residual)
        p4_to_p5 = self.bottom_p4_to_p5(p4)
        p5 = p5 + p4_to_p5  # Re-enhanced P5

        # =================== CROSS-LEVEL ATTENTION ===================
        p3, p4, p5 = self.cross_attention(p3, p4, p5)

        # =================== FEATURE REFINEMENT ===================
        p3 = self.refine_p3(p3)
        p4 = self.refine_p4(p4)
        p5 = self.refine_p5(p5)

        # =================== DETECTION OUTPUTS ===================
        outputs = []
        for i, (p, cls_head, reg_head) in enumerate(zip(
            [p3, p4, p5], self.cls_head, self.reg_head
        )):
            cls = cls_head(p)
            reg = reg_head(p)
            # Cat: [reg, cls] seperti format YOLO Detect
            output = torch.cat([reg, cls], dim=1)
            outputs.append(output)

        return outputs


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """
    TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and customize the model by truncating or unwrapping layers.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.

    Args:
        c1 (int): Input channels.
        c2 (): Output channels.
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): If True, unwraps the model to a sequential containing all but the last `truncate` layers. Default is True.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.
    """

    def __init__(self, c1, c2, model, weights="DEFAULT", unwrap=True, truncate=2, split=False):
        """Load the model and weights from torchvision."""
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())[:-truncate]
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*layers)
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x):
        """Forward pass through the model."""
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y

import logging
logger = logging.getLogger(__name__)

USE_FLASH_ATTN = False
try:
    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
        from flash_attn.flash_attn_interface import flash_attn_func
        USE_FLASH_ATTN = True
    else:
        from torch.nn.functional import scaled_dot_product_attention as sdpa
        logger.warning("FlashAttention is not available on this device. Using scaled_dot_product_attention instead.")
except Exception:
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    logger.warning("FlashAttention is not available on this device. Using scaled_dot_product_attention instead.")

class AAttn(nn.Module):
    """
    Area-attention module with the requirement of flash attention.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1.

    Methods:
        forward: Performs a forward process of input tensor and outputs a tensor after the execution of the area attention mechanism.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import AAttn
        >>> model = AAttn(dim=64, num_heads=2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.

    """

    def __init__(self, dim, num_heads, area=1):
        """Initializes the area-attention module, a simple yet efficient attention module for YOLO."""
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qk = Conv(dim, all_head_dim * 2, 1, act=False)
        self.v = Conv(dim, all_head_dim, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)

        self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)


    def forward(self, x):
        """Processes the input tensor 'x' through the area-attention"""
        B, C, H, W = x.shape
        N = H * W

        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x)
        pp = self.pe(v)
        v = v.flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape
        q, k = qk.split([C, C], dim=2)

        if x.is_cuda and USE_FLASH_ATTN:
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_heads, self.head_dim)
            v = v.view(B, N, self.num_heads, self.head_dim)

            x = flash_attn_func(
                q.contiguous().half(),
                k.contiguous().half(),
                v.contiguous().half()
            ).to(q.dtype)
        else:
            q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)

            attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
            max_attn = attn.max(dim=-1, keepdim=True).values
            exp_attn = torch.exp(attn - max_attn)
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            x = (v @ attn.transpose(-2, -1))

            x = x.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return self.proj(x + pp)
    

class ABlock(nn.Module):
    """
    ABlock class implementing a Area-Attention block with effective feature extraction.

    This class encapsulates the functionality for applying multi-head attention with feature map are dividing into areas
    and feed-forward neural network layers.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        area (int, optional): Number of areas the feature map is divided.  Defaults to 1.

    Methods:
        forward: Performs a forward pass through the ABlock, applying area-attention and feed-forward layers.

    Examples:
        Create a ABlock and perform a forward pass
        >>> model = ABlock(dim=64, num_heads=2, mlp_ratio=1.2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        """Initializes the ABlock with area-attention and feed-forward layers for faster feature extraction."""
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using a truncated normal distribution."""
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Executes a forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class A2C2f(nn.Module):  
    """
    A2C2f module with residual enhanced feature extraction using ABlock blocks with area-attention. Also known as R-ELAN

    This class extends the C2f module by incorporating ABlock blocks for fast attention mechanisms and feature extraction.

    Attributes:
        c1 (int): Number of input channels;
        c2 (int): Number of output channels;
        n (int, optional): Number of 2xABlock modules to stack. Defaults to 1;
        a2 (bool, optional): Whether use area-attention. Defaults to True;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1;
        residual (bool, optional): Whether use the residual (with layer scale). Defaults to False;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        e (float, optional): Expansion ratio for R-ELAN modules. Defaults to 0.5;
        g (int, optional): Number of groups for grouped convolution. Defaults to 1;
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to True;

    Methods:
        forward: Performs a forward pass through the A2C2f module.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import A2C2f
        >>> model = A2C2f(c1=64, c2=64, n=2, a2=True, area=4, residual=True, e=0.5)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        # num_heads = c_ // 64 if c_ // 64 >= 2 else c_ // 32
        num_heads = c_ // 32

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)  # optional act=FReLU(c2)

        init_values = 0.01  # or smaller
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2))) if a2 else C3k(c_, c_, 2, shortcut, g) for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through R-ELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        if self.gamma is not None:
            return x + self.gamma.view(1, -1, 1, 1) * self.cv2(torch.cat(y, 1))
        return self.cv2(torch.cat(y, 1))


class SmallObjectBlock(nn.Module):
    """
    Advanced Small Object Detection Block optimized for BTA/AFB detection.
    
    Implements multi-scale feature enhancement, context-aware attention, and 
    residual connections specifically designed for very small object detection.
    
    Features:
    - Multi-scale feature extraction (1x1, 3x3, 5x5 convolutions)
    - Enhanced CBAM with residual connection
    - Feature fusion with learnable weights
    - Spatial detail preservation
    
    Args:
        c1 (int): Input channels (auto-inferred from previous layer)
        c2 (int): Output channels
        kernel_size (int): Kernel size for spatial attention (default: 7)
        use_residual (bool): Whether to use residual connection (default: True)
    """
    
    def __init__(self, c1, c2, kernel_size=7, use_residual=True):
        """Initialize SmallObjectBlock for enhanced small object detection."""
        super().__init__()
        from .conv import CBAM, Conv, ChannelAttention, SpatialAttention
        
        self.use_residual = use_residual and (c1 == c2)
        
        # Multi-scale feature extraction untuk capture berbagai ukuran small objects
        # 1x1 conv untuk global context
        self.conv1x1 = Conv(c1, c2 // 4, k=1, s=1)
        
        # 3x3 conv untuk local features (standard)
        self.conv3x3 = Conv(c1, c2 // 2, k=3, s=1, p=1)
        
        # 5x5 conv untuk larger receptive field (untuk context)
        self.conv5x5 = Conv(c1, c2 // 4, k=5, s=1, p=2)
        
        # Feature fusion dengan learnable weights (3 branches: 1x1, 3x3, 5x5)
        self.fusion_conv = Conv(c2, c2, k=1, s=1)
        
        # Enhanced attention mechanism
        self.channel_attn = ChannelAttention(c2)
        self.spatial_attn = SpatialAttention(kernel_size)
        
        # Feature refinement
        self.refine_conv = Conv(c2, c2, k=3, s=1, p=1)
        
        # Learnable fusion weight untuk 3 branches
        self.fusion_weight = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, x):
        """
        Forward pass through small object detection block.
        
        Process:
        1. Multi-scale feature extraction (1x1, 3x3, 5x5)
        2. Feature fusion with learnable weights
        3. Enhanced attention (channel + spatial)
        4. Feature refinement
        5. Residual connection (if applicable)
        """
        identity = x
        
        # Multi-scale feature extraction
        feat1x1 = self.conv1x1(x)  # Point-wise features
        feat3x3 = self.conv3x3(x)  # Local features
        feat5x5 = self.conv5x5(x)  # Larger context
        
        # Weighted fusion of multi-scale features (3 branches)
        # Normalize weights
        weights = torch.softmax(self.fusion_weight, dim=0)
        fused = torch.cat([
            feat1x1 * weights[0],
            feat3x3 * weights[1],
            feat5x5 * weights[2]
        ], dim=1)
        
        # Feature fusion
        x = self.fusion_conv(fused)
        
        # Enhanced attention mechanism (channel first, then spatial)
        x = self.channel_attn(x)  # Channel attention
        x = self.spatial_attn(x)  # Spatial attention
        
        # Feature refinement
        x = self.refine_conv(x)
        
        # Residual connection (preserve original features untuk small objects)
        if self.use_residual:
            x = x + identity
        
        return x


class DPCB(nn.Module):
    """
    Detail-Preserve Context Block (DPCB) for small object detection.
    
    Designed to add context without destroying detail information.
    Uses dilated depthwise convolution to increase receptive field without pooling.
    
    Architecture:
    - Conv 1×1 (reduce channel)
    - DWConv 3×3 (stride 1) → preserve detail
    - Dilated DWConv 3×3 (d=2) → wider context without pooling
    - Conv 1×1 (restore)
    - Skip connection (residual)
    
    Why suitable for small objects:
    - No maxpooling (preserves spatial detail)
    - Receptive field increases via dilation
    - Residual connection preserves original features
    
    Args:
        c1 (int): Input channels (auto-inferred from previous layer)
        c2 (int): Output channels
        reduction (float): Channel reduction factor for intermediate layers (default: 0.5)
        dilation (int): Dilation rate for dilated DWConv (default: 2)
        use_residual (bool): Whether to use residual connection (default: True)
    """
    
    def __init__(self, c1, c2, reduction=0.5, dilation=2, use_residual=True):
        """Initialize DPCB for detail-preserving context enhancement."""
        super().__init__()
        from .conv import Conv, DWConv
        
        self.use_residual = use_residual and (c1 == c2)
        
        # Calculate intermediate channels
        c_reduced = int(c2 * reduction)
        
        # Conv 1×1 (reduce channel)
        self.conv_reduce = Conv(c1, c_reduced, k=1, s=1, act=True)
        
        # DWConv 3×3 (stride 1) → preserve detail
        self.dwconv_detail = DWConv(c_reduced, c_reduced, k=3, s=1, d=1, act=True)
        
        # Dilated DWConv 3×3 (d=2) → wider context without pooling
        # Padding = dilation * (kernel_size - 1) / 2 = 2 * (3 - 1) / 2 = 2
        self.dwconv_context = DWConv(c_reduced, c_reduced, k=3, s=1, d=dilation, act=True)
        
        # Conv 1×1 (restore)
        self.conv_restore = Conv(c_reduced, c2, k=1, s=1, act=True)
        
    def forward(self, x):
        """
        Forward pass through DPCB.
        
        Process:
        1. Reduce channels (1×1 conv)
        2. Detail preservation (3×3 DWConv)
        3. Context expansion (dilated 3×3 DWConv)
        4. Restore channels (1×1 conv)
        5. Residual connection (if applicable)
        """
        identity = x
        
        # Reduce channels
        x = self.conv_reduce(x)
        
        # Detail preservation (local features)
        x = self.dwconv_detail(x)
        
        # Context expansion (wider receptive field without pooling)
        x = self.dwconv_context(x)
        
        # Restore channels
        x = self.conv_restore(x)
        
        # Residual connection (preserve original features)
        if self.use_residual:
            x = x + identity
        
        return x


class BFB(nn.Module):
    """
    Balanced Fusion Block (BFB) for neck architecture.
    
    Solves the problem where high-resolution features (P3) often get "dominated" 
    by features from lower layers in neck fusion.
    
    Architecture:
    - Split concatenated features into high-res and low-res branches
    - Conv1×1 on high-res branch
    - Conv1×1 on low-res branch
    - Concat both branches
    - DWConv3×3 + Conv1×1 (repeat 1-2x)
    - Residual connection
    
    Why suitable for small objects:
    - Balanced processing of high-res and low-res features
    - Prevents high-res features from being overwhelmed
    - Preserves detail from high-resolution branch
    - Efficient with depthwise convolution
    
    Args:
        c1 (int): Input channels (auto-inferred, should be 2x output after Concat)
        c2 (int): Output channels
        n (int): Number of DWConv3×3 + Conv1×1 repetitions (default: 2)
        use_residual (bool): Whether to use residual connection (default: True)
    """
    
    def __init__(self, c1, c2, n=2, use_residual=True):
        """Initialize BFB for balanced feature fusion in neck."""
        super().__init__()
        from .conv import Conv, DWConv
        
        self.use_residual = use_residual
        self.n = n
        
        # Split input channels: assume high-res and low-res are concatenated
        # Each branch gets half the channels
        c_half = c1 // 2
        
        # Conv1×1 on high-res branch
        self.conv_high = Conv(c_half, c_half, k=1, s=1, act=True)
        
        # Conv1×1 on low-res branch
        self.conv_low = Conv(c_half, c_half, k=1, s=1, act=True)
        
        # Fusion blocks: DWConv3×3 + Conv1×1 (repeat n times)
        self.fusion_blocks = nn.ModuleList()
        for _ in range(n):
            self.fusion_blocks.append(nn.Sequential(
                DWConv(c1, c1, k=3, s=1, d=1, act=True),  # DWConv3×3
                Conv(c1, c1, k=1, s=1, act=True)          # Conv1×1
            ))
        
        # Final output projection
        self.conv_out = Conv(c1, c2, k=1, s=1, act=True)
        
        # Residual connection only works if input and output channels match
        # After Concat, input is usually 2x output, so residual is disabled
        self.can_use_residual = use_residual and (c1 == c2)
        
    def forward(self, x):
        """
        Forward pass through BFB.
        
        Process:
        1. Split input into high-res and low-res branches (by channel)
        2. Process each branch with Conv1×1
        3. Concat both branches
        4. Apply DWConv3×3 + Conv1×1 (n times)
        5. Residual connection (if applicable)
        """
        identity = x
        
        # Split channels: first half = high-res, second half = low-res
        c = x.shape[1] // 2
        x_high = x[:, :c, :, :]  # High-res branch
        x_low = x[:, c:, :, :]   # Low-res branch
        
        # Process each branch with Conv1×1
        x_high = self.conv_high(x_high)
        x_low = self.conv_low(x_low)
        
        # Concat both branches
        x = torch.cat([x_high, x_low], dim=1)
        
        # Apply fusion blocks: DWConv3×3 + Conv1×1 (n times)
        for fusion_block in self.fusion_blocks:
            x = fusion_block(x)
        
        # Final output projection
        x = self.conv_out(x)
        
        # Residual connection (only if input and output channels match)
        if self.can_use_residual:
            x = x + identity
        
        return x


class EGB(nn.Module):
    """
    Edge-Gated Block (EGB) for enhanced edge/texture feature extraction.
    
    Helps the model better understand textures/edges (point/line objects).
    Uses a gating mechanism to selectively enhance edge-like features.
    
    Architecture:
    - DWConv 3×3 (extract edge-like features)
    - Conv 1×1 (process edge features)
    - Sigmoid gate (generate attention weights)
    - Multiply gate with input (selective enhancement)
    - Residual connection
    
    Formula: DW3 → Conv1x1 → Sigmoid → x*gate + x
    
    Why suitable for small objects:
    - Focuses on edge/texture information critical for small objects
    - Gating mechanism allows selective feature enhancement
    - Preserves original features via residual connection
    - Efficient with depthwise convolution
    
    Args:
        c1 (int): Input channels (auto-inferred from previous layer)
        c2 (int): Output channels
        use_residual (bool): Whether to use residual connection (default: True)
    """
    
    def __init__(self, c1, c2, use_residual=True):
        """Initialize EGB for edge/texture feature enhancement."""
        super().__init__()
        from .conv import Conv, DWConv
        
        self.use_residual = use_residual and (c1 == c2)
        
        # DWConv 3×3 (extract edge-like features)
        self.dwconv = DWConv(c1, c1, k=3, s=1, d=1, act=True)
        
        # Conv 1×1 (process edge features)
        self.conv = Conv(c1, c1, k=1, s=1, act=False)  # No activation before sigmoid
        
        # Sigmoid gate (will be applied in forward)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Forward pass through EGB.
        
        Process:
        1. Extract edge features (DWConv 3×3)
        2. Process edges (Conv 1×1)
        3. Generate gate weights (Sigmoid)
        4. Multiply gate with input (selective enhancement)
        5. Residual connection (preserve original features)
        """
        identity = x
        
        # DWConv 3×3 → Conv 1×1 → Sigmoid → gate
        gate = self.sigmoid(self.conv(self.dwconv(x)))
        
        # Multiply gate with input: x * gate
        x = x * gate
        
        # Residual connection: x * gate + x = x * (gate + 1)
        if self.use_residual:
            x = x + identity
        
        return x


class USF(nn.Module):
    """
    Upscale-Safe Fusion Block (USF) for neck architecture.
    
    Solves the problem where fine features from high-resolution layers (P3) 
    get "overwhelmed" by coarse features from lower layers during upsample+concat fusion.
    
    Architecture:
    - Conv 1×1 on high-res feature (normalize)
    - Conv 1×1 on low-res feature (balance)
    - Concat both processed features
    - 2× (DWConv 3×3 + Conv1×1) for lightweight fusion
    - Residual connection
    
    Flow: Conv1x1_hi → Conv1x1_lo → Concat → [DWConv3x3→Conv1x1]×2 + skip
    
    Why suitable for small objects:
    - Balances high-res and low-res features before fusion
    - Prevents fine features from being dominated
    - Preserves detail from high-resolution branch
    - Efficient with depthwise convolution
    
    Args:
        c1_high (int): Input channels from high-res branch (auto-inferred)
        c1_low (int): Input channels from low-res branch (auto-inferred)
        c2 (int): Output channels
        n (int): Number of DWConv3×3 + Conv1×1 repetitions (default: 2)
        use_residual (bool): Whether to use residual connection (default: True)
    """
    
    def __init__(self, c1_high, c1_low, c2, n=2, use_residual=True):
        """Initialize USF for upscale-safe feature fusion in neck."""
        super().__init__()
        from .conv import Conv, DWConv
        
        self.use_residual = use_residual
        self.n = n
        
        # Conv 1×1 on high-res branch (normalize)
        self.conv_high = Conv(c1_high, c1_high, k=1, s=1, act=True)
        
        # Conv 1×1 on low-res branch (balance)
        self.conv_low = Conv(c1_low, c1_low, k=1, s=1, act=True)
        
        # After concat, total channels = c1_high + c1_low
        c_concat = c1_high + c1_low
        
        # Fusion blocks: DWConv3×3 + Conv1×1 (repeat n times)
        self.fusion_blocks = nn.ModuleList()
        for _ in range(n):
            self.fusion_blocks.append(nn.Sequential(
                DWConv(c_concat, c_concat, k=3, s=1, d=1, act=True),  # DWConv3×3
                Conv(c_concat, c_concat, k=1, s=1, act=True)          # Conv1×1
            ))
        
        # Final output projection
        self.conv_out = Conv(c_concat, c2, k=1, s=1, act=True)
        
    def forward(self, x):
        """
        Forward pass through USF.
        
        Args:
            x: List of 2 tensors [x_high, x_low]
                x_high: High-resolution feature (already upsampled)
                x_low: Low-resolution feature (from backbone)
        
        Process:
        1. Normalize high-res feature with Conv1×1
        2. Balance low-res feature with Conv1×1
        3. Concat both processed features
        4. Apply DWConv3×3 + Conv1×1 (n times)
        5. Residual connection (if applicable)
        """
        if isinstance(x, (list, tuple)) and len(x) == 2:
            x_high, x_low = x
        else:
            raise ValueError(f"USF expects list of 2 tensors, got {type(x)}")
        
        # Normalize high-res feature
        x_high = self.conv_high(x_high)
        
        # Balance low-res feature
        x_low = self.conv_low(x_low)
        
        # Concat both processed features
        x = torch.cat([x_high, x_low], dim=1)
        identity = x
        
        # Apply fusion blocks: DWConv3×3 + Conv1×1 (n times)
        for fusion_block in self.fusion_blocks:
            x = fusion_block(x)
        
        # Residual connection: skip from after concat to after fusion blocks
        if self.use_residual:
            x = x + identity
        
        # Final output projection
        x = self.conv_out(x)
        
        return x


class MSA(nn.Module):
    """
    Multi-Scale Attention Block (MSA) for enhanced multi-scale feature extraction.
    
    Combines multi-scale convolution with spatial attention to capture features
    at different receptive fields simultaneously.
    
    Architecture:
    - Conv 1×1 (channel reduction)
    - Parallel branches with different dilations:
      * Conv 3×3 dilation=1 (local features)
      * Conv 3×3 dilation=2 (medium-range features)
      * Conv 3×3 dilation=3 (long-range features)
    - Concat all branches
    - Spatial Attention (Conv 7×7)
    - Conv 1×1 (channel fusion)
    
    Why suitable for small objects:
    - Multi-scale receptive fields capture objects at different scales
    - Spatial attention emphasizes important spatial locations
    - Parallel processing maintains efficiency
    - Effective for objects with varying sizes
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        reduction_ratio (int): Channel reduction ratio for parallel branches (default: 4)
                              Reduced channels = c1 // reduction_ratio
    """
    
    def __init__(self, c1, c2, reduction_ratio=4):
        """Initialize MSA for multi-scale feature extraction with spatial attention."""
        super().__init__()
        from .conv import Conv, SpatialAttention
        
        # Channel reduction for parallel branches
        c_reduced = max(c1 // reduction_ratio, 1)  # Ensure at least 1 channel
        
        # Conv 1×1 (channel reduction)
        self.conv_reduce = Conv(c1, c_reduced, k=1, s=1, act=True)
        
        # Parallel branches with different dilations
        # Branch 1: Conv 3×3 dilation=1 (local features)
        self.branch1 = Conv(c_reduced, c_reduced, k=3, s=1, d=1, act=True)
        
        # Branch 2: Conv 3×3 dilation=2 (medium-range features)
        self.branch2 = Conv(c_reduced, c_reduced, k=3, s=1, d=2, act=True)
        
        # Branch 3: Conv 3×3 dilation=3 (long-range features)
        self.branch3 = Conv(c_reduced, c_reduced, k=3, s=1, d=3, act=True)
        
        # After concat, total channels = 3 * c_reduced
        c_concat = 3 * c_reduced
        
        # Spatial Attention (Conv 7×7)
        self.spatial_attn = SpatialAttention(kernel_size=7)
        
        # Conv 1×1 (channel fusion to output channels)
        self.conv_fusion = Conv(c_concat, c2, k=1, s=1, act=True)
        
    def forward(self, x):
        """
        Forward pass through MSA.
        
        Process:
        1. Channel reduction with Conv1×1
        2. Process through 3 parallel branches (dilation 1, 2, 3)
        3. Concat all branches
        4. Apply spatial attention
        5. Channel fusion with Conv1×1
        """
        # Channel reduction
        x_reduced = self.conv_reduce(x)
        
        # Parallel branches
        branch1 = self.branch1(x_reduced)  # dilation=1
        branch2 = self.branch2(x_reduced)  # dilation=2
        branch3 = self.branch3(x_reduced)  # dilation=3
        
        # Concat all branches
        x = torch.cat([branch1, branch2, branch3], dim=1)
        
        # Spatial attention
        x = self.spatial_attn(x)
        
        # Channel fusion
        x = self.conv_fusion(x)
        
        return x


class SOE(nn.Module):
    """
    Small Object Enhancement Block (SOE) for fine-grained feature extraction.
    
    Focuses on fine-grained features critical for small object detection.
    
    Architecture:
    - Conv 1×1
    - DepthWise Conv 5×5 stride=1 padding=2
    - Channel Attention (Global AvgPool + FC)
    - Conv 3×3
    - Residual connection
    
    Why suitable for small objects:
    - DepthWise Conv 5×5 captures larger spatial context
    - Channel attention emphasizes important channels
    - Residual connection preserves original features
    - Focused on fine-grained details
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
    """
    
    def __init__(self, c1, c2):
        """Initialize SOE for small object enhancement."""
        super().__init__()
        from .conv import Conv, DWConv, ChannelAttention
        
        # Conv 1×1
        self.conv1 = Conv(c1, c1, k=1, s=1, act=True)
        
        # DepthWise Conv 5×5 stride=1 padding=2
        self.dwconv = DWConv(c1, c1, k=5, s=1, d=1, act=True)
        
        # Channel Attention (Global AvgPool + FC)
        self.channel_attn = ChannelAttention(c1)
        
        # Conv 3×3
        self.conv3 = Conv(c1, c2, k=3, s=1, act=True)
        
        # Residual connection (only if channels match)
        self.use_residual = (c1 == c2)
        
    def forward(self, x):
        """
        Forward pass through SOE.
        
        Process:
        1. Conv 1×1
        2. DepthWise Conv 5×5
        3. Channel Attention
        4. Conv 3×3
        5. Residual connection
        """
        identity = x
        
        # Conv 1×1
        x = self.conv1(x)
        
        # DepthWise Conv 5×5
        x = self.dwconv(x)
        
        # Channel Attention
        x = self.channel_attn(x)
        
        # Conv 3×3
        x = self.conv3(x)
        
        # Residual connection
        if self.use_residual:
            x = x + identity
        
        return x


class CA(nn.Module):
    """
    Context Aggregation Block (CA) for capturing context around small objects.
    
    Aggregates multi-scale contextual information using parallel pooling and convolution.
    
    Architecture:
    - Conv 1×1
    - Parallel branches:
      * MaxPool 3×3 stride=1 padding=1
      * AvgPool 3×3 stride=1 padding=1
      * Conv 3×3
    - Concat
    - Conv 1×1 (feature fusion)
    - SE/CBAM attention
    
    Why suitable for small objects:
    - Multiple pooling strategies capture different context types
    - Parallel processing maintains efficiency
    - Attention mechanism emphasizes important features
    - Effective for contextual understanding
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        use_cbam (bool): Use CBAM instead of ChannelAttention (default: False)
    """
    
    def __init__(self, c1, c2, use_cbam=False):
        """Initialize CA for context aggregation."""
        super().__init__()
        from .conv import Conv, ChannelAttention, CBAM
        
        # Conv 1×1
        self.conv1 = Conv(c1, c1, k=1, s=1, act=True)
        
        # Parallel branches
        # Branch 1: MaxPool 3×3 stride=1 padding=1
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        # Branch 2: AvgPool 3×3 stride=1 padding=1
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        
        # Branch 3: Conv 3×3
        self.conv3 = Conv(c1, c1, k=3, s=1, act=True)
        
        # After concat, total channels = 3 * c1
        c_concat = 3 * c1
        
        # Conv 1×1 (feature fusion)
        self.conv_fusion = Conv(c_concat, c2, k=1, s=1, act=True)
        
        # SE/CBAM attention
        if use_cbam:
            self.attention = CBAM(c2, kernel_size=7)
        else:
            self.attention = ChannelAttention(c2)
        
    def forward(self, x):
        """
        Forward pass through CA.
        
        Process:
        1. Conv 1×1
        2. Parallel branches (MaxPool, AvgPool, Conv3×3)
        3. Concat
        4. Conv 1×1 fusion
        5. Attention (SE/CBAM)
        """
        # Conv 1×1
        x = self.conv1(x)
        
        # Parallel branches
        branch1 = self.maxpool(x)  # MaxPool
        branch2 = self.avgpool(x)  # AvgPool
        branch3 = self.conv3(x)    # Conv 3×3
        
        # Concat
        x = torch.cat([branch1, branch2, branch3], dim=1)
        
        # Conv 1×1 fusion
        x = self.conv_fusion(x)
        
        # Attention
        x = self.attention(x)
        
        return x


class HRP(nn.Module):
    """
    High-Resolution Preservation Block (HRP) for preventing information loss in small objects.
    
    Maintains high-resolution features without pooling/downsampling operations.
    
    Architecture:
    - Conv 3×3 stride=1 (NO pooling/downsampling)
    - Dilated Conv 3×3 dilation=2
    - Conv 1×1
    - Skip connection
    
    Why suitable for small objects:
    - NO pooling preserves spatial resolution
    - Dilated convolution increases receptive field without downsampling
    - Skip connection preserves original information
    - Critical for small object detection where every pixel matters
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
    """
    
    def __init__(self, c1, c2):
        """Initialize HRP for high-resolution preservation."""
        super().__init__()
        from .conv import Conv
        
        # Conv 3×3 stride=1 (NO pooling/downsampling)
        self.conv3 = Conv(c1, c1, k=3, s=1, act=True)
        
        # Dilated Conv 3×3 dilation=2
        self.dilated_conv = Conv(c1, c1, k=3, s=1, d=2, act=True)
        
        # Conv 1×1
        self.conv1 = Conv(c1, c2, k=1, s=1, act=True)
        
        # Skip connection (only if channels match)
        self.use_residual = (c1 == c2)
        
    def forward(self, x):
        """
        Forward pass through HRP.
        
        Process:
        1. Conv 3×3 stride=1
        2. Dilated Conv 3×3 dilation=2
        3. Conv 1×1
        4. Skip connection
        """
        identity = x
        
        # Conv 3×3 stride=1
        x = self.conv3(x)
        
        # Dilated Conv 3×3 dilation=2
        x = self.dilated_conv(x)
        
        # Conv 1×1
        x = self.conv1(x)
        
        # Skip connection
        if self.use_residual:
            x = x + identity
        
        return x


class PFR(nn.Module):
    """
    Pyramid Feature Refinement Block (PFR) for refining P2 features with info from P3/P4.
    
    Explicitly refines P2 features using information from higher pyramid levels.
    
    Architecture:
    - Upsample P3 to P2 size (bilinear)
    - Upsample P4 to P2 size (bilinear)
    - Concat [P2, P3_up, P4_up]
    - Conv 1×1 -> reduce channels
    - Conv 3×3 -> Conv 3×3 (cascade refinement)
    - Spatial + Channel Attention
    - Conv 1×1 output
    
    Args:
        c2 (int): P2 channels
        c3 (int): P3 channels
        c4 (int): P4 channels
        c_out (int): Output channels
        reduction_ratio (int): Channel reduction ratio (default: 2)
    """
    
    def __init__(self, c2, c3, c4, c_out, reduction_ratio=2):
        """Initialize PFR for pyramid feature refinement."""
        super().__init__()
        from .conv import Conv, CBAM
        
        # After concat: c2 + c3 + c4 channels
        c_concat = c2 + c3 + c4
        
        # Conv 1×1 -> reduce channels
        c_reduced = max(c_concat // reduction_ratio, c_out)
        self.conv_reduce = Conv(c_concat, c_reduced, k=1, s=1, act=True)
        
        # Conv 3×3 -> Conv 3×3 (cascade refinement)
        self.conv3_1 = Conv(c_reduced, c_reduced, k=3, s=1, act=True)
        self.conv3_2 = Conv(c_reduced, c_reduced, k=3, s=1, act=True)
        
        # Spatial + Channel Attention (using CBAM)
        self.attention = CBAM(c_reduced, kernel_size=7)
        
        # Conv 1×1 output
        self.conv_out = Conv(c_reduced, c_out, k=1, s=1, act=True)
        
    def forward(self, x):
        """
        Forward pass through PFR.
        
        Args:
            x: List of 3 tensors [P2, P3, P4]
        """
        if isinstance(x, (list, tuple)) and len(x) == 3:
            p2, p3, p4 = x
        else:
            raise ValueError(f"PFR expects list of 3 tensors [P2, P3, P4], got {type(x)}")
        
        # Get P2 spatial size
        _, _, h2, w2 = p2.shape
        
        # Upsample P3 to P2 size (bilinear)
        p3_up = F.interpolate(p3, size=(h2, w2), mode='bilinear', align_corners=False)
        
        # Upsample P4 to P2 size (bilinear)
        p4_up = F.interpolate(p4, size=(h2, w2), mode='bilinear', align_corners=False)
        
        # Concat [P2, P3_up, P4_up]
        x = torch.cat([p2, p3_up, p4_up], dim=1)
        
        # Conv 1×1 -> reduce channels
        x = self.conv_reduce(x)
        
        # Conv 3×3 -> Conv 3×3 (cascade refinement)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        
        # Spatial + Channel Attention
        x = self.attention(x)
        
        # Conv 1×1 output
        x = self.conv_out(x)
        
        return x


class ARF(nn.Module):
    """
    Adaptive Receptive Field Block (ARF) for dynamic receptive field adjustment.
    
    Uses parallel multi-kernel depthwise convolutions with learnable weights.
    
    Architecture:
    - Conv 1×1
    - Parallel Multi-Kernel (depthwise):
      * Conv 3×3 groups=channels
      * Conv 5×5 groups=channels
      * Conv 7×7 groups=channels
    - Learn weights for each kernel (1×1 conv -> sigmoid)
    - Weighted sum of all kernels
    - Conv 1×1 projection
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
    """
    
    def __init__(self, c1, c2):
        """Initialize ARF for adaptive receptive field."""
        super().__init__()
        from .conv import DWConv, Conv
        
        # Conv 1×1
        self.conv1 = Conv(c1, c1, k=1, s=1, act=True)
        
        # Parallel Multi-Kernel (depthwise)
        self.dwconv3 = DWConv(c1, c1, k=3, s=1, d=1, act=True)  # 3×3
        self.dwconv5 = DWConv(c1, c1, k=5, s=1, d=1, act=True)  # 5×5
        self.dwconv7 = DWConv(c1, c1, k=7, s=1, d=1, act=True)  # 7×7
        
        # Learn weights for each kernel (1×1 conv -> sigmoid)
        self.weight_conv = Conv(c1, 3, k=1, s=1, act=False)  # Output 3 weights
        self.sigmoid = nn.Sigmoid()
        
        # Conv 1×1 projection
        self.conv_out = Conv(c1, c2, k=1, s=1, act=True)
        
    def forward(self, x):
        """
        Forward pass through ARF.
        
        Process:
        1. Conv 1×1
        2. Parallel multi-kernel depthwise convs
        3. Learn weights and weighted sum
        4. Conv 1×1 projection
        """
        # Conv 1×1
        x = self.conv1(x)
        
        # Parallel multi-kernel depthwise convs
        branch3 = self.dwconv3(x)  # 3×3
        branch5 = self.dwconv5(x)  # 5×5
        branch7 = self.dwconv7(x)  # 7×7
        
        # Learn weights for each kernel
        weights = self.sigmoid(self.weight_conv(x))  # [B, 3, H, W]
        w3 = weights[:, 0:1, :, :]  # [B, 1, H, W]
        w5 = weights[:, 1:2, :, :]
        w7 = weights[:, 2:3, :, :]
        
        # Weighted sum of all kernels
        x = w3 * branch3 + w5 * branch5 + w7 * branch7
        
        # Conv 1×1 projection
        x = self.conv_out(x)
        
        return x


class SOP(nn.Module):
    """
    Small Object Prior Block (SOP) for biasing network toward small objects using deformable conv.
    
    Uses deformable convolution with offset learning to focus on small object features.
    
    Architecture:
    - Deformable Conv 3×3 (offset learning)
    - Conv 3×3 standard
    - Element-wise multiply
    - Channel Shuffle
    - Conv 1×1
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
    """
    
    def __init__(self, c1, c2):
        """Initialize SOP for small object prior."""
        super().__init__()
        from .conv import Conv
        
        # Offset learning for deformable conv (simplified: only 2 offsets for x and y)
        self.offset_conv = Conv(c1, 2, k=3, s=1, act=False)  # Only 2 offsets (x, y) for simplicity
        
        # Standard Conv 3×3
        self.conv_standard = Conv(c1, c1, k=3, s=1, act=True)
        
        # Conv 1×1 for final output
        self.conv_out = Conv(c1, c2, k=1, s=1, act=True)
        
        # Channel shuffle groups
        self.channel_shuffle_groups = max(4, c1 // 64)  # Adaptive groups
        
    def forward(self, x):
        """
        Forward pass through SOP.
        
        Process:
        1. Generate offsets for deformable conv
        2. Apply deformable conv (using grid_sample)
        3. Standard conv 3×3
        4. Element-wise multiply
        5. Channel shuffle
        6. Conv 1×1
        """
        B, C, H, W = x.shape
        
        # Skip deformable conv if spatial size too small
        if min(H, W) < 3:
            # Fallback: just use standard conv
            x = self.conv_standard(x)
            x = self.conv_out(x)
            return x
        
        # Generate offsets (simplified: only x and y offsets)
        offsets = self.offset_conv(x)  # [B, 2, H, W]
        offsets = offsets * 0.1  # Scale offsets for stability
        
        # Create base grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, dtype=x.dtype, device=x.device),
            torch.arange(W, dtype=x.dtype, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([x_coords, y_coords], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]
        
        # Apply offsets
        grid = grid + offsets
        
        # Normalize grid to [-1, 1] (handle edge case when H or W = 1)
        grid = grid.permute(0, 2, 3, 1)  # [B, H, W, 2]
        if W > 1:
            grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / (W - 1) - 1.0
        else:
            grid[:, :, :, 0] = 0.0
        if H > 1:
            grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / (H - 1) - 1.0
        else:
            grid[:, :, :, 1] = 0.0
        
        # Apply deformable sampling
        x_deform = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        # Standard Conv 3×3
        x_standard = self.conv_standard(x)
        
        # Element-wise multiply
        x = x_deform * x_standard
        
        # Channel Shuffle (only if channels divisible by groups)
        groups = self.channel_shuffle_groups
        if C % groups == 0 and groups > 1:
            n, c, h, w = x.shape
            x = x.view(n, groups, c // groups, h, w)
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = x.view(n, c, h, w)
        
        # Conv 1×1
        x = self.conv_out(x)
        
        return x


class FA(nn.Module):
    """
    Feature Amplification Block (FA) for amplifying small object signals.
    
    Amplifies local detail features using global context information.
    
    Architecture:
    - Global Context (GAP + Conv 1×1)
    - Local Detail (Conv 3×3)
    - Difference = abs(Local - Global_broadcast)
    - Gate = sigmoid(Conv 1×1(Difference))
    - Output = Local * (1 + Gate)
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
    """
    
    def __init__(self, c1, c2):
        """Initialize FA for feature amplification."""
        super().__init__()
        from .conv import Conv
        
        # Global Context (GAP + Conv 1×1 without BN to avoid 1x1 BatchNorm error)
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        # Use simple conv without BN for 1x1 input to avoid BatchNorm error
        self.global_conv = nn.Conv2d(c1, c1, kernel_size=1, stride=1, bias=True)
        self.global_act = nn.SiLU()
        nn.init.kaiming_normal_(self.global_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.global_conv.bias, 0)
        
        # Local Detail (Conv 3×3)
        self.local_conv = Conv(c1, c1, k=3, s=1, act=True)
        
        # Gate generation
        self.gate_conv = Conv(c1, c1, k=1, s=1, act=False)
        self.sigmoid = nn.Sigmoid()
        
        # Final output conv
        self.conv_out = Conv(c1, c2, k=1, s=1, act=True)
        
    def forward(self, x):
        """
        Forward pass through FA.
        
        Process:
        1. Global Context (GAP + Conv 1×1)
        2. Local Detail (Conv 3×3)
        3. Difference = abs(Local - Global_broadcast)
        4. Gate = sigmoid(Conv 1×1(Difference))
        5. Output = Local * (1 + Gate)
        """
        # Global Context (GAP + Conv 1×1)
        global_feat = self.gap(x)  # [B, C, 1, 1]
        global_feat = self.global_act(self.global_conv(global_feat))  # [B, C, 1, 1] - no BN to avoid error
        
        # Local Detail (Conv 3×3)
        local_feat = self.local_conv(x)  # [B, C, H, W]
        
        # Broadcast global to local spatial size
        global_broadcast = global_feat.expand_as(local_feat)  # [B, C, H, W]
        
        # Difference = abs(Local - Global_broadcast)
        diff = torch.abs(local_feat - global_broadcast)
        
        # Gate = sigmoid(Conv 1×1(Difference))
        gate = self.sigmoid(self.gate_conv(diff))
        
        # Output = Local * (1 + Gate)
        x = local_feat * (1 + gate)
        
        # Final conv
        x = self.conv_out(x)
        
        return x


class ASFF(nn.Module):
    """
    Adaptive Spatial Feature Fusion (ASFF) Block for multi-scale feature fusion.
    
    Allows each scale to "borrow" information from other scales adaptively.
    Focuses on small objects by allowing P3 to use details from P2.
    
    Architecture (for P3 output):
    - Resize P2 to P3 size (downsample)
    - Keep P3 original
    - Resize P4 to P3 size (upsample)
    - Concat all aligned features
    - 1×1 gate (generate adaptive weights per-pixel)
    - Weighted sum
    - Conv for final fusion
    
    Args:
        c2 (int): P2 channels
        c3 (int): P3 channels
        c4 (int): P4 channels
        c_out (int): Output channels (default: same as P3)
        target_scale (str): Target scale for output ('P2', 'P3', or 'P4', default: 'P3')
    """
    
    def __init__(self, c2, c3, c4, c_out=None, target_scale='P3'):
        """Initialize ASFF for adaptive spatial feature fusion."""
        super().__init__()
        from .conv import Conv
        
        self.target_scale = target_scale
        
        # Determine output channels
        if c_out is None:
            c_out = c3  # Default to P3 channels
        
        # Align channels: all features should have same channels for fusion
        # Use P3 channels as reference
        self.align_p2 = Conv(c2, c3, k=1, s=1, act=True) if c2 != c3 else nn.Identity()
        self.align_p3 = nn.Identity()  # P3 stays as is
        self.align_p4 = Conv(c4, c3, k=1, s=1, act=True) if c4 != c3 else nn.Identity()
        
        # After concat: 3 * c3 channels
        c_concat = 3 * c3
        
        # 1×1 gate (generate adaptive weights per-pixel)
        # Output 3 weights (one for each scale: P2, P3, P4)
        self.gate_conv = Conv(c_concat, 3, k=1, s=1, act=False)
        self.softmax = nn.Softmax(dim=1)
        
        # Final conv for fusion
        self.conv_out = Conv(c3, c_out, k=1, s=1, act=True)
        
    def forward(self, x):
        """
        Forward pass through ASFF.
        
        Args:
            x: List of 3 tensors [P2, P3, P4]
        
        Process:
        1. Resize P2 and P4 to P3 size
        2. Align channels
        3. Concat
        4. Generate adaptive weights (gate)
        5. Weighted sum
        6. Final conv
        """
        if isinstance(x, (list, tuple)) and len(x) == 3:
            p2, p3, p4 = x
        else:
            raise ValueError(f"ASFF expects list of 3 tensors [P2, P3, P4], got {type(x)}")
        
        # Get target spatial size (P3 size)
        _, _, h3, w3 = p3.shape
        
        # Resize P2 to P3 size (downsample if needed)
        if p2.shape[2:] != (h3, w3):
            p2_resized = F.interpolate(p2, size=(h3, w3), mode='bilinear', align_corners=False)
        else:
            p2_resized = p2
        
        # P3 stays original
        p3_original = p3
        
        # Resize P4 to P3 size (upsample)
        if p4.shape[2:] != (h3, w3):
            p4_resized = F.interpolate(p4, size=(h3, w3), mode='bilinear', align_corners=False)
        else:
            p4_resized = p4
        
        # Align channels (do this after resize to ensure same channels)
        p2_aligned = self.align_p2(p2_resized)
        p3_aligned = p3_original
        p4_aligned = self.align_p4(p4_resized)
        
        # Final safety check: ensure all have same spatial size before concat
        # This handles edge cases where Conv might have changed size (shouldn't happen with 1x1, but safety first)
        target_size = p3_aligned.shape[2:]
        if p2_aligned.shape[2:] != target_size:
            p2_aligned = F.interpolate(p2_aligned, size=target_size, mode='bilinear', align_corners=False)
        if p4_aligned.shape[2:] != target_size:
            p4_aligned = F.interpolate(p4_aligned, size=target_size, mode='bilinear', align_corners=False)
        
        # Concat all aligned features
        x_concat = torch.cat([p2_aligned, p3_aligned, p4_aligned], dim=1)  # [B, 3*C, H, W]
        
        # 1×1 gate (generate adaptive weights per-pixel)
        weights = self.gate_conv(x_concat)  # [B, 3, H, W]
        weights = self.softmax(weights)  # [B, 3, H, W] - normalized weights
        
        # Weighted sum: w1*P2 + w2*P3 + w3*P4
        w1 = weights[:, 0:1, :, :]  # [B, 1, H, W]
        w2 = weights[:, 1:2, :, :]
        w3 = weights[:, 2:3, :, :]
        
        x = w1 * p2_aligned + w2 * p3_aligned + w3 * p4_aligned
        
        # Final conv for fusion
        x = self.conv_out(x)
        
        return x


class RFCBAM(nn.Module):
    """
    Receptive Field Channel and Spatial Attention Module (RFCBAM) for SOD-YOLO.
    
    RFCBAM uses receptive field attention mechanisms to perform downsampling operations.
    Unlike conventional convolution, RFCBAM focuses on local receptive field characteristics
    and uses attention to evaluate the significance of each feature point, mitigating
    information dilution caused by parameter sharing.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        k (int): Kernel size (default: 3)
        s (int): Stride (1 for same size, 2 for downsampling)
        g (int): Groups for group convolution (default: None, auto-calculate)
    """
    
    def __init__(self, c1, c2, k=3, s=1, g=None):
        """Initialize RFCBAM module for receptive field attention-based convolution."""
        super().__init__()
        from .conv import Conv, ChannelAttention, SpatialAttention
        
        self.stride = s
        if g is None:
            g = max(1, c1 // 4)  # Group convolution dengan reasonable groups
        
        # Stage 1: 3x3 group convolution untuk extract receptive field features
        self.group_conv = nn.Conv2d(c1, c1, k, s, k//2, groups=g, bias=False)
        self.group_bn = nn.BatchNorm2d(c1)
        
        # Dimensional transformation: expand receptive field features by factor of 3x3 = 9
        # Each spatial location gets expanded to 9 features (3x3 receptive field)
        # Use simple conv without BN untuk avoid BatchNorm error pada small feature maps
        self.expand_conv = nn.Conv2d(c1, c1 * 9, kernel_size=1, stride=1, bias=True)
        
        # Spatial attention dengan mean pooling dan max pooling
        self.spatial_attn = SpatialAttention(kernel_size=7)
        
        # Channel attention dengan SE (Squeeze-and-Excitation)
        self.channel_attn = ChannelAttention(c1)
        
        # Final 3x3 convolution untuk resize
        # Jika stride=1, final stride=3 untuk maintain size; jika stride=2, final stride=2 untuk downsampling
        final_stride = 3 if s == 1 else s
        # Use simple conv without BN untuk avoid BatchNorm error pada small feature maps
        self.final_conv = nn.Conv2d(c1, c2, kernel_size=3, stride=final_stride, padding=1, bias=True)
        
    def forward(self, x):
        """
        Forward pass through RFCBAM module.
        
        Process:
        1. Group convolution untuk extract receptive field features
        2. Dimensional transformation (expand receptive field by 3x3)
        3. Spatial attention (mean + max pooling)
        4. Channel attention (SE)
        5. Fuse attention dengan element-wise multiplication
        6. Final convolution untuk resize
        """
        B, C, H, W = x.shape
        min_spatial_size = min(H, W)
        
        # Stage 1: Group convolution untuk extract receptive field features
        x = self.group_conv(x)
        
        # Skip BatchNorm jika spatial size terlalu kecil (untuk menghindari BatchNorm error)
        if min_spatial_size > 1:
            x = self.group_bn(x)
            x = F.relu(x)
        else:
            # Fallback: skip BN untuk very small feature maps
            x = F.relu(x)
        
        # Dimensional transformation: expand receptive field by 3x3
        # Skip expansion jika spatial size terlalu kecil
        if min_spatial_size > 1:
            x_expanded = self.expand_conv(x)  # [B, C*9, H, W]
            x_expanded = F.relu(x_expanded)
            
            # Reshape untuk create expanded receptive field feature map
            # Split into 9 parts (representing 3x3 receptive field positions)
            x_expanded = x_expanded.view(B, C, 9, H, W)
            
            # Aggregate expanded features (simplified: use mean, full impl would rearrange spatially)
            x_expanded = x_expanded.mean(dim=2)  # [B, C, H, W]
        else:
            # Fallback: skip expansion untuk very small feature maps
            x_expanded = x
        
        # Spatial attention dengan mean pooling dan max pooling
        # Skip spatial attention jika spatial size terlalu kecil
        if min_spatial_size > 1:
            x_spatial = self.spatial_attn(x_expanded)
        else:
            x_spatial = x_expanded
        
        # Channel attention dengan SE (Squeeze-and-Excitation) - selalu bisa digunakan
        x_channel = self.channel_attn(x_expanded)
        
        # Fuse attention dengan element-wise multiplication
        x = x_spatial * x_channel
        
        # Final 3x3 convolution untuk resize
        x = self.final_conv(x)
        x = F.relu(x)
        
        return x


class DySample(nn.Module):
    """
    Dynamic Upsampling Module (DySample) for SOD-YOLO.
    
    DySample adaptively generates upsampling coordinates based on input feature map content.
    Unlike traditional upsampling with fixed interpolation rules, DySample generates
    a distribution of sampling points based on input feature content.
    
    Args:
        channels (int): Input channels (untuk generate offset)
        scale_factor (int): Upsampling scale factor (default: 2)
    """
    
    def __init__(self, channels, scale_factor=2):
        """Initialize DySample module for dynamic upsampling."""
        super().__init__()
        self.scale_factor = scale_factor
        
        # Linear layer untuk transform feature map dan generate offset
        self.offset_conv = nn.Conv2d(channels, 2, 1, 1, 0, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
    def forward(self, x):
        """
        Forward pass through DySample module.
        
        Process:
        1. Generate offset map dari input feature
        2. Create base grid coordinates
        3. Add offset to grid untuk dynamic sampling
        4. Grid sampling berdasarkan calculated coordinates
        """
        B, C, H, W = x.shape
        
        # Generate offset map dari input feature
        offset = self.offset_conv(x)  # [B, 2, H, W]
        offset = offset * 0.1  # Scale offset untuk stability
        
        # Create base grid coordinates [-1, 1]
        grid_h = torch.linspace(-1, 1, H * self.scale_factor, device=x.device, dtype=x.dtype)
        grid_w = torch.linspace(-1, 1, W * self.scale_factor, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(grid_h, grid_w, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H*scale, W*scale]
        
        # Upsample offset to target size
        offset_upsampled = F.interpolate(offset, size=(H * self.scale_factor, W * self.scale_factor), 
                                         mode='bilinear', align_corners=False)
        
        # Add offset to grid untuk dynamic sampling coordinates
        grid = grid + offset_upsampled
        
        # Grid sampling berdasarkan calculated coordinates
        grid = grid.permute(0, 2, 3, 1)  # [B, H*scale, W*scale, 2]
        x_upsampled = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        return x_upsampled


class SOFP(nn.Module):
    """
    Small Object Feature Pyramid Block for enhanced small object detection.
    
    Architecture:
    - Conv 3x3 stride=1
    - Conv 3x3 stride=1
    - Upsample 2x (bilinear)
    - Conv 1x1
    - Concat with backbone P2
    - Conv 3x3 stride=1
    
    Args:
        c1 (int): Input channels (from neck)
        c2 (int): Output channels
        c_p2 (int): Backbone P2 channels (for concatenation)
    """
    
    def __init__(self, c1, c2, c_p2):
        """Initialize SOFP for small object feature pyramid."""
        super().__init__()
        from .conv import Conv
        
        # Conv 3x3 stride=1
        self.conv1 = Conv(c1, c1, k=3, s=1, act=True)
        # Conv 3x3 stride=1
        self.conv2 = Conv(c1, c1, k=3, s=1, act=True)
        # Upsample 2x (bilinear)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Conv 1x1 to reduce to c_p2 channels
        self.conv_reduce = Conv(c1, c_p2, k=1, s=1, act=True)
        # Channel alignment for P2 - create with max possible channels to handle width scaling
        # We'll use a flexible approach: create alignment that can handle variable input
        # Store expected c_p2, but allow dynamic adjustment
        self.c_p2 = c_p2
        # Create a module dict for dynamic channel alignment (will be populated in forward if needed)
        self.conv_p2_align_dict = nn.ModuleDict()
        # Conv 3x3 stride=1 (after concat, input will be c_p2*2)
        # We'll handle this dynamically in forward to account for actual channel counts
        self.conv_out_dict = nn.ModuleDict()
        self.c2 = c2
        
    def forward(self, x):
        """
        Forward pass through SOFP.
        
        Args:
            x: List of 2 tensors [neck_output, p2_backbone]
        
        Returns:
            Enhanced feature for P2 detection head
        """
        if isinstance(x, (list, tuple)) and len(x) == 2:
            neck_out, p2 = x
        else:
            raise ValueError(f"SOFP expects list of 2 tensors [neck_output, p2_backbone], got {type(x)}")
        
        # Process through conv layers
        x = self.conv1(neck_out)
        x = self.conv2(x)
        # Upsample 2x
        x = self.upsample(x)
        # Reduce channels to match expected c_p2
        x = self.conv_reduce(x)
        
        # Align P2 channels if needed (in case actual P2 channels differ from expected c_p2)
        _, c_p2_actual, _, _ = p2.shape
        if c_p2_actual != self.c_p2:
            # Create or get alignment conv for this channel count
            align_key = str(c_p2_actual)
            if align_key not in self.conv_p2_align_dict:
                from .conv import Conv
                self.conv_p2_align_dict[align_key] = Conv(c_p2_actual, self.c_p2, k=1, s=1, act=True)
            p2 = self.conv_p2_align_dict[align_key](p2)
        
        # Concat with backbone P2
        x = torch.cat([x, p2], dim=1)
        
        # Get actual concat channel count and create/get appropriate conv_out
        _, c_concat, _, _ = x.shape
        concat_key = str(c_concat)
        if concat_key not in self.conv_out_dict:
            from .conv import Conv
            self.conv_out_dict[concat_key] = Conv(c_concat, self.c2, k=3, s=1, act=True)
        
        # Final conv
        x = self.conv_out_dict[concat_key](x)
        return x


class HRDE(nn.Module):
    """
    High-Res Detail Extractor Block for preserving fine details in P2 stage.
    
    Architecture:
    - Conv 1x1 (reduce channel)
    - DepthwiseConv 5x5 stride=1 padding=2
    - Conv 1x1 (expand channel)
    - Add (residual)
    - GELU activation
    - Conv 3x3 stride=1
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        reduction (float): Channel reduction ratio (default: 0.5)
    """
    
    def __init__(self, c1, c2, reduction=0.5):
        """Initialize HRDE for high-resolution detail extraction."""
        super().__init__()
        from .conv import Conv, DWConv
        
        c_reduced = int(c2 * reduction)
        self.use_residual = (c1 == c2)
        
        # Conv 1x1 (reduce channel)
        self.conv_reduce = Conv(c1, c_reduced, k=1, s=1, act=True)
        # DepthwiseConv 5x5 stride=1 padding=2
        self.dwconv = DWConv(c_reduced, c_reduced, k=5, s=1, d=1, act=True)
        # Conv 1x1 (expand channel)
        self.conv_expand = Conv(c_reduced, c2, k=1, s=1, act=False)  # No act before residual
        # GELU activation
        self.gelu = nn.GELU()
        # Conv 3x3 stride=1
        self.conv_out = Conv(c2, c2, k=3, s=1, act=True)
        
    def forward(self, x):
        """Forward pass through HRDE."""
        identity = x
        # Reduce channel
        x = self.conv_reduce(x)
        # Depthwise conv
        x = self.dwconv(x)
        # Expand channel
        x = self.conv_expand(x)
        # Add residual
        if self.use_residual:
            x = x + identity
        # GELU activation
        x = self.gelu(x)
        # Final conv
        x = self.conv_out(x)
        return x


class MDA(nn.Module):
    """
    Multi-Dilation Aggregator Block for multi-scale feature aggregation.
    
    Architecture:
    - Conv 1x1
    - Branch parallel:
      - Conv 3x3 dilation=1
      - Conv 3x3 dilation=2
      - Conv 3x3 dilation=3
    - Concat all branches
    - Conv 1x1 (channel reduction)
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        reduction_ratio (int): Channel reduction ratio for parallel branches (default: 4)
    """
    
    def __init__(self, c1, c2, reduction_ratio=4):
        """Initialize MDA for multi-dilation aggregation."""
        super().__init__()
        from .conv import Conv
        
        c_reduced = max(c1 // reduction_ratio, 1)
        c_concat = c_reduced * 3
        
        # Conv 1x1
        self.conv_reduce = Conv(c1, c_reduced, k=1, s=1, act=True)
        # Parallel branches with different dilations
        self.branch1 = Conv(c_reduced, c_reduced, k=3, s=1, d=1, act=True)  # dilation=1
        self.branch2 = Conv(c_reduced, c_reduced, k=3, s=1, d=2, act=True)  # dilation=2
        self.branch3 = Conv(c_reduced, c_reduced, k=3, s=1, d=3, act=True)  # dilation=3
        # Conv 1x1 (channel reduction)
        self.conv_out = Conv(c_concat, c2, k=1, s=1, act=True)
        
    def forward(self, x):
        """Forward pass through MDA."""
        # Reduce channels
        x = self.conv_reduce(x)
        # Parallel branches
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        # Concat all branches
        x = torch.cat([branch1, branch2, branch3], dim=1)
        # Channel reduction
        x = self.conv_out(x)
        return x


class DSOB(nn.Module):
    """
    Dense Small Object Block for dense feature extraction in P2 head.
    
    Architecture:
    - Conv 3x3 stride=1 → f1
    - Conv 3x3 stride=1 → f2
    - Concat [input, f1, f2]
    - Conv 1x1 (bottleneck)
    - Conv 3x3 stride=1
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        bottleneck_ratio (float): Bottleneck channel ratio (default: 0.5)
    """
    
    def __init__(self, c1, c2, bottleneck_ratio=0.5):
        """Initialize DSOB for dense small object processing."""
        super().__init__()
        from .conv import Conv
        
        c_bottleneck = int(c1 * bottleneck_ratio)
        
        # Conv 3x3 stride=1 → f1
        self.conv_f1 = Conv(c1, c1, k=3, s=1, act=True)
        # Conv 3x3 stride=1 → f2
        self.conv_f2 = Conv(c1, c1, k=3, s=1, act=True)
        # Conv 1x1 (bottleneck) - input will be c1*3 after concat
        self.conv_bottleneck = Conv(c1 * 3, c_bottleneck, k=1, s=1, act=True)
        # Conv 3x3 stride=1
        self.conv_out = Conv(c_bottleneck, c2, k=3, s=1, act=True)
        
    def forward(self, x):
        """Forward pass through DSOB."""
        identity = x
        # Generate f1 and f2
        f1 = self.conv_f1(x)
        f2 = self.conv_f2(f1)  # Process f1 to get f2
        # Concat [input, f1, f2]
        x = torch.cat([identity, f1, f2], dim=1)
        # Bottleneck
        x = self.conv_bottleneck(x)
        # Final conv
        x = self.conv_out(x)
        return x


class EAE(nn.Module):
    """
    Edge-Aware Enhancement Block for edge-aware feature enhancement.
    
    Architecture:
    - Conv 3x3 stride=1 → feature_main
    - Sobel/Laplacian filter (fixed weight) → edge_map
    - Conv 1x1 on edge_map
    - Multiply: feature_main * edge_enhanced
    - Conv 3x3 stride=1
    - Add residual
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
    """
    
    def __init__(self, c1, c2):
        """Initialize EAE for edge-aware enhancement."""
        super().__init__()
        from .conv import Conv
        
        self.use_residual = (c1 == c2)
        
        # Conv 3x3 stride=1 → feature_main
        self.conv_main = Conv(c1, c1, k=3, s=1, act=True)
        # Fixed Sobel filter for edge detection (X direction)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        # Register as buffer (not trainable parameter)
        self.register_buffer('sobel_x', sobel_x.repeat(c1, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(c1, 1, 1, 1))
        # Conv 1x1 on edge_map
        self.conv_edge = Conv(c1, c1, k=1, s=1, act=True)
        # Conv 3x3 stride=1
        self.conv_out = Conv(c1, c2, k=3, s=1, act=True)
        
    def forward(self, x):
        """Forward pass through EAE."""
        identity = x
        # Feature main
        feature_main = self.conv_main(x)
        # Edge detection using Sobel filters
        edge_x = F.conv2d(x, self.sobel_x, padding=1, groups=x.shape[1])
        edge_y = F.conv2d(x, self.sobel_y, padding=1, groups=x.shape[1])
        edge_map = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)  # Gradient magnitude
        # Conv 1x1 on edge_map
        edge_enhanced = self.conv_edge(edge_map)
        # Multiply: feature_main * edge_enhanced
        x = feature_main * (1 + edge_enhanced)  # Element-wise multiply with enhancement
        # Conv 3x3 stride=1
        x = self.conv_out(x)
        # Add residual
        if self.use_residual:
            x = x + identity
        return x


class CIB2(nn.Module):
    """
    Context Injection Block for injecting context from P3 to P2.
    
    Architecture:
    - Input dari P3: Upsample 2x
    - Input dari P2: identity
    - Concat [P2, P3_up]
    - Conv 3x3 stride=1
    - MaxPool 3x3 stride=1 padding=1
    - Conv 3x3 stride=1
    - Conv 1x1
    
    Args:
        c_p2 (int): P2 channels
        c_p3 (int): P3 channels
        c_out (int): Output channels (default: same as P2)
    """
    
    def __init__(self, c_p2, c_p3, c_out=None):
        """Initialize CIB2 for context injection."""
        super().__init__()
        from .conv import Conv
        
        if c_out is None:
            c_out = c_p2
        
        # Upsample 2x (will be done in forward)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Conv 1x1 to match P2 channels (before concat)
        self.conv_p3 = Conv(c_p3, c_p2, k=1, s=1, act=True)
        # Concat [P2, P3_up] -> input will be c_p2*2
        # Conv 3x3 stride=1
        self.conv1 = Conv(c_p2 * 2, c_p2, k=3, s=1, act=True)
        # MaxPool 3x3 stride=1 padding=1
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # Conv 3x3 stride=1
        self.conv2 = Conv(c_p2, c_p2, k=3, s=1, act=True)
        # Conv 1x1
        self.conv_out = Conv(c_p2, c_out, k=1, s=1, act=True)
        
    def forward(self, x):
        """
        Forward pass through CIB2.
        
        Args:
            x: List of 2 tensors [p2, p3]
        
        Returns:
            Enhanced P2 feature with P3 context
        """
        if isinstance(x, (list, tuple)) and len(x) == 2:
            p2, p3 = x
        else:
            raise ValueError(f"CIB2 expects list of 2 tensors [p2, p3], got {type(x)}")
        
        # Upsample P3 2x
        p3_up = self.upsample(p3)
        # Match channels
        p3_up = self.conv_p3(p3_up)
        # Concat [P2, P3_up]
        x = torch.cat([p2, p3_up], dim=1)
        # Conv 3x3 stride=1
        x = self.conv1(x)
        # MaxPool 3x3 stride=1 padding=1
        x = self.maxpool(x)
        # Conv 3x3 stride=1
        x = self.conv2(x)
        # Conv 1x1
        x = self.conv_out(x)
        return x


class TEB(nn.Module):
    """
    Texture Enhancement Block for enhancing high-frequency details for AFB detection.
    
    Architecture:
    - Laplacian filter/edge detection branch
    - Original feature branch
    - Weighted fusion (learnable)
    - Conv 1x1 projection
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
    """
    
    def __init__(self, c1, c2):
        """Initialize TEB for texture enhancement."""
        super().__init__()
        from .conv import Conv
        
        # Laplacian filter for edge detection (fixed weight)
        # Laplacian kernel: [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
        laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        # Register as buffer (not trainable parameter)
        self.register_buffer('laplacian', laplacian.repeat(c1, 1, 1, 1))
        
        # Edge detection branch: Conv 1x1 on edge features
        self.conv_edge = Conv(c1, c1, k=1, s=1, act=True)
        
        # Original feature branch: Conv 1x1
        self.conv_original = Conv(c1, c1, k=1, s=1, act=True)
        
        # Weighted fusion (learnable weights)
        # Generate fusion weights using 1x1 conv + sigmoid
        self.fusion_conv = Conv(c1 * 2, 2, k=1, s=1, act=False)  # Output 2 weights
        self.softmax = nn.Softmax(dim=1)
        
        # Conv 1x1 projection
        self.conv_out = Conv(c1, c2, k=1, s=1, act=True)
        
    def forward(self, x):
        """Forward pass through TEB."""
        # Laplacian filter/edge detection branch
        edge_features = F.conv2d(x, self.laplacian, padding=1, groups=x.shape[1])
        edge_features = torch.abs(edge_features)  # Absolute value for edge magnitude
        edge_branch = self.conv_edge(edge_features)
        
        # Original feature branch
        original_branch = self.conv_original(x)
        
        # Concat for weighted fusion
        concat_features = torch.cat([edge_branch, original_branch], dim=1)
        
        # Generate learnable weights
        weights = self.fusion_conv(concat_features)  # [B, 2, H, W]
        weights = self.softmax(weights)  # Normalize weights
        
        # Weighted fusion
        w_edge = weights[:, 0:1, :, :]  # [B, 1, H, W]
        w_original = weights[:, 1:2, :, :]  # [B, 1, H, W]
        
        fused = w_edge * edge_branch + w_original * original_branch
        
        # Conv 1x1 projection
        x = self.conv_out(fused)
        return x


class FDB(nn.Module):
    """
    Feature Decoupling Block for separating semantic vs localization features.
    
    Architecture:
    - Dual pathway: semantic branch (Conv 3x3 dilation=2) + localization branch (Conv 3x3 stride=1)
    - Cross-fusion antara kedua branch
    - Concatenate output
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
    """
    
    def __init__(self, c1, c2):
        """Initialize FDB for feature decoupling."""
        super().__init__()
        from .conv import Conv
        
        c_mid = c1 // 2
        
        # Semantic branch: Conv 3x3 dilation=2
        self.semantic_conv = Conv(c1, c_mid, k=3, s=1, d=2, act=True)
        
        # Localization branch: Conv 3x3 stride=1
        self.localization_conv = Conv(c1, c_mid, k=3, s=1, act=True)
        
        # Cross-fusion: each branch processes the other's output
        self.semantic_fusion = Conv(c_mid, c_mid, k=1, s=1, act=True)
        self.localization_fusion = Conv(c_mid, c_mid, k=1, s=1, act=True)
        
        # Final output: concatenate both branches
        self.conv_out = Conv(c_mid * 2, c2, k=1, s=1, act=True)
        
    def forward(self, x):
        """Forward pass through FDB."""
        # Dual pathway
        semantic = self.semantic_conv(x)  # Semantic branch
        localization = self.localization_conv(x)  # Localization branch
        
        # Cross-fusion
        semantic_fused = self.semantic_fusion(localization) + semantic
        localization_fused = self.localization_fusion(semantic) + localization
        
        # Concatenate output
        x = torch.cat([semantic_fused, localization_fused], dim=1)
        x = self.conv_out(x)
        return x


class SACB(nn.Module):
    """
    Scale-Aware Convolution Block for adaptive kernel size based on feature statistics.
    
    Architecture:
    - Compute spatial variance/mean per channel
    - Generate kernel size weights (small object → larger kernel)
    - Apply multi-kernel conv with learned weights
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
    """
    
    def __init__(self, c1, c2):
        """Initialize SACB for scale-aware convolution."""
        super().__init__()
        from .conv import Conv
        
        # Multi-kernel convolutions (3x3, 5x5, 7x7)
        self.conv3 = Conv(c1, c2, k=3, s=1, act=True)
        self.conv5 = Conv(c1, c2, k=5, s=1, act=True)
        self.conv7 = Conv(c1, c2, k=7, s=1, act=True)
        
        # Weight generation: compute statistics and generate weights
        # After concat mean+variance, channels become 2*c1
        # Use nn.Conv2d directly (without BatchNorm) for 1x1 spatial inputs to avoid BatchNorm error
        self.stat_conv = nn.Conv2d(2 * c1, c1, kernel_size=1, stride=1, bias=True)
        self.weight_conv = nn.Conv2d(c1, 3, kernel_size=1, stride=1, bias=True)  # Output 3 weights for 3 kernels
        self.act = nn.SiLU()  # Activation for stat_conv
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        """Forward pass through SACB."""
        B, C, H, W = x.shape
        
        # Compute spatial variance/mean per channel
        # Mean per channel: [B, C, 1, 1]
        mean = x.mean(dim=[2, 3], keepdim=True)
        # Variance per channel: [B, C, 1, 1]
        variance = ((x - mean) ** 2).mean(dim=[2, 3], keepdim=True)
        
        # Combine mean and variance for statistics
        # Apply activation after stat_conv (no BatchNorm, so apply activation manually)
        stats = self.act(self.stat_conv(torch.cat([mean, variance], dim=1)))
        
        # Generate kernel size weights (small object → larger kernel)
        # Higher variance/mean → smaller objects → use larger kernel
        weights = self.weight_conv(stats)  # [B, 3, 1, 1]
        weights = self.softmax(weights)
        
        # Apply multi-kernel conv
        out3 = self.conv3(x)  # 3x3 kernel
        out5 = self.conv5(x)  # 5x5 kernel
        out7 = self.conv7(x)  # 7x7 kernel
        
        # Weighted combination
        w3 = weights[:, 0:1, :, :]
        w5 = weights[:, 1:2, :, :]
        w7 = weights[:, 2:3, :, :]
        
        x = w3 * out3 + w5 * out5 + w7 * out7
        return x


class FBSB(nn.Module):
    """
    Foreground-Background Separation Block for explicit FG/BG modeling.
    
    Architecture:
    - Generate attention mask (sigmoid(Conv 1x1))
    - Foreground stream: features * mask
    - Background stream: features * (1-mask)
    - Process separately, concat
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
    """
    
    def __init__(self, c1, c2):
        """Initialize FBSB for foreground-background separation."""
        super().__init__()
        from .conv import Conv
        
        # Generate attention mask
        self.mask_conv = Conv(c1, 1, k=1, s=1, act=False)
        self.sigmoid = nn.Sigmoid()
        
        # Foreground stream processing
        self.fg_conv = Conv(c1, c2 // 2, k=3, s=1, act=True)
        
        # Background stream processing
        self.bg_conv = Conv(c1, c2 // 2, k=3, s=1, act=True)
        
        # Final fusion
        self.conv_out = Conv(c2, c2, k=1, s=1, act=True)
        
    def forward(self, x):
        """Forward pass through FBSB."""
        # Generate attention mask
        mask = self.sigmoid(self.mask_conv(x))  # [B, 1, H, W]
        
        # Foreground stream: features * mask
        fg_features = x * mask
        fg_out = self.fg_conv(fg_features)
        
        # Background stream: features * (1-mask)
        bg_features = x * (1 - mask)
        bg_out = self.bg_conv(bg_features)
        
        # Concat and final processing
        x = torch.cat([fg_out, bg_out], dim=1)
        x = self.conv_out(x)
        return x


class FBSBE(nn.Module):
    """
    FBSB-Enhanced: Deeper processing per stream + residual connection.
    
    Architecture:
    - Generate mask with BatchNorm
    - Foreground: x * mask → Conv3x3 → BN → SiLU (deeper refinement)
    - Background: x * (1-mask) → Conv3x3 → BN → SiLU
    - Concat → Conv1x1 → residual add with input
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
    """
    
    def __init__(self, c1, c2):
        """Initialize FBSB-Enhanced for deeper foreground-background separation."""
        super().__init__()
        from .conv import Conv
        
        # Generate attention mask with BatchNorm
        self.mask_conv = nn.Conv2d(c1, 1, kernel_size=1, stride=1, bias=False)
        self.mask_bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
        
        # Foreground stream: deeper processing
        self.fg_conv = Conv(c1, c2 // 2, k=3, s=1, act=True)
        
        # Background stream: deeper processing
        self.bg_conv = Conv(c1, c2 // 2, k=3, s=1, act=True)
        
        # Final fusion
        self.conv_out = Conv(c2, c2, k=1, s=1, act=True)
        
        # Residual connection (only if channels match)
        self.use_residual = (c1 == c2)
        
    def forward(self, x):
        """Forward pass through FBSB-Enhanced."""
        identity = x
        
        # Generate attention mask with BatchNorm
        mask = self.mask_conv(x)
        mask = self.mask_bn(mask)
        mask = self.sigmoid(mask)  # [B, 1, H, W]
        
        # Foreground stream: features * mask → deeper refinement
        fg_features = x * mask
        fg_out = self.fg_conv(fg_features)
        
        # Background stream: features * (1-mask) → deeper refinement
        bg_features = x * (1 - mask)
        bg_out = self.bg_conv(bg_features)
        
        # Concat and final processing
        x = torch.cat([fg_out, bg_out], dim=1)
        x = self.conv_out(x)
        
        # Residual connection
        if self.use_residual:
            x = x + identity
        
        return x


class FBSBMS(nn.Module):
    """
    FBSB-MultiScale: Multi-scale mask generation for different object sizes.
    
    Architecture:
    - Generate masks at multiple scales (1x1, 3x3, 5x5)
    - Fuse masks: Conv1x1(concat[mask_1x1, mask_3x3, mask_5x5]) → Sigmoid
    - Foreground: x * mask_final → Conv3x3
    - Background: x * (1-mask_final) → Conv3x3
    - Concat → Conv1x1
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
    """
    
    def __init__(self, c1, c2):
        """Initialize FBSB-MultiScale for multi-scale mask generation."""
        super().__init__()
        from .conv import Conv, DWConv
        
        # Multi-scale mask generation
        self.mask_1x1 = nn.Conv2d(c1, 1, kernel_size=1, stride=1, bias=False)
        self.mask_3x3 = nn.Conv2d(c1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.mask_5x5 = DWConv(c1, 1, k=5, s=1, d=1, act=False)
        
        # Fuse masks
        self.mask_fuse = Conv(3, 1, k=1, s=1, act=False)
        self.sigmoid = nn.Sigmoid()
        
        # Foreground stream processing
        self.fg_conv = Conv(c1, c2 // 2, k=3, s=1, act=True)
        
        # Background stream processing
        self.bg_conv = Conv(c1, c2 // 2, k=3, s=1, act=True)
        
        # Final fusion
        self.conv_out = Conv(c2, c2, k=1, s=1, act=True)
        
    def forward(self, x):
        """Forward pass through FBSB-MultiScale."""
        # Generate masks at multiple scales
        mask_1x1 = self.mask_1x1(x)  # [B, 1, H, W]
        mask_3x3 = self.mask_3x3(x)  # [B, 1, H, W]
        mask_5x5 = self.mask_5x5(x)  # [B, 1, H, W]
        
        # Fuse masks
        mask_concat = torch.cat([mask_1x1, mask_3x3, mask_5x5], dim=1)  # [B, 3, H, W]
        mask_fused = self.mask_fuse(mask_concat)  # [B, 1, H, W]
        mask_final = self.sigmoid(mask_fused)
        
        # Foreground stream: features * mask_final
        fg_features = x * mask_final
        fg_out = self.fg_conv(fg_features)
        
        # Background stream: features * (1-mask_final)
        bg_features = x * (1 - mask_final)
        bg_out = self.bg_conv(bg_features)
        
        # Concat and final processing
        x = torch.cat([fg_out, bg_out], dim=1)
        x = self.conv_out(x)
        return x


class FBSBT(nn.Module):
    """
    FBSB-Triple: Three-way separation for small objects, medium objects, and background.
    
    Architecture:
    - Generate 3-class mask: Conv1x1 → Softmax (3 classes)
    - Small objects: x * mask[:,:,:,0] → Conv3x3
    - Medium objects: x * mask[:,:,:,1] → Conv3x3
    - Background: x * mask[:,:,:,2] → Conv3x3
    - Concat all → Conv1x1
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
    """
    
    def __init__(self, c1, c2):
        """Initialize FBSB-Triple for three-way object separation."""
        super().__init__()
        from .conv import Conv
        
        # Generate 3-class mask (small, medium, background)
        self.mask_conv = Conv(c1, 3, k=1, s=1, act=False)
        self.softmax = nn.Softmax(dim=1)
        
        # Process each category separately
        # Output channels divided by 3 for each stream
        stream_channels = c2 // 3
        self.small_conv = Conv(c1, stream_channels, k=3, s=1, act=True)
        self.medium_conv = Conv(c1, stream_channels, k=3, s=1, act=True)
        self.bg_conv = Conv(c1, stream_channels, k=3, s=1, act=True)
        
        # Final fusion (handle remainder if c2 not divisible by 3)
        self.conv_out = Conv(c2, c2, k=1, s=1, act=True)
        
    def forward(self, x):
        """Forward pass through FBSB-Triple."""
        # Generate 3-class mask
        mask_logits = self.mask_conv(x)  # [B, 3, H, W]
        mask = self.softmax(mask_logits)  # [B, 3, H, W]
        
        # Split mask into 3 classes
        mask_small = mask[:, 0:1, :, :]   # [B, 1, H, W]
        mask_medium = mask[:, 1:2, :, :]  # [B, 1, H, W]
        mask_bg = mask[:, 2:3, :, :]      # [B, 1, H, W]
        
        # Small objects stream
        small_features = x * mask_small
        small_out = self.small_conv(small_features)
        
        # Medium objects stream
        medium_features = x * mask_medium
        medium_out = self.medium_conv(medium_features)
        
        # Background stream
        bg_features = x * mask_bg
        bg_out = self.bg_conv(bg_features)
        
        # Concat all streams
        x = torch.cat([small_out, medium_out, bg_out], dim=1)
        x = self.conv_out(x)
        return x


class FPI(nn.Module):
    """
    Feature Pyramid Injection: Inject P4 and P5 features into P3.
    
    Architecture:
    - P5 → Conv1x1 reduce → Downsample 4x → inject to P3
    - P4 → Conv1x1 reduce → Downsample 2x → inject to P3
    - Concat [P3_original, P4_inject, P5_inject]
    - Conv3x3 → Conv1x1 bottleneck
    
    Args:
        c_p3 (int): P3 channels
        c_p4 (int): P4 channels
        c_p5 (int): P5 channels
        c_out (int): Output channels
    """
    
    def __init__(self, c_p3, c_p4, c_p5, c_out):
        """Initialize FPI for feature pyramid injection."""
        super().__init__()
        from .conv import Conv
        
        # Reduce and downsample P5 (4x)
        self.p5_reduce = Conv(c_p5, c_p3, k=1, s=1, act=True)
        self.p5_downsample = nn.AvgPool2d(kernel_size=4, stride=4)
        
        # Reduce and downsample P4 (2x)
        self.p4_reduce = Conv(c_p4, c_p3, k=1, s=1, act=True)
        self.p4_downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # After concat: P3 + P4_inject + P5_inject = 3 * c_p3
        self.conv_fusion = Conv(c_p3 * 3, c_p3, k=3, s=1, act=True)
        self.conv_bottleneck = Conv(c_p3, c_out, k=1, s=1, act=True)
        
    def forward(self, x):
        """
        Forward pass through FPI.
        
        Args:
            x: List of 3 tensors [P3, P4, P5]
        
        Returns:
            Enhanced P3 features
        """
        if isinstance(x, (list, tuple)) and len(x) == 3:
            p3, p4, p5 = x
        else:
            raise ValueError(f"FPI expects list of 3 tensors [P3, P4, P5], got {type(x)}")
        
        # Process P5: reduce and downsample 4x
        p5_reduced = self.p5_reduce(p5)
        p5_inject = self.p5_downsample(p5_reduced)
        
        # Process P4: reduce and downsample 2x
        p4_reduced = self.p4_reduce(p4)
        p4_inject = self.p4_downsample(p4_reduced)
        
        # Ensure spatial dimensions match P3
        _, _, h3, w3 = p3.shape
        if p4_inject.shape[2:] != (h3, w3):
            p4_inject = F.interpolate(p4_inject, size=(h3, w3), mode='bilinear', align_corners=False)
        if p5_inject.shape[2:] != (h3, w3):
            p5_inject = F.interpolate(p5_inject, size=(h3, w3), mode='bilinear', align_corners=False)
        
        # Concat all features
        x = torch.cat([p3, p4_inject, p5_inject], dim=1)
        
        # Fusion and bottleneck
        x = self.conv_fusion(x)
        x = self.conv_bottleneck(x)
        return x


class SPP3(nn.Module):
    """
    Spatial Pyramid Pooling for P3: Multi-scale context aggregation.
    
    Architecture:
    - Parallel pooling: MaxPool 5x5, 9x9, 13x13
    - Original features
    - Concat → Conv1x1
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
    """
    
    def __init__(self, c1, c2):
        """Initialize SPP3 for spatial pyramid pooling."""
        super().__init__()
        from .conv import Conv
        
        # Parallel pooling branches
        self.pool5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool9 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool13 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        
        # After concat: 4 * c1 (original + 3 pools)
        self.conv_out = Conv(c1 * 4, c2, k=1, s=1, act=True)
        
    def forward(self, x):
        """Forward pass through SPP3."""
        # Parallel pooling
        pool5_out = self.pool5(x)
        pool9_out = self.pool9(x)
        pool13_out = self.pool13(x)
        
        # Concat all
        x = torch.cat([x, pool5_out, pool9_out, pool13_out], dim=1)
        x = self.conv_out(x)
        return x


class CSFR(nn.Module):
    """
    Cross-Stage Feature Refinement: Bi-directional flow between P3 and P4.
    
    Architecture:
    - P3 → Conv1x1 → send to P4
    - P4 process → Upsample 2x → return to P3
    - P3_refined = P3_original + returned_from_P4
    - Conv3x3
    
    Args:
        c_p3 (int): P3 channels
        c_p4 (int): P4 channels
        c_out (int): Output channels
    """
    
    def __init__(self, c_p3, c_p4, c_out):
        """Initialize CSFR for cross-stage feature refinement."""
        super().__init__()
        from .conv import Conv
        
        # P3 → P4: reduce and prepare
        self.p3_to_p4 = Conv(c_p3, c_p4, k=1, s=1, act=True)
        
        # P4 processing
        self.p4_process = Conv(c_p4, c_p4, k=3, s=1, act=True)
        
        # P4 → P3: upsample and return
        self.p4_to_p3 = Conv(c_p4, c_p3, k=1, s=1, act=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Final refinement
        self.conv_refine = Conv(c_p3, c_out, k=3, s=1, act=True)
        
    def forward(self, x):
        """
        Forward pass through CSFR.
        
        Args:
            x: List of 2 tensors [P3, P4]
        
        Returns:
            Refined P3 features
        """
        if isinstance(x, (list, tuple)) and len(x) == 2:
            p3, p4 = x
        else:
            raise ValueError(f"CSFR expects list of 2 tensors [P3, P4], got {type(x)}")
        
        # P3 → P4
        p3_sent = self.p3_to_p4(p3)
        
        # P4 process (add P3 info)
        p4_enhanced = self.p4_process(p4 + p3_sent)
        
        # P4 → P3: return refined features
        p4_returned = self.p4_to_p3(p4_enhanced)
        p4_returned = self.upsample(p4_returned)
        
        # Ensure spatial match
        _, _, h3, w3 = p3.shape
        if p4_returned.shape[2:] != (h3, w3):
            p4_returned = F.interpolate(p4_returned, size=(h3, w3), mode='bilinear', align_corners=False)
        
        # Refine P3
        p3_refined = p3 + p4_returned
        x = self.conv_refine(p3_refined)
        return x


class DenseP3(nn.Module):
    """
    Dense Connection for P3 Neck: Dense information flow.
    
    Architecture:
    - Layer 1: Conv3x3 → save f1
    - Layer 2: Conv3x3(concat[input, f1]) → save f2
    - Layer 3: Conv3x3(concat[input, f1, f2])
    - Conv1x1 bottleneck
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
    """
    
    def __init__(self, c1, c2):
        """Initialize DenseP3 for dense connections."""
        super().__init__()
        from .conv import Conv
        
        # Dense layers
        self.conv1 = Conv(c1, c1, k=3, s=1, act=True)
        self.conv2 = Conv(c1 * 2, c1, k=3, s=1, act=True)  # input + f1
        self.conv3 = Conv(c1 * 3, c1, k=3, s=1, act=True)  # input + f1 + f2
        
        # Bottleneck
        self.conv_bottleneck = Conv(c1, c2, k=1, s=1, act=True)
        
    def forward(self, x):
        """Forward pass through DenseP3."""
        identity = x
        
        # Layer 1
        f1 = self.conv1(x)
        
        # Layer 2: concat input + f1
        x2 = torch.cat([identity, f1], dim=1)
        f2 = self.conv2(x2)
        
        # Layer 3: concat input + f1 + f2
        x3 = torch.cat([identity, f1, f2], dim=1)
        f3 = self.conv3(x3)
        
        # Bottleneck
        x = self.conv_bottleneck(f3)
        return x


class DeformableHead(nn.Module):
    """
    Deformable Detection Head: Adaptive geometric transformation.
    
    Architecture:
    - Normal flow → Deformable Conv3x3 (offset learning)
    - Conv3x3 → output
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
    """
    
    def __init__(self, c1, c2):
        """Initialize DeformableHead for adaptive detection."""
        super().__init__()
        from .conv import Conv
        
        # Generate offsets for deformable conv
        self.offset_conv = Conv(c1, 18, k=3, s=1, act=False)  # 2 * 3 * 3 = 18 for 3x3 kernel
        
        # Deformable conv (simulated with grid_sample)
        self.conv_normal = Conv(c1, c2, k=3, s=1, act=True)
        
    def forward(self, x):
        """Forward pass through DeformableHead."""
        B, C, H, W = x.shape
        
        # Skip deformable if spatial size too small
        if min(H, W) < 3:
            return self.conv_normal(x)
        
        # Generate offsets
        offsets = self.offset_conv(x)  # [B, 18, H, W]
        offsets = offsets * 0.1  # Scale for stability
        offsets = offsets.view(B, 2, 9, H, W)  # [B, 2, 9, H, W] (2 for x,y, 9 for 3x3 kernel)
        
        # Create base grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, dtype=x.dtype, device=x.device),
            torch.arange(W, dtype=x.dtype, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([x_coords, y_coords], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]
        
        # Apply offsets (simplified: use center offset)
        center_offset = offsets[:, :, 4, :, :]  # [B, 2, H, W] (center of 3x3 kernel)
        grid = grid + center_offset
        
        # Normalize grid
        grid = grid.permute(0, 2, 3, 1)  # [B, H, W, 2]
        if W > 1:
            grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / (W - 1) - 1.0
        else:
            grid[:, :, :, 0] = 0.0
        if H > 1:
            grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / (H - 1) - 1.0
        else:
            grid[:, :, :, 1] = 0.0
        
        # Apply deformable sampling
        x_deform = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        # Final conv
        x = self.conv_normal(x_deform)
        return x


class OCS(nn.Module):
    """
    Object-Context Separation: Separate object and context features.
    
    Architecture:
    - Context branch: LargeKernel Conv (7x7 or 9x9)
    - Object branch: SmallKernel Conv (3x3)
    - Gate = sigmoid(Conv1x1(context))
    - Output = object * gate + context * (1-gate)
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        large_kernel (int): Large kernel size for context (default: 7)
    """
    
    def __init__(self, c1, c2, large_kernel=7):
        """Initialize OCS for object-context separation."""
        super().__init__()
        from .conv import Conv
        
        # Context branch: large kernel
        padding = large_kernel // 2
        self.context_conv = Conv(c1, c2, k=large_kernel, s=1, p=padding, act=True)
        
        # Object branch: small kernel
        self.object_conv = Conv(c1, c2, k=3, s=1, act=True)
        
        # Gate generation
        self.gate_conv = Conv(c2, c2, k=1, s=1, act=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """Forward pass through OCS."""
        # Context branch
        context = self.context_conv(x)
        
        # Object branch
        obj = self.object_conv(x)
        
        # Generate gate from context
        gate = self.sigmoid(self.gate_conv(context))
        
        # Combine: object * gate + context * (1-gate)
        x = obj * gate + context * (1 - gate)
        return x


class RPP(nn.Module):
    """
    Resolution-Preserved Path: Direct connection from P2 to P3 detection.
    
    Architecture:
    - From P2 backbone: Conv1x1 reduce
    - Upsample or keep → direct connect to P3 detection
    - Concat with normal P3 neck output
    
    Args:
        c_p2 (int): P2 channels
        c_p3 (int): P3 channels
        c_out (int): Output channels
    """
    
    def __init__(self, c_p2, c_p3, c_out):
        """Initialize RPP for resolution-preserved path."""
        super().__init__()
        from .conv import Conv
        
        # Reduce P2 channels
        self.p2_reduce = Conv(c_p2, c_p3, k=1, s=1, act=True)
        
        # Upsample if needed (P2 is 2x larger than P3)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Fusion
        self.conv_fusion = Conv(c_p3 * 2, c_out, k=3, s=1, act=True)
        
    def forward(self, x):
        """
        Forward pass through RPP.
        
        Args:
            x: List of 2 tensors [P3_neck, P2_backbone]
        
        Returns:
            Enhanced P3 features with preserved resolution
        """
        if isinstance(x, (list, tuple)) and len(x) == 2:
            p3_neck, p2_backbone = x
        else:
            raise ValueError(f"RPP expects list of 2 tensors [P3_neck, P2_backbone], got {type(x)}")
        
        # Process P2
        p2_reduced = self.p2_reduce(p2_backbone)
        p2_upsampled = self.upsample(p2_reduced)
        
        # Ensure spatial match
        _, _, h3, w3 = p3_neck.shape
        if p2_upsampled.shape[2:] != (h3, w3):
            p2_upsampled = F.interpolate(p2_upsampled, size=(h3, w3), mode='bilinear', align_corners=False)
        
        # Concat and fuse
        x = torch.cat([p3_neck, p2_upsampled], dim=1)
        x = self.conv_fusion(x)
        return x


class FDEB(nn.Module):
    """
    Frequency Domain Enhancement Block for FFT-based high-frequency boost.
    
    Architecture:
    - FFT transform
    - Amplify high-frequency components
    - IFFT back
    - Residual add dengan original
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
    """
    
    def __init__(self, c1, c2):
        """Initialize FDEB for frequency domain enhancement."""
        super().__init__()
        from .conv import Conv
        
        self.use_residual = (c1 == c2)
        
        # Learnable high-frequency amplification
        # After concat real+imaginary, channels become 2*c1
        self.amp_conv = Conv(2 * c1, 2 * c1, k=1, s=1, act=True)
        
        # Final projection
        self.conv_out = Conv(c1, c2, k=1, s=1, act=True)
        
    def forward(self, x):
        """Forward pass through FDEB."""
        identity = x
        B, C, H, W = x.shape
        original_dtype = x.dtype
        
        # Always convert to float32 for FFT to avoid cuFFT half precision issues
        x_float = x.float()
        
        # Check if dimensions are power of 2 (required for cuFFT in half precision)
        # If not, pad to nearest power of 2
        def next_power_of_2(n):
            return 1 << (n - 1).bit_length()
        
        def is_power_of_2(n):
            return (n & (n - 1)) == 0 and n > 0
        
        # Pad if needed
        H_pad = H if is_power_of_2(H) else next_power_of_2(H)
        W_pad = W if is_power_of_2(W) else next_power_of_2(W)
        
        if H_pad != H or W_pad != W:
            # Pad to power of 2
            pad_h = (H_pad - H) // 2
            pad_w = (W_pad - W) // 2
            x_padded = F.pad(x_float, (pad_w, W_pad - W - pad_w, pad_h, H_pad - H - pad_h), mode='reflect')
        else:
            x_padded = x_float
        
        # FFT transform (apply to each channel separately)
        x_fft = torch.fft.rfft2(x_padded, norm='ortho')  # [B, C, H_pad, W_pad//2+1] complex
        
        # Separate real and imaginary parts
        x_fft_real = x_fft.real
        x_fft_imag = x_fft.imag
        
        # Concatenate real and imaginary for processing
        x_fft_concat = torch.cat([x_fft_real, x_fft_imag], dim=1)  # [B, 2*C, H_pad, W_pad//2+1]
        
        # Amplify high-frequency components (learnable)
        # Get the dtype of amp_conv weights to ensure dtype match
        amp_conv_dtype = next(self.amp_conv.parameters()).dtype
        x_fft_concat = x_fft_concat.to(amp_conv_dtype)
        x_fft_amp = self.amp_conv(x_fft_concat)
        # Convert back to float32 for IFFT
        x_fft_amp = x_fft_amp.float()
        
        # Clamp to prevent extreme values that can cause NaN in IFFT
        # Use a reasonable range to prevent numerical instability
        x_fft_amp = torch.clamp(x_fft_amp, min=-1e6, max=1e6)
        
        # Split back
        x_fft_amp_real = x_fft_amp[:, :C, :, :]
        x_fft_amp_imag = x_fft_amp[:, C:, :, :]
        
        # Clamp real and imaginary parts separately to prevent NaN
        x_fft_amp_real = torch.clamp(x_fft_amp_real, min=-1e6, max=1e6)
        x_fft_amp_imag = torch.clamp(x_fft_amp_imag, min=-1e6, max=1e6)
        
        # Reconstruct complex tensor
        x_fft_enhanced = torch.complex(x_fft_amp_real, x_fft_amp_imag)
        
        # IFFT back
        x_enhanced = torch.fft.irfft2(x_fft_enhanced, s=(H_pad, W_pad), norm='ortho')
        
        # Check for NaN/Inf and replace with zeros to prevent propagation
        x_enhanced = torch.nan_to_num(x_enhanced, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Crop back to original size if padded
        if H_pad != H or W_pad != W:
            x_enhanced = x_enhanced[:, :, :H, :W]
        
        # Convert back to original dtype before residual and final conv
        x_enhanced = x_enhanced.to(original_dtype)
        
        # Clamp output to reasonable range before residual to prevent NaN propagation
        x_enhanced = torch.clamp(x_enhanced, min=-1e3, max=1e3)
        
        # Residual add dengan original (ensure same dtype)
        if self.use_residual:
            x_enhanced = x_enhanced + identity.to(original_dtype)
            # Check for NaN after residual addition
            x_enhanced = torch.nan_to_num(x_enhanced, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Final projection (input and weights will match dtype now)
        x = self.conv_out(x_enhanced)
        
        # Final NaN check before returning
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x


class DPRB(nn.Module):
    """
    Dense Prediction Refinement Block for extra dense conv layers untuk P2.
    
    Architecture:
    - Conv 3x3 → Conv 3x3 → Conv 3x3 (3 cascade)
    - Dense connections (connect all to all)
    - Bottleneck di akhir
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
    """
    
    def __init__(self, c1, c2):
        """Initialize DPRB for dense prediction refinement."""
        super().__init__()
        from .conv import Conv
        
        c_mid = c2 // 2
        
        # 3 cascade conv layers
        self.conv1 = Conv(c1, c_mid, k=3, s=1, act=True)
        self.conv2 = Conv(c1 + c_mid, c_mid, k=3, s=1, act=True)  # Dense: input + conv1 output
        self.conv3 = Conv(c1 + c_mid + c_mid, c_mid, k=3, s=1, act=True)  # Dense: input + conv1 + conv2
        
        # Bottleneck di akhir
        self.conv_bottleneck = Conv(c1 + c_mid * 3, c2, k=1, s=1, act=True)
        
    def forward(self, x):
        """Forward pass through DPRB."""
        identity = x
        
        # Conv 3x3 cascade with dense connections
        f1 = self.conv1(x)
        x1 = torch.cat([identity, f1], dim=1)  # Dense connection
        
        f2 = self.conv2(x1)
        x2 = torch.cat([identity, f1, f2], dim=1)  # Dense connection
        
        f3 = self.conv3(x2)
        x3 = torch.cat([identity, f1, f2, f3], dim=1)  # Dense connection
        
        # Bottleneck di akhir
        x = self.conv_bottleneck(x3)
        return x


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention for position-aware attention.
    Paper: Coordinate Attention for Efficient Mobile Network Design (CVPR 2021)
    
    Architecture:
    - X AvgPool (H dimension) -> [B, C, 1, W]
    - Y AvgPool (W dimension) -> [B, C, H, 1]
    - Concat -> Conv -> Split back
    - Apply X attention and Y attention
    - Multiply with features
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels (default: same as input)
        reduction (int): Reduction ratio for intermediate channels (default: 32)
    """
    
    def __init__(self, c1, c2=None, reduction=32):
        """Initialize Coordinate Attention."""
        super().__init__()
        from .conv import Conv
        
        if c2 is None:
            c2 = c1
        
        # Intermediate channels after reduction
        c_mid = max(8, c1 // reduction)
        
        # X-direction pooling (H dimension)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # Y-direction pooling (W dimension)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        # Shared conv for both directions
        self.conv1 = Conv(c1, c_mid, k=1, s=1, act=True)
        
        # Separate convs for X and Y
        self.conv_h = Conv(c_mid, c2, k=1, s=1, act=False)
        self.conv_w = Conv(c_mid, c2, k=1, s=1, act=False)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Forward pass through Coordinate Attention.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Enhanced features with coordinate attention
        """
        identity = x
        B, C, H, W = x.shape
        
        # X-direction: [B, C, H, W] -> [B, C, H, 1]
        x_h = self.pool_h(x)  # [B, C, H, 1]
        
        # Y-direction: [B, C, H, W] -> [B, C, 1, W]
        x_w = self.pool_w(x)  # [B, C, 1, W]
        
        # Concat along spatial dimension: [B, C, H, 1] -> [B, C, 1, H] then concat with [B, C, 1, W]
        x_h = x_h.permute(0, 1, 3, 2)  # [B, C, 1, H]
        x_cat = torch.cat([x_h, x_w], dim=3)  # [B, C, 1, H+W]
        
        # Shared conv: [B, C, 1, H+W] -> [B, c_mid, 1, H+W]
        x_cat = self.conv1(x_cat)  # [B, c_mid, 1, H+W]
        
        # Split back: [B, c_mid, 1, H+W] -> [B, c_mid, 1, H] and [B, c_mid, 1, W]
        x_h, x_w = torch.split(x_cat, [H, W], dim=3)
        x_h = x_h.permute(0, 1, 3, 2)  # [B, c_mid, H, 1]
        x_w = x_w  # [B, c_mid, 1, W]
        
        # Separate convs
        att_h = self.sigmoid(self.conv_h(x_h))  # [B, C, H, 1]
        att_w = self.sigmoid(self.conv_w(x_w))  # [B, C, 1, W]
        
        # Apply attention
        out = identity * att_h * att_w
        return out


class SimAM(nn.Module):
    """
    SimAM: A Simple, Parameter-Free Attention Module (ICCV 2021)
    Parameter-free attention mechanism based on energy function.
    
    Architecture:
    E = (feature - mean)^2 / (variance + lambda)
    Attention = 1 / (1 + E)
    Multiply with original features
    
    Args:
        c1 (int): Input channels (for compatibility, not used)
        c2 (int): Output channels (for compatibility, not used)
        e_lambda (float): Lambda parameter for energy function (default: 1e-4)
    """
    
    def __init__(self, c1=None, c2=None, e_lambda=1e-4):
        """Initialize SimAM attention (parameter-free)."""
        super().__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
        
    def forward(self, x):
        """
        Forward pass through SimAM.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Enhanced features with SimAM attention
        """
        B, C, H, W = x.shape
        
        # Spatial mean: [B, C, H, W] -> [B, C, 1, 1]
        n = W * H - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        
        # Return attended features
        return x * self.activaton(y)


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block: A Modern Conv Block (CVPR 2022)
    Based on ConvNeXt architecture with modern design choices.
    
    Architecture:
    - DWConv 7x7
    - LayerNorm
    - Conv 1x1 (expand 4x)
    - GELU
    - Conv 1x1 (project back)
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels (default: same as input)
        expansion (int): Expansion ratio (default: 4)
        kernel_size (int): Kernel size for depthwise conv (default: 7)
    """
    
    def __init__(self, c1, c2=None, expansion=4, kernel_size=7):
        """Initialize ConvNeXt Block."""
        super().__init__()
        from .conv import Conv, DWConv
        
        if c2 is None:
            c2 = c1
        
        c_mid = c1 * expansion
        
        # Depthwise Conv 7x7
        self.dwconv = DWConv(c1, c1, k=kernel_size, s=1, act=False)
        
        # LayerNorm (applied per channel)
        self.norm = nn.LayerNorm(c1)
        
        # Pointwise Conv 1x1 (expand)
        self.pwconv1 = Conv(c1, c_mid, k=1, s=1, act=False)
        
        # GELU activation
        self.act = nn.GELU()
        
        # Pointwise Conv 1x1 (project)
        self.pwconv2 = Conv(c_mid, c2, k=1, s=1, act=False)
        
    def forward(self, x):
        """
        Forward pass through ConvNeXt Block.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Processed features
        """
        identity = x
        
        # DWConv 7x7
        x = self.dwconv(x)
        
        # LayerNorm: [B, C, H, W] -> [B, H, W, C] -> norm -> [B, C, H, W]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Pointwise expand
        x = self.pwconv1(x)
        x = self.act(x)
        
        # Pointwise project
        x = self.pwconv2(x)
        
        # Residual connection (if same channels)
        if identity.shape[1] == x.shape[1]:
            x = x + identity
        
        return x


class EdgePriorBlock(nn.Module):
    """
    Edge-Prior Block: High-frequency edge/texture enhancement for small object detection.
    Applies high-pass filter (HPF) via depthwise conv + gate mechanism.
    
    Architecture:
    - Depthwise 3x3 (high-pass filter effect)
    - Conv 1x1
    - Sigmoid gate
    - Residual: x_out = x + x * gate
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels (default: same as input)
    """
    
    def __init__(self, c1, c2=None):
        """Initialize Edge-Prior Block."""
        super().__init__()
        from .conv import Conv, DWConv
        
        if c2 is None:
            c2 = c1
        
        # Depthwise 3x3 for high-pass filter effect
        # Using Laplacian-like kernel for edge detection
        self.dwconv = DWConv(c1, c1, k=3, s=1, act=False)
        
        # Gate: 1x1 conv + sigmoid
        self.gate = nn.Sequential(
            Conv(c1, c1, k=1, s=1, act=False),
            nn.Sigmoid()
        )
        
        # Output projection if channels differ
        if c1 != c2:
            self.proj = Conv(c1, c2, k=1, s=1, act=True)
        else:
            self.proj = None
        
    def forward(self, x):
        """
        Forward pass through Edge-Prior Block.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Enhanced features with edge/texture prior
        """
        identity = x
        
        # High-pass filter effect via depthwise conv
        x_hpf = self.dwconv(x)
        
        # Gate mechanism
        gate = self.gate(x_hpf)
        
        # Residual: x + x * gate
        x = identity + identity * gate
        
        # Project if needed
        if self.proj is not None:
            x = self.proj(x)
        
        return x


class LocalContextMixer(nn.Module):
    """
    Local Context Mixer: Multi-dilated depthwise conv for local context aggregation.
    Captures multi-scale local context without losing detail.
    
    Architecture:
    - Split channels into 3 groups (or use parallel branches)
    - DWConv3x3 dilation=1
    - DWConv3x3 dilation=2
    - DWConv3x3 dilation=3
    - Concat → 1x1 fuse → residual
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels (default: same as input)
    """
    
    def __init__(self, c1, c2=None):
        """Initialize Local Context Mixer."""
        super().__init__()
        from .conv import Conv, DWConv
        
        if c2 is None:
            c2 = c1
        
        # Multi-dilated depthwise convs
        self.dwconv1 = DWConv(c1, c1, k=3, s=1, d=1, act=True)  # dilation=1
        self.dwconv2 = DWConv(c1, c1, k=3, s=1, d=2, act=True)  # dilation=2
        self.dwconv3 = DWConv(c1, c1, k=3, s=1, d=3, act=True)  # dilation=3
        
        # Fusion: 1x1 conv
        self.fuse = Conv(c1 * 3, c2, k=1, s=1, act=True)
        
    def forward(self, x):
        """
        Forward pass through Local Context Mixer.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Features with multi-scale local context
        """
        identity = x
        
        # Multi-dilated branches
        x1 = self.dwconv1(x)  # dilation=1: fine detail
        x2 = self.dwconv2(x)  # dilation=2: medium context
        x3 = self.dwconv3(x)  # dilation=3: wider context
        
        # Concat and fuse
        x_cat = torch.cat([x1, x2, x3], dim=1)  # [B, 3C, H, W]
        x = self.fuse(x_cat)  # [B, C, H, W]
        
        # Residual connection
        if identity.shape[1] == x.shape[1]:
            x = x + identity
        
        return x


class TinyObjectAlignment(nn.Module):
    """
    Tiny-Object Alignment: Lightweight offset warp for alignment before concat.
    Prevents misalignment during FPN upsample for tiny objects.
    
    Architecture:
    - Offset prediction: Conv1x1(x_up) -> 2 channels (dx, dy)
    - Grid sample with offset (or simple warp)
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels (default: same as input)
    """
    
    def __init__(self, c1, c2=None):
        """Initialize Tiny-Object Alignment."""
        super().__init__()
        from .conv import Conv
        
        if c2 is None:
            c2 = c1
        
        # Offset prediction: 2 channels for (dx, dy) per location
        self.offset_conv = Conv(c1, 2, k=1, s=1, act=False)
        
        # Output projection
        if c1 != c2:
            self.proj = Conv(c1, c2, k=1, s=1, act=True)
        else:
            self.proj = None
        
    def forward(self, x):
        """
        Forward pass through Tiny-Object Alignment.
        
        Args:
            x: Input tensor [B, C, H, W] (upsampled feature)
        
        Returns:
            Aligned features
        """
        B, C, H, W = x.shape
        
        # Predict offset: [B, 2, H, W]
        offset = self.offset_conv(x)
        
        # Normalize offset to [-1, 1] range for grid_sample
        # Scale offset to reasonable range (e.g., ±2 pixels)
        offset = offset * 0.1  # Small offset range
        
        # Create base grid: [B, H, W, 2] format for grid_sample
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, dtype=torch.float32, device=x.device),
            torch.arange(0, W, dtype=torch.float32, device=x.device),
            indexing='ij'
        )
        # Stack: [H, W, 2] where last dim is [x, y]
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]
        
        # Normalize grid to [-1, 1] range
        grid = grid / torch.tensor([W - 1, H - 1], dtype=torch.float32, device=x.device) * 2.0 - 1.0
        
        # Apply offset: offset is [B, 2, H, W], need to permute to [B, H, W, 2]
        offset_perm = offset.permute(0, 2, 3, 1)  # [B, H, W, 2]
        # Normalize offset to grid space (scale by feature size)
        offset_perm = offset_perm / torch.tensor([W, H], dtype=torch.float32, device=x.device) * 2.0
        grid = grid + offset_perm * 0.1  # Small offset range
        
        # Grid sample: expects [B, H, W, 2] format
        x_aligned = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        # Project if needed
        if self.proj is not None:
            x_aligned = self.proj(x_aligned)
        
        return x_aligned


class AntiFPGate(nn.Module):
    """
    Anti-FP Gate: Suppress background speckle/noise before prediction.
    Prevents false positives from background noise that looks like small objects.
    
    Architecture:
    - SimAM or sigmoid gate
    - Residual: x = x * gate + x
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels (default: same as input)
        use_simam (bool): Use SimAM or simple sigmoid gate (default: True)
    """
    
    def __init__(self, c1, c2=None, use_simam=True):
        """Initialize Anti-FP Gate."""
        super().__init__()
        from .conv import Conv
        
        if c2 is None:
            c2 = c1
        
        self.use_simam = use_simam
        
        if use_simam:
            # Use SimAM for parameter-free attention
            self.gate = SimAM(c1, c1, e_lambda=1e-4)
        else:
            # Simple sigmoid gate
            self.gate = nn.Sequential(
                Conv(c1, c1, k=1, s=1, act=False),
                nn.Sigmoid()
            )
        
        # Output projection if needed
        if c1 != c2:
            self.proj = Conv(c1, c2, k=1, s=1, act=True)
        else:
            self.proj = None
        
    def forward(self, x):
        """
        Forward pass through Anti-FP Gate.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Features with suppressed background noise
        """
        identity = x
        
        # Gate mechanism
        gate = self.gate(x)
        
        # Residual: x * gate + x
        x = identity * gate + identity
        
        # Project if needed
        if self.proj is not None:
            x = self.proj(x)
        
        return x


class BackgroundSuppressionGate(nn.Module):
    """
    Background Suppression Gate (anti-nukleus): Suppress large activations (cells/nucleus) 
    so BTA can be read.
    
    Concept: Create low-freq mask (large structures) then "gate" features.
    
    Architecture:
    - bg = DWConv(k=7/9, stride=1) (low-pass filter)
    - fg = x - bg (high-pass residual)
    - gate = sigmoid(Conv1x1(fg))
    - out = x * gate + x (residual)
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels (default: same as input)
        kernel_size (int): Kernel size for low-pass filter (default: 7, can be 9)
    """
    
    def __init__(self, c1, c2=None, kernel_size=7):
        """Initialize Background Suppression Gate."""
        super().__init__()
        from .conv import Conv, DWConv
        
        if c2 is None:
            c2 = c1
        
        # Low-pass filter: DWConv with large kernel (7 or 9)
        # This captures large structures (background/nucleus)
        self.bg_conv = DWConv(c1, c1, k=kernel_size, s=1, act=False)
        
        # Gate: 1x1 conv + sigmoid on high-pass features
        self.gate = nn.Sequential(
            Conv(c1, c1, k=1, s=1, act=False),
            nn.Sigmoid()
        )
        
        # Output projection if channels differ
        if c1 != c2:
            self.proj = Conv(c1, c2, k=1, s=1, act=True)
        else:
            self.proj = None
    
    def forward(self, x):
        """
        Forward pass through Background Suppression Gate.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Features with suppressed large activations
        """
        identity = x
        
        # Low-pass: capture background/large structures
        bg = self.bg_conv(x)
        
        # High-pass: foreground = original - background
        fg = x - bg
        
        # Gate mechanism on high-pass features
        gate = self.gate(fg)
        
        # Residual: x * gate + x
        x = identity * gate + identity
        
        # Project if needed
        if self.proj is not None:
            x = self.proj(x)
        
        return x


class LocalContrastGate(nn.Module):
    """
    Local Contrast Gate: Suppress homogeneous texture, enhance local anomalies (small objects).
    
    Konsep:
    - Texture = statistik lokal konsisten (homogen)
    - Objek kecil = outlier lokal (kontras tinggi)
    - Local mean = AvgPool (texture pattern)
    - Contrast = x - local_mean (anomaly signal)
    - Output = x + alpha * contrast (enhance anomaly)
    
    Keuntungan:
    - Tidak learn parameter → stabil
    - Tidak nambah compute berat
    - Secara eksplisit menghukum texture homogen
    - Lebih efektif dari attention untuk stain-heavy cases
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels (default: same as input)
        k (int): Kernel size for local averaging (default: 5)
        alpha (float): Contrast enhancement strength (default: 0.7)
    """
    
    def __init__(self, c1, k=5, alpha=0.7, c2=None):
        """Initialize Local Contrast Gate."""
        super().__init__()
        from .conv import Conv
        
        if c2 is None:
            c2 = c1
        
        # Local averaging (low-pass filter untuk detect texture)
        self.avg = nn.AvgPool2d(k, stride=1, padding=k//2)
        self.alpha = alpha
        
        # Output projection if channels differ
        if c1 != c2:
            self.proj = Conv(c1, c2, k=1, s=1, act=True)
        else:
            self.proj = None
    
    def forward(self, x):
        """
        Forward pass through Local Contrast Gate.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Features with suppressed texture and enhanced local anomalies
        """
        identity = x
        
        # Local mean (texture pattern)
        local_mean = self.avg(x)
        
        # Contrast = anomaly signal (x - local_mean)
        contrast = x - local_mean
        
        # Enhance contrast: x + alpha * contrast
        out = x + self.alpha * contrast
        
        # Project if needed
        if self.proj is not None:
            out = self.proj(out)
        
        return out


class ShapeGate(nn.Module):
    """
    Shape Gate: Enhance edges/structure for rod-like objects, not color/texture.
    
    Konsep:
    - Depthwise conv → fokus spatial (edge detection)
    - Pointwise conv → output projection
    - Residual: x + edge → enhance edge signal
    - Tidak pakai Sobel eksplisit (stabil gradient, CNN-friendly)
    
    Keuntungan:
    - Depthwise → fokus spatial (bukan channel)
    - Tidak pakai Sobel eksplisit (stabil gradien)
    - CNN-friendly
    - Cocok untuk rod/elongated objects
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels (default: same as input)
    """
    
    def __init__(self, c1, c2=None):
        """Initialize Shape Gate."""
        super().__init__()
        from .conv import Conv
        
        if c2 is None:
            c2 = c1
        
        # Depthwise conv for edge detection (fokus spatial)
        self.dw = nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False)
        
        # Pointwise conv for output projection
        self.pw = Conv(c1, c2, k=1, s=1, act=True)
    
    def forward(self, x):
        """
        Forward pass through Shape Gate.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Features with enhanced edge/structure signal
        """
        identity = x
        
        # Edge detection via depthwise conv
        edge = self.dw(x)
        
        # Enhance edge: x + edge
        out = identity + edge
        
        # Pointwise projection
        out = self.pw(out)
        
        return out


class BGSuppressP3(nn.Module):
    """
    Background Suppression P3 (Light): Suppress large structures (background/nucleus) 
    while enhancing small details (BTA) di P3.
    
    Konsep operasi:
    - bg = DWConv(k=7) atau AvgPool (low-freq background estimate)
    - fg = x - bg (foreground detail)
    - gate = sigmoid(Conv1x1(fg)) (gate untuk detail kecil)
    - out = x + α * (fg * gate) (α learnable scalar, init 0.1)
    
    Intinya: struktur besar (bg) dikurangin, detail kecil (fg) dikuatin secara terkontrol.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels (default: same as input)
        kernel_size (int): Kernel size for background estimate (default: 7, can be 5)
        use_avgpool (bool): Use AvgPool instead of DWConv for bg (default: False)
        alpha_init (float): Initial value for learnable alpha (default: 0.1)
    """
    
    def __init__(self, c1, c2=None, kernel_size=7, use_avgpool=False, alpha_init=0.1):
        """Initialize BGSuppressP3."""
        super().__init__()
        from .conv import Conv, DWConv
        
        if c2 is None:
            c2 = c1
        
        self.kernel_size = kernel_size
        self.use_avgpool = use_avgpool
        
        # Background estimate: low-freq filter
        if use_avgpool:
            # Alternative: AvgPool for background
            padding = kernel_size // 2
            self.bg_estimate = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=padding)
        else:
            # DWConv for background (default)
            self.bg_estimate = DWConv(c1, c1, k=kernel_size, s=1, act=False)
        
        # Gate: Conv1x1 on foreground
        self.gate_conv = Conv(c1, c1, k=1, s=1, act=False)
        self.gate_act = nn.Sigmoid()
        
        # Learnable alpha (init kecil biar gak merusak awal training)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        
        # Output projection if channels differ
        if c1 != c2:
            self.proj = Conv(c1, c2, k=1, s=1, act=True)
        else:
            self.proj = None
    
    def forward(self, x):
        """
        Forward pass through BGSuppressP3.
        
        Args:
            x: Input tensor [B, C, H, W] (P3 feature)
        
        Returns:
            Features with suppressed background and enhanced small details
        """
        identity = x
        
        # Background estimate (low-freq)
        bg = self.bg_estimate(x)
        
        # Foreground detail (high-freq)
        fg = x - bg
        
        # Gate from foreground (detail kecil lebih lewat)
        gate = self.gate_act(self.gate_conv(fg))
        
        # Output: x + α * (fg * gate)
        # Struktur besar dikurangin, detail kecil dikuatin secara terkontrol
        x = identity + self.alpha * (fg * gate)
        
        # Project if needed
        if self.proj is not None:
            x = self.proj(x)
        
        return x


class EdgeLineEnhancement(nn.Module):
    """
    Edge/Line Enhancement Block: Enhance small edges/lines (especially for small rods).
    
    Grad-CAM shows only cell edges. This block enhances "small edges" correctly.
    
    Concept: Light high-pass + channel attention.
    
    Architecture:
    - hp = DWConv3x3(x) - AvgPool(x) (Laplacian-ish high-pass)
    - gate = sigmoid(MLP(GAP(hp)))
    - out = x + hp * gate
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels (default: same as input)
        reduction (int): MLP reduction ratio (default: 4)
    """
    
    def __init__(self, c1, c2=None, reduction=4):
        """Initialize Edge/Line Enhancement Block."""
        super().__init__()
        from .conv import Conv, DWConv
        from .transformer import MLP
        
        if c2 is None:
            c2 = c1
        
        # High-pass: DWConv3x3 - AvgPool (Laplacian-ish)
        self.dwconv = DWConv(c1, c1, k=3, s=1, act=False)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        
        # Channel attention: GAP + MLP
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        mlp_hidden = max(c1 // reduction, 8)  # At least 8 hidden units
        self.mlp = MLP(c1, mlp_hidden, c1, num_layers=2, act=nn.ReLU, sigmoid=True)
        
        # Output projection if channels differ
        if c1 != c2:
            self.proj = Conv(c1, c2, k=1, s=1, act=True)
        else:
            self.proj = None
    
    def forward(self, x):
        """
        Forward pass through Edge/Line Enhancement Block.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Enhanced features with emphasized small edges/lines
        """
        identity = x
        
        # High-pass: DWConv3x3 - AvgPool (Laplacian-ish)
        hp_dw = self.dwconv(x)
        hp_avg = self.avgpool(x)
        hp = hp_dw - hp_avg  # High-pass residual
        
        # Channel attention: GAP + MLP
        hp_gap = self.gap(hp)  # [B, C, 1, 1]
        hp_gap = hp_gap.squeeze(-1).squeeze(-1)  # [B, C]
        gate = self.mlp(hp_gap)  # [B, C]
        gate = gate.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        
        # Residual: x + hp * gate
        x = identity + hp * gate
        
        # Project if needed
        if self.proj is not None:
            x = self.proj(x)
        
        return x


class AggressiveBackgroundSuppression(nn.Module):
    """
    Aggressive Background Suppression: Multi-stage suppression untuk P3/P4.
    Lebih agresif dari BSG biasa - langsung subtract background, bukan hanya gate.
    
    Architecture:
    - bg1 = DWConv(k=9, stride=1) (low-pass 1)
    - bg2 = DWConv(k=7, stride=1) (low-pass 2) 
    - bg_combined = (bg1 + bg2) / 2
    - fg = x - bg_combined (high-pass)
    - threshold = adaptive threshold based on fg magnitude
    - out = fg * sigmoid(threshold) + x * (1 - sigmoid(threshold)) * 0.3
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels (default: same as input)
        suppression_strength (float): How much to suppress background (0-1, default: 0.7)
    """
    
    def __init__(self, c1, c2=None, suppression_strength=0.7):
        """Initialize Aggressive Background Suppression."""
        super().__init__()
        from .conv import Conv, DWConv
        
        if c2 is None:
            c2 = c1
        
        self.suppression_strength = suppression_strength
        
        # Multi-scale low-pass filters
        self.bg_conv1 = DWConv(c1, c1, k=9, s=1, act=False)  # Large kernel
        self.bg_conv2 = DWConv(c1, c1, k=7, s=1, act=False)  # Medium kernel
        
        # Adaptive threshold for suppression
        self.threshold_conv = nn.Sequential(
            Conv(c1, c1, k=1, s=1, act=False),
            nn.Sigmoid()
        )
        
        # Output projection if channels differ
        if c1 != c2:
            self.proj = Conv(c1, c2, k=1, s=1, act=True)
        else:
            self.proj = None
    
    def forward(self, x):
        """
        Forward pass through Aggressive Background Suppression.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Features with aggressively suppressed large activations
        """
        identity = x
        
        # Multi-scale background extraction
        bg1 = self.bg_conv1(x)
        bg2 = self.bg_conv2(x)
        bg_combined = (bg1 + bg2) / 2.0
        
        # High-pass: foreground = original - background
        fg = x - bg_combined
        
        # Adaptive threshold based on foreground magnitude
        fg_magnitude = torch.abs(fg)
        threshold = self.threshold_conv(fg_magnitude)
        
        # Aggressive suppression: keep foreground, suppress background
        # fg gets full weight, background gets reduced weight
        # Add small residual to prevent complete zeroing
        x = fg * threshold + identity * (1 - threshold) * (1 - self.suppression_strength) + identity * 0.1
        
        # Project if needed
        if self.proj is not None:
            x = self.proj(x)
        
        return x


class CrossScaleSuppression(nn.Module):
    """
    Cross-Scale Background Suppression: P3 dan P4 saling reference untuk suppress background.
    P4 (coarser) membantu identify large structures di P3 (finer).
    Returns only P3 (main focus) - P4 can be suppressed separately.
    
    Architecture:
    - P4_down = Downsample(P4) to match P3 spatial size
    - bg_mask = sigmoid(Conv(P4_down)) - identifies large structures
    - P3_suppressed = P3 * (1 - bg_mask * strength)
    
    Args:
        c_p3 (int): P3 channels
        c_p4 (int): P4 channels
        c_out (int): Output channels
        suppression_strength (float): Suppression strength (default: 0.8)
    """
    
    def __init__(self, c_p3, c_p4, c_out, suppression_strength=0.8):
        """Initialize Cross-Scale Suppression."""
        super().__init__()
        from .conv import Conv
        
        self.suppression_strength = suppression_strength
        
        # P4 → P3: Downsample and process
        self.p4_to_p3 = Conv(c_p4, c_p3, k=1, s=1, act=True)
        
        # Background mask from P4
        self.bg_mask_conv = nn.Sequential(
            Conv(c_p3, c_p3, k=3, s=1, act=False),
            nn.Sigmoid()
        )
        
        # Output projection
        self.proj_p3 = Conv(c_p3, c_out, k=1, s=1, act=True)
    
    def forward(self, x):
        """
        Forward pass through Cross-Scale Suppression.
        
        Args:
            x: List of 2 tensors [P3, P4]
        
        Returns:
            Suppressed P3 features (single tensor)
        """
        if isinstance(x, (list, tuple)) and len(x) == 2:
            p3, p4 = x
        else:
            raise ValueError(f"CrossScaleSuppression expects list of 2 tensors [P3, P4], got {type(x)}")
        
        # P4 → P3: Downsample P4 to match P3 spatial size
        _, _, h3, w3 = p3.shape
        p4_down = F.interpolate(p4, size=(h3, w3), mode='bilinear', align_corners=False)
        p4_down = self.p4_to_p3(p4_down)
        
        # Background mask from P4 (identifies large structures)
        bg_mask_p3 = self.bg_mask_conv(p4_down)
        
        # Suppress P3: reduce activations where P4 shows large structures
        # Add small residual to prevent complete zeroing
        p3_suppressed = p3 * (1 - bg_mask_p3 * self.suppression_strength) + p3 * 0.1
        p3_out = self.proj_p3(p3_suppressed)
        
        return p3_out


class MultiScaleEdgeEnhancement(nn.Module):
    """
    Multi-Scale Edge Enhancement: Enhance edges di multiple scales untuk rod kecil.
    Lebih agresif dari ELEB biasa - multiple high-pass filters + spatial attention.
    
    Architecture:
    - hp1 = DWConv3x3(x) - AvgPool3x3(x)
    - hp2 = DWConv5x5(x) - AvgPool5x5(x)  
    - hp_combined = concat([hp1, hp2]) → Conv1x1
    - spatial_attn = sigmoid(Conv1x1(GAP(hp_combined)))
    - channel_attn = sigmoid(MLP(GAP(hp_combined)))
    - out = x + hp_combined * spatial_attn * channel_attn * 2.0
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels (default: same as input)
        enhancement_strength (float): Edge enhancement strength (default: 2.0)
    """
    
    def __init__(self, c1, c2=None, enhancement_strength=2.0):
        """Initialize Multi-Scale Edge Enhancement."""
        super().__init__()
        from .conv import Conv, DWConv
        from .transformer import MLP
        
        if c2 is None:
            c2 = c1
        
        self.enhancement_strength = enhancement_strength
        
        # Multi-scale high-pass filters
        self.dwconv3 = DWConv(c1, c1, k=3, s=1, act=False)
        self.avgpool3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        
        self.dwconv5 = DWConv(c1, c1, k=5, s=1, act=False)
        self.avgpool5 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        
        # Combine multi-scale edges
        self.fuse = Conv(c1 * 2, c1, k=1, s=1, act=True)
        
        # Spatial attention: where to enhance
        self.spatial_attn = nn.Sequential(
            Conv(c1, c1, k=1, s=1, act=False),
            nn.Sigmoid()
        )
        
        # Channel attention: which channels to enhance
        self.gap = nn.AdaptiveAvgPool2d(1)
        mlp_hidden = max(c1 // 4, 8)
        self.channel_attn = MLP(c1, mlp_hidden, c1, num_layers=2, act=nn.ReLU, sigmoid=True)
        
        # Output projection if channels differ
        if c1 != c2:
            self.proj = Conv(c1, c2, k=1, s=1, act=True)
        else:
            self.proj = None
    
    def forward(self, x):
        """
        Forward pass through Multi-Scale Edge Enhancement.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Enhanced features with multi-scale edge emphasis
        """
        identity = x
        
        # Multi-scale high-pass
        hp3_dw = self.dwconv3(x)
        hp3_avg = self.avgpool3(x)
        hp3 = hp3_dw - hp3_avg  # Scale 3x3
        
        hp5_dw = self.dwconv5(x)
        hp5_avg = self.avgpool5(x)
        hp5 = hp5_dw - hp5_avg  # Scale 5x5
        
        # Combine multi-scale edges
        hp_combined = torch.cat([hp3, hp5], dim=1)  # [B, 2C, H, W]
        hp_combined = self.fuse(hp_combined)  # [B, C, H, W]
        
        # Spatial attention: where to enhance
        spatial_gate = self.spatial_attn(hp_combined)
        
        # Channel attention: which channels to enhance
        hp_gap = self.gap(hp_combined)  # [B, C, 1, 1]
        hp_gap = hp_gap.squeeze(-1).squeeze(-1)  # [B, C]
        channel_gate = self.channel_attn(hp_gap)  # [B, C]
        channel_gate = channel_gate.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        
        # Enhanced output: x + edges * attention * strength
        x = identity + hp_combined * spatial_gate * channel_gate * self.enhancement_strength
        
        # Project if needed
        if self.proj is not None:
            x = self.proj(x)
        
        return x


class FSNeck(nn.Module):
    """
    Frequency-Separated Neck (FS-Neck) untuk P3.
    
    Masalah yang disasar: Grad-CAM lu panas ke blob besar (low-frequency).
    Ide: di neck P3, pisahkan komponen low/high frequency, lalu prioritaskan high-frequency untuk P3.
    
    Architecture:
    - low = AvgPool(x, k=5/7) atau DWConv(k=7) (low-pass)
    - high = x - low (high-pass)
    - out = Conv([x, high]) atau out = x + Conv(high)
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels (default: same as input)
        kernel_size (int): Kernel size for low-pass filter (default: 7, can be 5)
        use_avgpool (bool): Use AvgPool instead of DWConv for low-pass (default: False)
    """
    
    def __init__(self, c1, c2=None, kernel_size=7, use_avgpool=False):
        """Initialize Frequency-Separated Neck."""
        super().__init__()
        from .conv import Conv, DWConv
        
        if c2 is None:
            c2 = c1
        
        self.kernel_size = kernel_size
        self.use_avgpool = use_avgpool
        
        # Low-pass filter: extract low-frequency (large blobs)
        if use_avgpool:
            self.low_pass = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        else:
            self.low_pass = DWConv(c1, c1, k=kernel_size, s=1, act=False)
        
        # High-frequency processing: prioritize high-freq components
        self.high_conv = Conv(c1, c1, k=1, s=1, act=True)
        
        # Fusion: combine original, low, and high
        self.fusion = Conv(c1 * 2, c2, k=1, s=1, act=True)
    
    def forward(self, x):
        """
        Forward pass through Frequency-Separated Neck.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Features with high-frequency prioritized for small object detection
        """
        identity = x
        
        # Low-frequency component (large blobs/background)
        low = self.low_pass(x)
        
        # High-frequency component (small objects/details)
        high = x - low
        
        # Process high-frequency
        high_processed = self.high_conv(high)
        
        # Fusion: concat original and high-freq, prioritize high-freq
        x_concat = torch.cat([identity, high_processed], dim=1)
        x = self.fusion(x_concat)
        
        return x


class DIFuse(nn.Module):
    """
    Learnable P2→P3 Detail Injection (DI-Fuse) dengan Gate.
    
    Masalah: BTA itu borderline P2 detail; P3 doang kadang telat.
    Ide: inject detail dari P2 ke P3 head tapi digate supaya noise stain nggak kebawa.
    
    Architecture:
    - Ambil P2 feat (stride /4), downsample ke /8: P2toP3 = Conv(stride=2)
    - Hitung gate: g = sigmoid(Conv1x1(P3)) atau sigmoid(Conv1x1(P2toP3))
    - Fuse: P3 = P3 + α * (g * P2toP3) (α learnable kecil)
    
    Args:
        c_p2 (int): P2 channels
        c_p3 (int): P3 channels
        c_out (int): Output channels (default: same as P3)
        alpha_init (float): Initial value for learnable alpha (default: 0.1)
        gate_from_p3 (bool): Compute gate from P3 instead of P2toP3 (default: True)
    """
    
    def __init__(self, c_p2, c_p3, c_out=None, alpha_init=0.1, gate_from_p3=True):
        """Initialize Detail Injection Fuse."""
        super().__init__()
        from .conv import Conv
        
        if c_out is None:
            c_out = c_p3
        
        self.c_p2_expected = c_p2
        self.c_p3_expected = c_p3
        self.c_out = c_out
        self.gate_from_p3 = gate_from_p3
        self.alpha_init = alpha_init
        self._initialized = False
        
        # Will be initialized lazily in first forward pass with actual channels
        self.p2_align = None
        self.p2_downsample = None
        self.gate_conv = None
        self.proj = None
        
        # Learnable fusion weight (alpha) - always initialized
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
    
    def forward(self, x):
        """
        Forward pass through Detail Injection Fuse.
        
        Args:
            x: List of 2 tensors [P3, P2] or single tensor P3 (if P2 is pre-processed)
        
        Returns:
            Enhanced P3 features with gated P2 detail injection
        """
        if isinstance(x, (list, tuple)) and len(x) == 2:
            p3, p2 = x
        else:
            raise ValueError(f"DIFuse expects list of 2 tensors [P3, P2], got {type(x)}")
        
        # Auto-detect actual channels from input tensors (handle width scaling)
        _, c_p2_actual, _, _ = p2.shape
        _, c_p3_actual, _, _ = p3.shape
        
        # Lazy initialization with actual channels
        if not self._initialized or c_p2_actual != self.c_p2_expected or c_p3_actual != self.c_p3_expected:
            from .conv import Conv
            
            # Update expected channels
            self.c_p2_expected = c_p2_actual
            self.c_p3_expected = c_p3_actual
            
            # P2 → P3: Channel alignment first, then downsample
            self.p2_align = Conv(c_p2_actual, c_p3_actual, k=1, s=1, act=True).to(p2.device)
            self.p2_downsample = Conv(c_p3_actual, c_p3_actual, k=3, s=2, act=True).to(p2.device)
            
            # Register as submodules so parameters are tracked
            self.add_module('p2_align', self.p2_align)
            self.add_module('p2_downsample', self.p2_downsample)
            
            # Gate: learnable attention to control injection
            if self.gate_from_p3:
                self.gate_conv = nn.Sequential(
                    Conv(c_p3_actual, c_p3_actual, k=1, s=1, act=False),
                    nn.Sigmoid()
                ).to(p3.device)
            else:
                self.gate_conv = nn.Sequential(
                    Conv(c_p3_actual, c_p3_actual, k=1, s=1, act=False),
                    nn.Sigmoid()
                ).to(p2.device)
            
            self.add_module('gate_conv', self.gate_conv)
            
            # Output projection if needed
            if c_p3_actual != self.c_out:
                self.proj = Conv(c_p3_actual, self.c_out, k=1, s=1, act=True).to(p3.device)
                self.add_module('proj', self.proj)
            else:
                self.proj = None
            
            self._initialized = True
        
        # P2 → P3: Channel alignment and downsample
        p2_aligned = self.p2_align(p2)
        p2_to_p3 = self.p2_downsample(p2_aligned)
        
        # Ensure spatial match (safety check - handle case where stride=2 doesn't exactly match)
        _, _, h3, w3 = p3.shape
        if p2_to_p3.shape[2:] != (h3, w3):
            p2_to_p3 = F.interpolate(p2_to_p3, size=(h3, w3), mode='bilinear', align_corners=False)
        
        # Compute gate
        if self.gate_from_p3:
            gate = self.gate_conv(p3)
        else:
            gate = self.gate_conv(p2_to_p3)
        
        # Gated fusion: P3 + α * (gate * P2toP3)
        p3_enhanced = p3 + self.alpha * (gate * p2_to_p3)
        
        # Project if needed
        if self.proj is not None:
            p3_enhanced = self.proj(p3_enhanced)
        
        return p3_enhanced


class SADHead(nn.Module):
    """
    Scale-Aware Dynamic NMS Head (SAD-Head) untuk P3.
    
    Bukan NMS "di luar", tapi head menghasilkan prediksi dengan prior tiny.
    Masalah: model sering high-conf di blob besar; P3 harus "anti-box-besar".
    Ide: di P3 head, tambahkan size prior gate yang menurunkan confidence untuk bbox besar.
    
    Architecture:
    - Reg branch memprediksi box distribution/width-height (atau proxy)
    - Buat penalty: pen = sigmoid(Conv(proxy_wh))
    - Output cls: cls = cls * (1 - β*pen) (β learnable)
    - Jadi P3 cenderung confident pada bbox kecil
    
    Args:
        c1 (int): Input channels (P3 feature channels)
        reg_max (int): DFL reg_max (default: 16)
        beta_init (float): Initial value for learnable beta penalty (default: 0.5)
    """
    
    def __init__(self, c1, reg_max=16, beta_init=0.5):
        """Initialize Scale-Aware Dynamic Head."""
        super().__init__()
        from .conv import Conv
        
        self.reg_max = reg_max
        
        # Proxy for box size: predict width-height proxy from features
        # Use a simple conv to estimate "box size proxy"
        self.size_proxy_conv = nn.Sequential(
            Conv(c1, c1 // 2, k=3, s=1, act=True),
            Conv(c1 // 2, 1, k=1, s=1, act=False)  # Single channel: box size proxy
        )
        
        # Penalty computation: larger boxes get higher penalty
        self.penalty_conv = nn.Sequential(
            Conv(1, 1, k=3, s=1, act=False),
            nn.Sigmoid()
        )
        
        # Learnable penalty strength (beta)
        self.beta = nn.Parameter(torch.tensor(beta_init))
    
    def forward(self, x):
        """
        Forward pass through Scale-Aware Dynamic Head.
        
        Args:
            x: Input tensor [B, C, H, W] (P3 features)
        
        Returns:
            Modified features with size-aware penalty applied
            Note: This should be applied to cls logits before sigmoid in Detect head
        """
        # Compute box size proxy (larger values = larger boxes)
        size_proxy = self.size_proxy_conv(x)  # [B, 1, H, W]
        
        # Compute penalty (higher for larger boxes)
        penalty = self.penalty_conv(size_proxy)  # [B, 1, H, W]
        
        # Store penalty for later use in Detect head
        # We'll return both features and penalty
        # The penalty will be used to modulate cls logits
        return x, penalty
    
    def apply_penalty_to_cls(self, cls_logits, penalty):
        """
        Apply size-aware penalty to classification logits.
        
        Args:
            cls_logits: Classification logits [B, nc, H, W]
            penalty: Size penalty [B, 1, H, W]
        
        Returns:
            Penalized classification logits
        """
        # Expand penalty to match cls_logits channels
        # penalty: [B, 1, H, W] -> [B, nc, H, W]
        penalty_expanded = penalty.expand_as(cls_logits)
        
        # Apply penalty: cls = cls * (1 - β * penalty)
        # Larger boxes (higher penalty) get lower confidence
        cls_penalized = cls_logits * (1 - self.beta * penalty_expanded)
        
        return cls_penalized


class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive Feature Fusion (AFF) Block for learnable feature fusion.
    
    Ganti PAFPN biasa dengan fusion yang bisa "pilih" feature mana yang penting.
    P4 & P5 di-fuse dengan weight yang learnable.
    
    Architecture:
    - Upsample P5 ke size P4
    - Adaptive weighted fusion: (w_p4 * P4 + w_p5 * P5_upsampled) / (w_p4 + w_p5)
    - Fusion conv (1x1 + 3x3)
    - Local context extractor (depthwise conv)
    
    Args:
        c1 (int): P4 input channels (auto-inferred)
        c2 (int): Output channels
    """
    
    def __init__(self, c1, c2):
        """Initialize AdaptiveFeatureFusion for learnable feature fusion."""
        super().__init__()
        from .conv import Conv
        
        # Learnable fusion weight
        self.weight_p4 = nn.Parameter(torch.ones(1))
        self.weight_p5 = nn.Parameter(torch.ones(1))
        
        # Upsampling untuk P5
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Fusion conv
        self.fusion_conv = nn.Sequential(
            Conv(c1, c2, 1),
            Conv(c2, c2, 3)
        )
        
        # Local context extractor
        self.context = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, c2, 1),
        )
    
    def forward(self, x):
        """
        Forward pass through AdaptiveFeatureFusion.
        
        Args:
            x: List of 2 tensors [P5, P4] (P5 first, then P4)
               Note: In YAML, order is [P5_upsampled, P4], so we reverse it
        
        Returns:
            Fused features with adaptive weights
        """
        if isinstance(x, (list, tuple)) and len(x) == 2:
            p5, p4 = x  # x comes as [P5_upsampled, P4] from YAML
        else:
            raise ValueError(f"AdaptiveFeatureFusion expects list of 2 tensors [P5, P4], got {type(x)}")
        
        # Upsample P5 ke size P4 (if not already upsampled)
        _, _, h4, w4 = p4.shape
        if p5.shape[2:] != (h4, w4):
            p5_up = self.upsample(p5)
        else:
            p5_up = p5
        
        # Adaptive weighted fusion
        w_sum = self.weight_p4 + self.weight_p5 + 1e-4
        fused = (self.weight_p4 * p4 + self.weight_p5 * p5_up) / w_sum
        
        # Fusion conv
        out = self.fusion_conv(fused)
        
        # Add local context
        out = out + self.context(out)
        
        return out


class NoiseSuppression(nn.Module):
    """
    Noise Suppression Block (NSB) untuk mengurangi activation noise di low-level features.
    
    Mengurangi activation noise di P4/P5 dengan dual attention mechanism:
    - Channel attention: Deteksi feature channel mana yang noise
    - Spatial attention: Deteksi region mana yang tidak penting
    - Output = input × channel_mask × spatial_mask
    
    Architecture:
    - Channel attention: AvgPool + MaxPool → MLP → Sigmoid
    - Spatial attention: Mean + Max → Conv7x7 → Sigmoid
    - Combined suppression: x * channel_mask * spatial_mask
    
    Args:
        c1 (int): Input channels
        ratio (int): Channel reduction ratio untuk MLP (default: 4)
    """
    
    def __init__(self, c1, ratio=4):
        """Initialize NoiseSuppressionBlock for noise suppression."""
        super().__init__()
        
        # Channel attention untuk deteksi noise
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # MLP untuk generate suppression mask
        self.mlp = nn.Sequential(
            nn.Conv2d(c1*2, c1//ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1//ratio, c1, 1, bias=False)
        )
        
        # Spatial attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass through NoiseSuppressionBlock.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Suppressed features [B, C, H, W]
        """
        # Channel-wise suppression
        avg_feat = self.avg_pool(x)
        max_feat = self.max_pool(x)
        channel_mask = self.sigmoid(
            self.mlp(torch.cat([avg_feat, max_feat], 1))
        )
        
        # Spatial suppression
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_mask = self.spatial_conv(
            torch.cat([avg_spatial, max_spatial], 1)
        )
        
        # Combined suppression
        x = x * channel_mask * spatial_mask
        return x


class GlobalContextBlock(nn.Module):
    """
    Global Context (GC) Block untuk context augmentation di backbone.
    
    Lebih ringan dari Self-Attention tapi efektif untuk mAP50 karena 
    menangkap korelasi channel secara global.
    
    Architecture:
    - Global Average Pooling (GAP)
    - 1x1 Conv untuk context modeling
    - 1x1 Conv untuk transform
    - Broadcast dan add ke original features
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels (default: same as input)
        reduction (int): Channel reduction ratio (default: 4)
    """
    
    def __init__(self, c1, c2=None, reduction=4):
        """Initialize GlobalContextBlock for global context modeling."""
        super().__init__()
        from .conv import Conv
        
        if c2 is None:
            c2 = c1
        
        self.c2 = c2
        self.reduction = reduction
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Context modeling: 1x1 Conv untuk generate context
        self.context_conv = nn.Sequential(
            nn.Conv2d(c1, c1 // reduction, 1, bias=False),
            nn.BatchNorm2d(c1 // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // reduction, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass through GlobalContextBlock.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Enhanced features with global context [B, C2, H, W]
        """
        # Global Average Pooling
        context = self.gap(x)  # [B, C1, 1, 1]
        
        # Context modeling
        context = self.context_conv(context)  # [B, C2, 1, 1]
        
        # Broadcast dan multiply (bukan add, untuk gate mechanism)
        out = x * context
        
        return out


class LargeKernelConv(nn.Module):
    """
    Large Kernel Convolution untuk context augmentation.
    
    Ganti beberapa blok di stage terakhir backbone dengan Large Kernel (7x7 atau 9x9)
    untuk membantu model "melihat" hubungan antar fitur yang berjauhan.
    
    Architecture:
    - Large kernel conv (7x7 atau 9x9)
    - Optional: Dilated convolution untuk larger receptive field
    - BatchNorm + Activation
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        k (int): Kernel size (default: 7, bisa 7 atau 9)
        dilation (int): Dilation rate (default: 1, bisa 2 untuk larger RF)
    """
    
    def __init__(self, c1, c2, k=7, dilation=1):
        """Initialize LargeKernelConv for large receptive field."""
        super().__init__()
        from .conv import Conv
        
        padding = (k - 1) * dilation // 2
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, k, stride=1, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x):
        """Forward pass through LargeKernelConv."""
        return self.conv(x)


class BiFPN(nn.Module):
    """
    Weighted Bi-directional Feature Pyramid Network (BiFPN).
    
    Implement learnable weights saat melakukan fusion di Neck.
    Model bisa milih fitur mana (P3, P4, atau P5) yang paling berkontribusi.
    
    Architecture:
    - Top-down path: P5 → P4 → P3 dengan learnable weights
    - Bottom-up path: P3 → P4 → P5 dengan learnable weights
    - Each fusion uses weighted sum with learnable parameters
    
    Args:
        c3 (int): P3 channels
        c4 (int): P4 channels
        c5 (int): P5 channels
        c_out (int): Output channels (default: same as input)
    """
    
    def __init__(self, c3, c4, c5, c_out=None):
        """Initialize BiFPN for weighted bidirectional feature fusion."""
        super().__init__()
        from .conv import Conv
        
        if c_out is None:
            c_out = c4  # Default to P4 channels
        
        # Align channels
        self.align_p3 = Conv(c3, c_out, k=1, s=1, act=True) if c3 != c_out else nn.Identity()
        self.align_p4 = Conv(c4, c_out, k=1, s=1, act=True) if c4 != c_out else nn.Identity()
        self.align_p5 = Conv(c5, c_out, k=1, s=1, act=True) if c5 != c_out else nn.Identity()
        
        # Learnable weights untuk top-down path
        self.w_td_p5 = nn.Parameter(torch.ones(1))
        self.w_td_p4 = nn.Parameter(torch.ones(1))
        self.w_td_p3 = nn.Parameter(torch.ones(1))
        
        # Learnable weights untuk bottom-up path
        self.w_bu_p3 = nn.Parameter(torch.ones(1))
        self.w_bu_p4 = nn.Parameter(torch.ones(1))
        self.w_bu_p5 = nn.Parameter(torch.ones(1))
        
        # Upsampling dan downsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.Conv2d(c_out, c_out, 3, stride=2, padding=1, bias=False)
        
        # Fusion convs
        self.fusion_td_p4 = Conv(c_out * 2, c_out, k=3, s=1, act=True)
        self.fusion_td_p3 = Conv(c_out * 2, c_out, k=3, s=1, act=True)
        self.fusion_bu_p4 = Conv(c_out * 2, c_out, k=3, s=1, act=True)
        self.fusion_bu_p5 = Conv(c_out * 2, c_out, k=3, s=1, act=True)
    
    def forward(self, x):
        """
        Forward pass through BiFPN.
        
        Args:
            x: List of 3 tensors [P3, P4, P5]
        
        Returns:
            List of enhanced features [P3, P4, P5]
        """
        if isinstance(x, (list, tuple)) and len(x) == 3:
            p3, p4, p5 = x
        else:
            raise ValueError(f"BiFPN expects list of 3 tensors [P3, P4, P5], got {type(x)}")
        
        # Align channels
        p3 = self.align_p3(p3)
        p4 = self.align_p4(p4)
        p5 = self.align_p5(p5)
        
        # Top-down path: P5 → P4 → P3
        # P4 enhanced
        p5_up = self.upsample(p5)
        w_sum_td_p4 = self.w_td_p5 + self.w_td_p4 + 1e-4
        p4_td = (self.w_td_p5 * p5_up + self.w_td_p4 * p4) / w_sum_td_p4
        p4_td = self.fusion_td_p4(torch.cat([p5_up, p4], dim=1))
        
        # P3 enhanced
        p4_td_up = self.upsample(p4_td)
        w_sum_td_p3 = self.w_td_p4 + self.w_td_p3 + 1e-4
        p3_td = (self.w_td_p4 * p4_td_up + self.w_td_p3 * p3) / w_sum_td_p3
        p3_td = self.fusion_td_p3(torch.cat([p4_td_up, p3], dim=1))
        
        # Bottom-up path: P3 → P4 → P5
        # P4 enhanced
        p3_td_down = self.downsample(p3_td)
        w_sum_bu_p4 = self.w_bu_p3 + self.w_bu_p4 + 1e-4
        p4_bu = (self.w_bu_p3 * p3_td_down + self.w_bu_p4 * p4_td) / w_sum_bu_p4
        p4_bu = self.fusion_bu_p4(torch.cat([p3_td_down, p4_td], dim=1))
        
        # P5 enhanced
        p4_bu_down = self.downsample(p4_bu)
        w_sum_bu_p5 = self.w_bu_p4 + self.w_bu_p5 + 1e-4
        p5_bu = (self.w_bu_p4 * p4_bu_down + self.w_bu_p5 * p5) / w_sum_bu_p5
        p5_bu = self.fusion_bu_p5(torch.cat([p4_bu_down, p5], dim=1))
        
        return [p3_td, p4_bu, p5_bu]


class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA) Module.
    
    Lightweight channel attention with adaptive kernel size.
    Better than SE for small models with minimal parameters.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels (for compatibility, same as c1)
        gamma (int): Gamma parameter for kernel size calculation (default: 2)
        b (int): Bias parameter for kernel size calculation (default: 1)
    """
    
    def __init__(self, c1, c2=None, gamma=2, b=1):
        """Initialize ECA attention."""
        super().__init__()
        c2 = c2 or c1
        
        # Adaptive kernel size: k = |log2(C) + b) / gamma|_odd
        t = int(abs((math.log(c1, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Forward pass through ECA.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Enhanced features with channel attention
        """
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class LightAttention(nn.Module):
    """
    Lightweight Attention Wrapper.
    
    Supports SimAM (parameter-free) or ECA (minimal parameters).
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels (for compatibility)
        attn_type (str): 'simam' or 'eca' (default: 'simam')
    """
    
    def __init__(self, c1, c2=None, attn_type='simam'):
        """Initialize LightAttention."""
        super().__init__()
        c2 = c2 or c1
        attn_type = attn_type.lower()
        
        if attn_type == 'simam':
            self.attn = SimAM(c1, c2)
        elif attn_type == 'eca':
            self.attn = ECA(c1, c2)
        else:
            raise ValueError(f"Unknown attention type: {attn_type}. Use 'simam' or 'eca'")
    
    def forward(self, x):
        """Forward pass through lightweight attention."""
        return self.attn(x)


class SPDDown(nn.Module):
    """
    SPD Downsampling Module.
    
    Replaces stride=2 conv with Space-to-Depth + Conv mixing.
    Preserves spatial details better than standard downsampling.
    
    Architecture:
    1. Space-to-Depth: [B, C, H, W] -> [B, C*4, H/2, W/2]
    2. Conv mixing: [B, C*4, H/2, W/2] -> [B, C2, H/2, W/2]
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        k (int): Kernel size for mixing conv (default: 3)
        act (bool): Activation (default: True)
    """
    
    def __init__(self, c1, c2, k=3, act=True):
        """Initialize SPDDown module."""
        super().__init__()
        
        # Conv mixing after space-to-depth
        # Input: c1*4 channels (after SPD), Output: c2 channels
        self.conv_mix = Conv(c1 * 4, c2, k=k, s=1, act=act)
    
    def forward(self, x):
        """
        Forward pass: Space-to-Depth + Conv mixing.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Downsampled features [B, C2, H/2, W/2]
        """
        # Space-to-Depth: Split spatial dimension into 4 parts
        # [B, C, H, W] -> [B, C*4, H/2, W/2]
        x = torch.cat(
            [x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1
        )
        
        # Conv mixing: [B, C*4, H/2, W/2] -> [B, C2, H/2, W/2]
        return self.conv_mix(x)


class SPD_A_Block(nn.Module):
    """
    SPD + Attention Block.
    
    Complete block: SPDDown -> ConvMix -> LightAttention -> MainBlock (C3/C2f/A2C2f).
    
    Architecture:
    1. SPDDown: Space-to-Depth downsampling
    2. ConvMix: Channel mixing (optional, can be 1x1 or 3x3)
    3. LightAttention: SimAM or ECA
    4. MainBlock: C3/C2f/A2C2f for semantic building
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        block_type (str): 'C3', 'C2f', 'A2C2f', or 'C3k2' (default: 'C3k2')
        n (int): Number of blocks in main module (default: 1)
        attn_type (str): 'simam' or 'eca' (default: 'simam')
        mix_k (int): Kernel size for mixing conv (default: 3, use 1 for compression)
        shortcut (bool): Shortcut in main block (default: True)
    """
    
    def __init__(self, c1, c2, block_type='C3k2', n=1, attn_type='simam', mix_k=3, shortcut=True):
        """Initialize SPD_A_Block."""
        super().__init__()
        
        # Step 1: SPDDown
        self.spd_down = SPDDown(c1, c2, k=mix_k, act=True)
        
        # Step 2: LightAttention (after conv mixing in SPDDown)
        self.attn = LightAttention(c2, c2, attn_type=attn_type)
        
        # Step 3: Main block for semantic building
        block_type = block_type.lower()
        if block_type == 'c3':
            self.main_block = C3(c2, c2, n, shortcut=shortcut)
        elif block_type == 'c2f':
            self.main_block = C2f(c2, c2, n, shortcut=shortcut)
        elif block_type == 'a2c2f':
            self.main_block = A2C2f(c2, c2, n, shortcut=shortcut)
        elif block_type == 'c3k2':
            # C3k2 signature: (c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True)
            self.main_block = C3k2(c2, c2, n, c3k=False, shortcut=shortcut)
        else:
            raise ValueError(f"Unknown block type: {block_type}. Use 'C3', 'C2f', 'A2C2f', or 'C3k2'")
    
    def forward(self, x):
        """
        Forward pass: SPD -> ConvMix -> Attention -> MainBlock.
        
        Args:
            x: Input tensor [B, C1, H, W]
        
        Returns:
            Processed features [B, C2, H/2, W/2]
        """
        # SPDDown (includes space-to-depth + conv mixing)
        x = self.spd_down(x)
        
        # LightAttention
        x = self.attn(x)
        
        # Main block for semantic building
        x = self.main_block(x)
        
        return x


class E_ELAN(nn.Module):
    """
    Enhanced ELAN (E-ELAN) for Scale Sequence Refinement.
    
    Architecture with multiple branches and skip connections:
    - Main path: 1x1 Conv2D → 3x3 Conv2D → 3x3 Conv2D
    - Branch path: 1x1 Conv2D → 3x3 Conv2D
    - Multiple skip connections for gradient flow
    - Final concatenation and 1x1 Conv2D output
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        e (float): Expansion ratio (default: 0.5)
    """
    
    def __init__(self, c1, c2, e=0.5):
        """Initialize E-ELAN for scale sequence refinement."""
        super().__init__()
        from .conv import Conv
        
        c_ = int(c1 * e)  # Hidden channels
        
        # Main path: 1x1 Conv2D → 3x3 Conv2D → 3x3 Conv2D
        self.cv1_main = Conv(c1, c_, k=1, s=1, act=True)
        self.cv2_main = Conv(c_, c_, k=3, s=1, act=True)
        self.cv3_main = Conv(c_, c_, k=3, s=1, act=True)
        
        # Branch path: 1x1 Conv2D → 3x3 Conv2D
        self.cv1_branch = Conv(c1, c_, k=1, s=1, act=True)
        self.cv2_branch = Conv(c_, c_, k=3, s=1, act=True)
        
        # Final output: 1x1 Conv2D
        # Concatenation: cv1_main + cv2_main + cv3_main + cv2_branch = 4 * c_
        self.cv_out = Conv(4 * c_, c2, k=1, s=1, act=True)
        
    def forward(self, x):
        """
        Forward pass through E-ELAN.
        
        Process:
        1. Main path: cv1_main → cv2_main (with skip from cv1_main) → cv3_main
        2. Branch path: cv1_branch → cv2_branch (with skip from cv1_branch and cv2_main)
        3. Concatenate all outputs
        4. Final 1x1 Conv2D
        """
        # Main path
        x1 = self.cv1_main(x)  # [B, c_, H, W]
        x2 = self.cv2_main(x1)  # [B, c_, H, W] (skip connection from x1)
        x2 = x2 + x1  # Skip connection: cv1_main → cv2_main
        x3 = self.cv3_main(x2)  # [B, c_, H, W]
        
        # Branch path
        x4 = self.cv1_branch(x)  # [B, c_, H, W]
        x5 = self.cv2_branch(x4)  # [B, c_, H, W] (skip connection from x4)
        x5 = x5 + x4  # Skip connection: cv1_branch → cv2_branch
        x5 = x5 + x2  # Skip connection: cv2_main → cv2_branch
        
        # Concatenate all outputs: [x1, x2, x3, x5]
        x_cat = torch.cat([x1, x2, x3, x5], dim=1)  # [B, 4*c_, H, W]
        
        # Final output
        x_out = self.cv_out(x_cat)  # [B, c2, H, W]
        
        return x_out


class ASSN(nn.Module):
    """
    Attention-based Scale Sequence Network (ASSN).
    
    Architecture:
    - P5: 1x1 Conv2D → Upscaling
    - P4: 1x1 Conv2D → Upscaling
    - P3A: Channel Attention → Spatial Attention
    - P3 Attention Mapping: Broadcast and multiply with P5 and P4
    - Concatenate weighted features to form Scale Sequence
    - E-ELAN for Scale Sequence Refinement
    - Element-wise summation with original P3B
    
    Args:
        c_p5 (int): P5 channels
        c_p4 (int): P4 channels
        c_p3 (int): P3 channels
        c_out (int): Output channels (default: same as P3)
    """
    
    def __init__(self, c_p5, c_p4, c_p3, c_out=None):
        """Initialize ASSN for attention-based scale sequence fusion."""
        super().__init__()
        from .conv import Conv, ChannelAttention, SpatialAttention
        
        if c_out is None:
            c_out = c_p3
        
        self.c_out = c_out  # Store for use in forward
        
        # Align all channels to P3 channels for fusion
        c_align = c_p3
        
        # P5 processing: 1x1 Conv2D → Upscaling (2x for P5→P4, then handled in forward)
        self.p5_conv = Conv(c_p5, c_align, k=1, s=1, act=True)
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # P4 processing: 1x1 Conv2D → Upscaling (2x for P4→P3)
        self.p4_conv = Conv(c_p4, c_align, k=1, s=1, act=True)
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # P3A processing: Channel Attention → Spatial Attention
        self.p3_channel_attn = ChannelAttention(c_p3)
        self.p3_spatial_attn = SpatialAttention(kernel_size=7)
        
        # E-ELAN for Scale Sequence Refinement
        # After concat: 3 * c_align channels (P5, P4, P3A all weighted)
        self.e_elan = E_ELAN(3 * c_align, c_out, e=0.5)
        
    def forward(self, x):
        """
        Forward pass through ASSN.
        
        Args:
            x: List of 3 tensors [P5, P4, P3A]
               P5: [B, c_p5, H5, W5]
               P4: [B, c_p4, H4, W4]
               P3A: [B, c_p3, H3, W3]
        
        Returns:
            Enhanced P3B features: [B, c_out, H3, W3]
        """
        if isinstance(x, (list, tuple)) and len(x) == 3:
            p5, p4, p3a = x
        else:
            raise ValueError(f"ASSN expects list of 3 tensors [P5, P4, P3A], got {type(x)}")
        
        # Process P5: 1x1 Conv2D → Upscaling (may need 2x upscale if coming from P5)
        p5_processed = self.p5_conv(p5)
        p5_up = self.p5_upsample(p5_processed)  # First 2x upscale
        
        # Process P4: 1x1 Conv2D → Upscaling (2x upscale)
        p4_processed = self.p4_conv(p4)
        p4_up = self.p4_upsample(p4_processed)  # 2x upscale
        
        # Process P3A: Channel Attention → Spatial Attention
        p3a_ca = self.p3_channel_attn(p3a)
        p3a_attn = self.p3_spatial_attn(p3a_ca)  # [B, c_p3, H3, W3]
        
        # Ensure spatial dimensions match P3A (adaptive upscaling)
        _, _, h3, w3 = p3a_attn.shape
        if p5_up.shape[2:] != (h3, w3):
            # P5 might need additional upscaling (if it was at P5 size, need 4x total)
            p5_up = F.interpolate(p5_up, size=(h3, w3), mode='bilinear', align_corners=False)
        if p4_up.shape[2:] != (h3, w3):
            # P4 should match, but handle edge cases
            p4_up = F.interpolate(p4_up, size=(h3, w3), mode='bilinear', align_corners=False)
        
        # P3 Attention Mapping: Broadcast and multiply with P5 and P4
        p5_weighted = p5_up * p3a_attn  # Element-wise multiplication
        p4_weighted = p4_up * p3a_attn  # Element-wise multiplication
        p3a_weighted = p3a_attn * p3a_attn  # Self-attention (or just use p3a_attn)
        
        # Concatenate weighted features to form Scale Sequence
        scale_sequence = torch.cat([p5_weighted, p4_weighted, p3a_weighted], dim=1)  # [B, 3*c_align, H3, W3]
        
        # E-ELAN for Scale Sequence Refinement
        p3b_refined = self.e_elan(scale_sequence)  # [B, c_out, H3, W3]
        
        # Element-wise summation with original P3A (if channels match)
        if p3a.shape[1] == self.c_out:
            p3b = p3b_refined + p3a
        else:
            # If channels don't match, just return refined features
            p3b = p3b_refined
        
        return p3b


# ==================== ECAP-YOLOv5 Blocks ====================

class ECABottleneck(nn.Module):
    """
    ECA Bottleneck Block.
    
    Residual bottleneck with ECA attention.
    Structure: CBS → ECA → CBS + skip connection
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        shortcut (bool): Use shortcut connection (default: True)
    """
    
    def __init__(self, c1, c2, shortcut=True):
        """Initialize ECABottleneck."""
        super().__init__()
        from .conv import Conv
        
        c_ = c2 // 2  # Hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)  # 1x1 conv
        self.eca = ECA(c_, c_)  # ECA attention
        self.cv2 = Conv(c_, c2, 3, 1)  # 3x3 conv
        self.add = shortcut and c1 == c2
        
    def forward(self, x):
        """Forward pass through ECABottleneck."""
        return x + self.cv2(self.eca(self.cv1(x))) if self.add else self.cv2(self.eca(self.cv1(x)))


class EAC31(nn.Module):
    """
    EAC31 Block: ECA before sum in bottleneck.
    
    Structure: CBS → (CBS → ECA → CBS) + skip → CBS
    EAC31_n: n*ECABottleneck in lower branch
    EAC31_nF: No bottleneck, just CBS branches
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        n (int): Number of ECABottleneck repeats (default: 1)
        use_bottleneck (bool): Use bottleneck structure (default: True)
    """
    
    def __init__(self, c1, c2, n=1, use_bottleneck=True):
        """Initialize EAC31."""
        super().__init__()
        from .conv import Conv
        
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, 1, 1)  # Initial CBS
        
        if use_bottleneck:
            # Lower branch: n*ECABottleneck
            self.m = nn.Sequential(*(ECABottleneck(c_, c_, True) for _ in range(n)))
        else:
            # EAC31_nF: No bottleneck, just CBS
            self.m = Conv(c_, c_, 3, 1)
        
        # Upper branch: CBS
        self.cv2 = Conv(c_, c_, 1, 1)
        
        # Final CBS
        self.cv3 = Conv(c_ * 2, c2, 1, 1)
        
    def forward(self, x):
        """Forward pass through EAC31."""
        y1 = self.cv1(x)
        return self.cv3(torch.cat([self.m(y1), self.cv2(y1)], 1))


class EAC32(nn.Module):
    """
    EAC32 Block: ECA after C3 module.
    
    Structure: CBS → (CBS branches) → Concat → CBS → ECA
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        n (int): Number of ECABottleneck repeats (default: 1)
        use_bottleneck (bool): Use bottleneck structure (default: True)
    """
    
    def __init__(self, c1, c2, n=1, use_bottleneck=True):
        """Initialize EAC32."""
        super().__init__()
        from .conv import Conv
        
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        
        if use_bottleneck:
            # Lower branch: n*ECABottleneck
            self.m = nn.Sequential(*(ECABottleneck(c_, c_, True) for _ in range(n)))
        else:
            # EAC32_nF: Upper branch with CBS → CBS → CBS → ECA, lower branch with CBS
            self.cv_upper1 = Conv(c_, c_, 1, 1)
            self.cv_upper2 = Conv(c_, c_, 3, 1)
            self.cv_upper3 = Conv(c_, c_, 3, 1)
            self.eca_upper = ECA(c_, c_)
            self.m = None
        
        self.cv2 = Conv(c_, c_, 1, 1)  # Lower branch for bottleneck version
        
        # Final CBS and ECA
        self.cv3 = Conv(c_ * 2, c2, 1, 1)
        self.eca = ECA(c2, c2)
        
    def forward(self, x):
        """Forward pass through EAC32."""
        y1 = self.cv1(x)
        
        if self.m is not None:
            # Bottleneck version
            y = torch.cat([self.m(y1), self.cv2(y1)], 1)
        else:
            # EAC32_nF version
            y_upper = self.eca_upper(self.cv_upper3(self.cv_upper2(self.cv_upper1(y1))))
            y_lower = self.cv2(y1)
            y = torch.cat([y_upper, y_lower], 1)
        
        y = self.cv3(y)
        return self.eca(y)


class EAC33(nn.Module):
    """
    EAC33 Block: ECA in part of C3 without bottleneck.
    
    Structure: CBS → (CBS → ECA → CBS) + CBS → Concat → CBS
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        n (int): Number of repeats (default: 1)
    """
    
    def __init__(self, c1, c2, n=1):
        """Initialize EAC33."""
        super().__init__()
        from .conv import Conv
        
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        
        # Upper branch: CBS → ECA → CBS
        self.cv_upper1 = Conv(c_, c_, 1, 1)
        self.eca = ECA(c_, c_)
        self.cv_upper2 = Conv(c_, c_, 3, 1)
        
        # Lower branch: CBS
        self.cv2 = Conv(c_, c_, 1, 1)
        
        # Final CBS
        self.cv3 = Conv(c_ * 2, c2, 1, 1)
        
    def forward(self, x):
        """Forward pass through EAC33."""
        y1 = self.cv1(x)
        y_upper = self.cv_upper2(self.eca(self.cv_upper1(y1)))
        y_lower = self.cv2(y1)
        return self.cv3(torch.cat([y_upper, y_lower], 1))


class EAC34(nn.Module):
    """
    EAC34 Block: ECA between conv modules in bottleneck.
    
    Structure: CBS → (CBS → ECA → CBS) + skip → CBS
    EAC34_n: n*ECABottleneck in upper branch
    EAC34_F: CBS → CBS → ECA → CBS in upper branch
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        n (int): Number of ECABottleneck repeats (default: 1)
        use_bottleneck (bool): Use bottleneck structure (default: True)
    """
    
    def __init__(self, c1, c2, n=1, use_bottleneck=True):
        """Initialize EAC34."""
        super().__init__()
        from .conv import Conv
        
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        
        if use_bottleneck:
            # Upper branch: n*ECABottleneck
            self.m = nn.Sequential(*(ECABottleneck(c_, c_, True) for _ in range(n)))
        else:
            # EAC34_F: CBS → CBS → ECA → CBS
            self.cv_upper1 = Conv(c_, c_, 1, 1)
            self.cv_upper2 = Conv(c_, c_, 3, 1)
            self.eca_upper = ECA(c_, c_)
            self.cv_upper3 = Conv(c_, c_, 3, 1)
            self.m = None
        
        # Lower branch: CBS
        self.cv2 = Conv(c_, c_, 1, 1)
        
        # Final CBS
        self.cv3 = Conv(c_ * 2, c2, 1, 1)
        
    def forward(self, x):
        """Forward pass through EAC34."""
        y1 = self.cv1(x)
        
        if self.m is not None:
            # Bottleneck version
            y_upper = self.m(y1)
        else:
            # EAC34_F version
            y_upper = self.cv_upper3(self.eca_upper(self.cv_upper2(self.cv_upper1(y1))))
        
        y_lower = self.cv2(y1)
        return self.cv3(torch.cat([y_upper, y_lower], 1))


class MEAC(nn.Module):
    """
    Multi-scale Enhanced Attention Convolution (MEAC).
    
    Uses Global Max Pooling (GMP) instead of GAP in channel attention.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels (default: same as input)
        reduction (int): Reduction ratio (default: 16)
    """
    
    def __init__(self, c1, c2=None, reduction=16):
        """Initialize MEAC."""
        super().__init__()
        c2 = c2 or c1
        
        c_mid = max(8, c1 // reduction)
        
        # Global Max Pooling
        self.gmp = nn.AdaptiveMaxPool2d(1)
        
        # Two FC layers (implemented as 1x1 conv)
        self.fc1 = nn.Conv2d(c1, c_mid, 1, bias=False)
        self.fc2 = nn.Conv2d(c_mid, c2, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """Forward pass through MEAC."""
        y = self.gmp(x)  # [B, C, 1, 1]
        y = self.fc1(y)  # [B, C_mid, 1, 1]
        y = self.fc2(y)  # [B, C, 1, 1]
        y = self.sigmoid(y)  # [B, C, 1, 1]
        return x * y.expand_as(x)


class AEAC(nn.Module):
    """
    Atrous Efficient Attention Convolution (AEAC).
    
    Uses atrous convolution in channel attention with different strides.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels (default: same as input)
        kernel_size (int): Kernel size for 1D conv (default: 3)
        stride (int): Stride for 1D conv (default: 1)
    """
    
    def __init__(self, c1, c2=None, kernel_size=3, stride=1):
        """Initialize AEAC."""
        super().__init__()
        c2 = c2 or c1
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 1D convolution with atrous (dilated)
        padding = (kernel_size - 1) // 2
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=stride, 
                               padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """Forward pass through AEAC."""
        y = self.gap(x)  # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv1d(y)  # [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class EAP(nn.Module):
    """
    Efficient Channel Attention Pyramid (EAP).
    
    Multi-scale pyramid using ECA with different kernel sizes.
    Uses ECA instead of max pooling in SPP.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        kernel_sizes (list): List of kernel sizes for ECA (default: [5, 9, 13])
    """
    
    def __init__(self, c1, c2, kernel_sizes=[5, 9, 13]):
        """Initialize EAP."""
        super().__init__()
        from .conv import Conv
        
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        
        # Three parallel ECA branches with different kernel sizes
        self.eca_branches = nn.ModuleList()
        for k in kernel_sizes:
            # Create ECA with specific kernel size
            eca = ECA(c_, c_)
            self.eca_branches.append(eca)
        
        # Final CBS
        self.cv2 = Conv(c_ * len(kernel_sizes), c2, 1, 1)
        
    def forward(self, x):
        """Forward pass through EAP."""
        y = self.cv1(x)
        # Apply ECA branches (note: ECA uses adaptive kernel, but we use different instances)
        y_branches = [eca(y) for eca in self.eca_branches]
        return self.cv2(torch.cat(y_branches, 1))


class MEAP(nn.Module):
    """
    Max Pooling Efficient Channel Attention Pyramid (MEAP).
    
    Uses MEAC (with GMP) instead of ECA in pyramid structure.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        kernel_sizes (list): List of kernel sizes (for reference, MEAC uses GMP) (default: [5, 9, 13])
    """
    
    def __init__(self, c1, c2, kernel_sizes=[5, 9, 13]):
        """Initialize MEAP."""
        super().__init__()
        from .conv import Conv
        
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        
        # Three parallel MEAC branches
        self.meac_branches = nn.ModuleList([MEAC(c_, c_) for _ in kernel_sizes])
        
        # Final CBS
        self.cv2 = Conv(c_ * len(kernel_sizes), c2, 1, 1)
        
    def forward(self, x):
        """Forward pass through MEAP."""
        y = self.cv1(x)
        y_branches = [meac(y) for meac in self.meac_branches]
        return self.cv2(torch.cat(y_branches, 1))


class AEAP(nn.Module):
    """
    Atrous Efficient Channel Attention Pyramid (AEAP).
    
    Uses AEAC with different strides to create pyramid structure.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        kernel_sizes (list): List of kernel sizes (default: [5, 9, 13])
        strides (list): List of strides for atrous conv (default: [1, 2, 3])
    """
    
    def __init__(self, c1, c2, kernel_sizes=[5, 9, 13], strides=[1, 2, 3]):
        """Initialize AEAP."""
        super().__init__()
        from .conv import Conv
        
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        
        # Three parallel AEAC branches with different strides
        self.aeac_branches = nn.ModuleList()
        for k, s in zip(kernel_sizes, strides):
            aeac = AEAC(c_, c_, kernel_size=k, stride=s)
            self.aeac_branches.append(aeac)
        
        # Final CBS
        self.cv2 = Conv(c_ * len(kernel_sizes), c2, 1, 1)
        
    def forward(self, x):
        """Forward pass through AEAP."""
        y = self.cv1(x)
        y_branches = [aeac(y) for aeac in self.aeac_branches]
        return self.cv2(torch.cat(y_branches, 1))