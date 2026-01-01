# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

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
        
        # Align channels
        p2_aligned = self.align_p2(p2_resized) if isinstance(self.align_p2, Conv) else p2_resized
        p3_aligned = p3_original
        p4_aligned = self.align_p4(p4_resized) if isinstance(self.align_p4, Conv) else p4_resized
        
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