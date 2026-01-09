# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Noise-Robust modules for YOLOv12 based on NR-CNN paper."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import autopad


class NoiseMapLayer(nn.Module):
    """
    Noise Map Layer: Detects various types of noise and generates noise map.
    
    Based on NR-CNN paper:
    - Detects impulse noise, missing samples, packet loss, damaged/tampered images
    - Generates noise map (binary mask: 0=noisy, 1=clean)
    - Output: 4-channel image (noise_map + RGB)
    
    Args:
        method (str): Noise detection method ('simple' or 'advanced')
    """
    
    def __init__(self, method='simple'):
        """Initialize NoiseMapLayer with detection method."""
        super().__init__()
        self.method = method
        
        if method == 'advanced':
            # Advanced: Multiple detection methods
            # Local consensus for impulse noise
            self.impulse_detector = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 3, padding=1),
                nn.Sigmoid()
            )
        else:
            # Simple: Statistical-based detection
            self.threshold = 0.1  # Adaptive threshold
    
    def forward(self, x):
        """
        Forward pass: Generate noise map from input image.
        
        Args:
            x: Input image [N, 3, H, W] (RGB)
        
        Returns:
            noise_map: Binary noise map [N, 1, H, W] (0=noisy, 1=clean)
        """
        if self.method == 'advanced':
            # Advanced detection using learned features
            noise_map = self.impulse_detector(x)
        else:
            # Simple: Detect outliers using local statistics
            # Compute local mean and std
            kernel_size = 5
            padding = kernel_size // 2
            local_mean = F.avg_pool2d(x, kernel_size, stride=1, padding=padding)
            local_std = torch.sqrt(
                F.avg_pool2d(x ** 2, kernel_size, stride=1, padding=padding) - local_mean ** 2 + 1e-6
            )
            
            # Detect outliers (noisy pixels)
            diff = torch.abs(x - local_mean)
            threshold = self.threshold * local_std.mean()
            noise_map = (diff < threshold).float().mean(dim=1, keepdim=True)  # [N, 1, H, W]
        
        return noise_map


class AdaptiveConv2d(nn.Module):
    """
    Adaptive Convolutional Layer: Drops noisy connections based on noise map.
    
    Based on NR-CNN paper:
    - Adaptive filtering: Drop connections to noisy pixels
    - Uses noise map to mask out noisy regions
    - Prevents noisy pixels from entering next layers
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        k (int): Kernel size (default: 3)
        s (int): Stride (default: 1)
        p (int): Padding (auto-calculated if None)
        g (int): Groups (default: 1)
        act (bool): Activation (default: True)
    """
    
    default_act = nn.SiLU()
    
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        """Initialize AdaptiveConv2d with noise-aware convolution."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    
    def forward(self, x, noise_map=None):
        """
        Forward pass with adaptive filtering.
        
        Args:
            x: Input features [N, C1, H, W]
            noise_map: Noise map [N, 1, H, W] (0=noisy, 1=clean). If None, standard conv.
        
        Returns:
            Output features [N, C2, H', W']
        """
        # If noise_map provided, mask noisy pixels
        if noise_map is not None:
            # Expand noise_map to match input channels
            noise_mask = noise_map.expand_as(x)  # [N, C1, H, W]
            # Mask noisy pixels (set to 0)
            x = x * noise_mask
        
        # Standard convolution
        return self.act(self.bn(self.conv(x)))


class AdaptiveMaxPool2d(nn.Module):
    """
    Adaptive Max Pooling: Skips noisy pixels in pooling operation.
    
    Based on NR-CNN paper:
    - Prevents noisy pixels from being selected in max pooling
    - Handles both low and high density noise
    - Uses noise map to guide pooling operation
    
    Args:
        kernel_size (int): Pooling kernel size
        stride (int): Pooling stride (default: None, same as kernel_size)
        padding (int): Padding (default: 0)
    """
    
    def __init__(self, kernel_size, stride=None, padding=0):
        """Initialize AdaptiveMaxPool2d."""
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
    
    def forward(self, x, noise_map=None):
        """
        Forward pass with adaptive pooling.
        
        Args:
            x: Input features [N, C, H, W]
            noise_map: Noise map [N, 1, H, W] (0=noisy, 1=clean). If None, standard pooling.
        
        Returns:
            Pooled features [N, C, H', W']
        """
        if noise_map is not None:
            # Mask noisy pixels with very negative values (so they won't be selected in max pooling)
            noise_mask = noise_map.expand_as(x)  # [N, C, H, W]
            # Set noisy pixels to very negative value
            x_masked = x * noise_mask + (1 - noise_mask) * (-1e10)
            return F.max_pool2d(x_masked, self.kernel_size, self.stride, self.padding)
        else:
            # Standard max pooling
            return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)


class AdaptiveAvgPool2d(nn.Module):
    """
    Adaptive Average Pooling: Skips noisy pixels in average pooling.
    
    Based on NR-CNN paper:
    - Excludes noisy pixels from average computation
    - Normalizes by number of clean pixels only
    
    Args:
        output_size: Output size (int or tuple)
    """
    
    def __init__(self, output_size):
        """Initialize AdaptiveAvgPool2d."""
        super().__init__()
        self.output_size = output_size
    
    def forward(self, x, noise_map=None):
        """
        Forward pass with adaptive average pooling.
        
        Args:
            x: Input features [N, C, H, W]
            noise_map: Noise map [N, 1, H, W] (0=noisy, 1=clean). If None, standard pooling.
        
        Returns:
            Pooled features [N, C, H', W']
        """
        if noise_map is not None:
            # Mask noisy pixels
            noise_mask = noise_map.expand_as(x)  # [N, C, H, W]
            x_masked = x * noise_mask
            
            # Adaptive average: sum of clean pixels / count of clean pixels
            x_sum = F.adaptive_avg_pool2d(x_masked, self.output_size)
            count_sum = F.adaptive_avg_pool2d(noise_mask, self.output_size)
            # Avoid division by zero
            count_sum = torch.clamp(count_sum, min=1e-6)
            return x_sum / count_sum
        else:
            # Standard adaptive average pooling
            return F.adaptive_avg_pool2d(x, self.output_size)

