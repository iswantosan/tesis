"""
Visualize the effect of ECA attention inside C3k2Attn on a single image.

Captures the feature map BEFORE and AFTER the attention module at a target
layer (default: layer 4, which is C3k2Attn at the P3 stage in yolo12s.yaml),
plus the per-channel ECA attention weights, and saves a multi-panel figure.

Usage:
    python visualize_attention_effect.py \
        --weights runs/detect/your_run/weights/best.pt \
        --image  tuberculosis-phone-0346.jpg \
        --layer  4 \
        --out    attention_effect.png

Requirements: matplotlib (already pulled by ultralytics), opencv-python, torch.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from ultralytics import YOLO
from ultralytics.nn.modules.block import C3k2Attn, ECA


def letterbox(img, new_size=640, color=(114, 114, 114)):
    """Resize with unchanged aspect ratio, pad to square."""
    h, w = img.shape[:2]
    r = new_size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_size - nh) // 2
    bottom = new_size - nh - top
    left = (new_size - nw) // 2
    right = new_size - nw - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
    return padded


def feat_to_heatmap(feat, target_size):
    """Channel-mean → normalize 0-1 → resize."""
    fmap = feat[0].mean(0).cpu().float().numpy()
    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
    fmap = cv2.resize(fmap, target_size, interpolation=cv2.INTER_CUBIC)
    return fmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True, help='Path to trained .pt weights')
    ap.add_argument('--image', required=True, help='Path to input image')
    ap.add_argument('--layer', type=int, default=4,
                    help='Layer index of C3k2Attn to inspect (default: 4 = P3)')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--out', default='attention_effect.png')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)

    # ---------- Load model ----------
    print(f'[1/4] Loading model from {args.weights}')
    yolo = YOLO(args.weights)
    model = yolo.model.to(device).eval()

    target = model.model[args.layer]
    if not isinstance(target, C3k2Attn):
        raise TypeError(
            f'Layer {args.layer} is {type(target).__name__}, not C3k2Attn. '
            f'Pass --layer pointing to a C3k2Attn block.'
        )
    eca = target.attn.attn  # C3k2Attn.attn (LightAttention) → .attn (ECA)
    if not isinstance(eca, ECA):
        raise TypeError(f'Expected ECA submodule, got {type(eca).__name__}')
    print(f'      Target: model.model[{args.layer}] = C3k2Attn '
          f'(channels: {eca.conv.in_channels} → reweighted via 1D conv k={eca.conv.kernel_size[0]})')

    # ---------- Load & preprocess image ----------
    print(f'[2/4] Loading image {args.image}')
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_lb = letterbox(img_rgb, args.imgsz)
    img_t = torch.from_numpy(img_lb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    img_t = img_t.to(device)

    # ---------- Register hooks ----------
    caps = {}

    def hook_attn(_m, inp, out):
        # Input to LightAttention = output of C3k2 (before attention)
        # Output of LightAttention = after attention
        caps['before'] = inp[0].detach()
        caps['after'] = out.detach()

    def hook_eca(m, inp, _out):
        # Recompute ECA weights so we can plot them
        x = inp[0]
        y = m.avg_pool(x)                              # [B, C, 1, 1]
        y = m.conv(y.squeeze(-1).transpose(-1, -2))    # [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = m.sigmoid(y)
        caps['eca_w'] = y[0, :, 0, 0].detach()         # [C]

    h1 = target.attn.register_forward_hook(hook_attn)
    h2 = eca.register_forward_hook(hook_eca)

    # ---------- Forward pass ----------
    print('[3/4] Running forward pass')
    with torch.no_grad():
        _ = model(img_t)
    h1.remove(); h2.remove()

    H, W = args.imgsz, args.imgsz
    heat_before = feat_to_heatmap(caps['before'], (W, H))
    heat_after = feat_to_heatmap(caps['after'], (W, H))
    diff = heat_after - heat_before
    diff_n = diff / (np.abs(diff).max() + 1e-8)
    weights = caps['eca_w'].cpu().float().numpy()

    # ---------- Plot ----------
    print(f'[4/4] Saving {args.out}')
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 1])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img_lb); ax0.set_title('Input (letterboxed)', fontsize=12)
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(img_lb); ax1.imshow(heat_before, cmap='jet', alpha=0.55)
    ax1.set_title(f'Before ECA\n(C3k2 output, layer {args.layer})', fontsize=12)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(img_lb); ax2.imshow(heat_after, cmap='jet', alpha=0.55)
    ax2.set_title(f'After ECA\n(C3k2Attn output)', fontsize=12)
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 3])
    im = ax3.imshow(diff_n, cmap='RdBu_r', vmin=-1, vmax=1)
    ax3.set_title('Attention effect\n(red = boosted, blue = suppressed)', fontsize=12)
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046)

    # Bottom row: channel weights bar
    axw = fig.add_subplot(gs[1, :])
    colors = ['#d62728' if w > 0.5 else '#1f77b4' for w in weights]
    axw.bar(np.arange(len(weights)), weights, color=colors, width=1.0)
    axw.axhline(0.5, color='k', linestyle='--', alpha=0.6, label='neutral (0.5)')
    axw.set_xlim(-1, len(weights))
    axw.set_xlabel('Channel index')
    axw.set_ylabel('ECA weight')
    boost = (weights > 0.5).sum()
    suppress = (weights < 0.5).sum()
    axw.set_title(
        f'ECA channel attention weights (C={len(weights)}) — '
        f'boosted: {boost}  |  suppressed: {suppress}  |  '
        f'mean: {weights.mean():.3f}  |  range: [{weights.min():.3f}, {weights.max():.3f}]',
        fontsize=11,
    )
    axw.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(args.out, dpi=180, bbox_inches='tight')
    print(f'\nSaved: {Path(args.out).resolve()}')

    # ---------- Text summary ----------
    top_boost = np.argsort(weights)[-5:][::-1]
    top_supp = np.argsort(weights)[:5]
    print('\nTop-5 boosted channels :', list(zip(top_boost.tolist(), weights[top_boost].round(3).tolist())))
    print('Top-5 suppressed       :', list(zip(top_supp.tolist(), weights[top_supp].round(3).tolist())))


if __name__ == '__main__':
    main()
