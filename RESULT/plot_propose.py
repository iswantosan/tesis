"""
Generate per-epoch training curves untuk MODEL USULAN (folder Propose).

Output (di folder RESULT/figures/):
  --- Versi mean ± std (band style) ---
  fig_propose_box_loss.png      — train + val box loss (mean ± std dari 3 seed)
  fig_propose_cls_loss.png      — train + val cls loss (mean ± std dari 3 seed)
  fig_propose_dfl_loss.png      — train + val dfl loss (mean ± std dari 3 seed)
  fig_propose_map50.png         — mAP@0.5 (mean ± std dari 3 seed)
  fig_propose_map5095.png       — mAP@0.5:0.95 (mean ± std dari 3 seed)
  fig_propose_map_combined.png  — mAP@0.5 + mAP@0.5:0.95 dalam 1 figure (2 subplot)
  fig_propose_loss_combined.png — 3 loss dalam 1 figure (3 subplot)

  --- Versi 3-line-per-seed (untuk tesis, gaya sama dgn fig4_8*) ---
  fig4_15_map50_propose.png     — mAP@0.5 Model Usulan per seed (3 line)
  fig4_16_map5095_propose.png   — mAP@0.5:0.95 Model Usulan per seed (3 line)
  fig4_17_precision_propose.png — Precision Model Usulan per seed (3 line)
  fig4_18_recall_propose.png    — Recall Model Usulan per seed (3 line)

  --- Loss Model Usulan (gaya sama dgn fig4_5/4_6/4_7 baseline: mean ± std) ---
  fig4_19_box_loss_propose.png  — train + val box loss (mean ± std)
  fig4_20_cls_loss_propose.png  — train + val cls loss (mean ± std)
  fig4_21_dfl_loss_propose.png  — train + val dfl loss (mean ± std)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

RESULT_DIR = Path('D:/Project/yolov12/RESULT')
FIG_DIR = RESULT_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)

SEED_MAP = {1: 42, 2: 1050, 3: 2025}

seed_files = {sid: pd.read_csv(RESULT_DIR / f'Propose/results_seed_{i}.csv')
              for i, sid in SEED_MAP.items()}
for df in seed_files.values():
    df.columns = df.columns.str.strip()

print(f'Loaded {len(seed_files)} Propose CSVs')


def _stack(col):
    return np.stack([df[col].values for df in seed_files.values()])


epochs = seed_files[42]['epoch'].values


# =============================================================
# Helper: loss curve (train + val, mean ± std across 3 seeds)
# =============================================================
def plot_loss(loss_name, fig_path, ylabel):
    train_stack = _stack(f'train/{loss_name}')
    val_stack   = _stack(f'val/{loss_name}')

    train_mean, train_std = train_stack.mean(0), train_stack.std(0, ddof=1)
    val_mean,   val_std   = val_stack.mean(0),   val_stack.std(0, ddof=1)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(epochs, train_mean, color='#1f77b4', linewidth=2, label='Train (mean)')
    ax.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                    color='#1f77b4', alpha=0.15)
    ax.plot(epochs, val_mean, color='#d62728', linewidth=2, label='Validation (mean)')
    ax.fill_between(epochs, val_mean - val_std, val_mean + val_std,
                    color='#d62728', alpha=0.15)

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fig_path.name}')


# =============================================================
# Helper: mAP curve (mean ± std across 3 seeds)
# =============================================================
def plot_map(metric_col, fig_path, ylabel, color='#2ca02c'):
    stack = _stack(metric_col)
    mean, std = stack.mean(0), stack.std(0, ddof=1)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(epochs, mean, color=color, linewidth=2, label=f'{ylabel} (mean)')
    ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.18,
                    label='± 1 std (3 seed)')

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fig_path.name}')


# =============================================================
# Individual figures
# =============================================================
plot_loss('box_loss', FIG_DIR / 'fig_propose_box_loss.png', 'Box loss')
plot_loss('cls_loss', FIG_DIR / 'fig_propose_cls_loss.png', 'Classification loss')
plot_loss('dfl_loss', FIG_DIR / 'fig_propose_dfl_loss.png', 'Distribution focal loss (DFL)')

plot_map('metrics/mAP50(B)',    FIG_DIR / 'fig_propose_map50.png',
         'mAP@0.5', color='#2ca02c')
plot_map('metrics/mAP50-95(B)', FIG_DIR / 'fig_propose_map5095.png',
         'mAP@0.5:0.95', color='#9467bd')


# =============================================================
# Combined: 2 mAP curves dalam 1 figure
# =============================================================
def plot_map_combined(fig_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    for ax, metric_col, ylabel, color in zip(
        axes,
        ['metrics/mAP50(B)', 'metrics/mAP50-95(B)'],
        ['mAP@0.5', 'mAP@0.5:0.95'],
        ['#2ca02c', '#9467bd'],
    ):
        stack = _stack(metric_col)
        mean, std = stack.mean(0), stack.std(0, ddof=1)
        ax.plot(epochs, mean, color=color, linewidth=2, label=f'{ylabel} (mean)')
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.18,
                        label='± 1 std (3 seed)')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fig_path.name}')


plot_map_combined(FIG_DIR / 'fig_propose_map_combined.png')


# =============================================================
# Combined: 3 loss curves dalam 1 figure
# =============================================================
def plot_loss_combined(fig_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    for ax, loss_name, ylabel in zip(
        axes,
        ['box_loss', 'cls_loss', 'dfl_loss'],
        ['Box loss', 'Classification loss', 'DFL'],
    ):
        train_stack = _stack(f'train/{loss_name}')
        val_stack   = _stack(f'val/{loss_name}')
        tm, ts = train_stack.mean(0), train_stack.std(0, ddof=1)
        vm, vs = val_stack.mean(0),   val_stack.std(0, ddof=1)

        ax.plot(epochs, tm, color='#1f77b4', linewidth=2, label='Train (mean)')
        ax.fill_between(epochs, tm - ts, tm + ts, color='#1f77b4', alpha=0.15)
        ax.plot(epochs, vm, color='#d62728', linewidth=2, label='Validation (mean)')
        ax.fill_between(epochs, vm - vs, vm + vs, color='#d62728', alpha=0.15)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fig_path.name}')


plot_loss_combined(FIG_DIR / 'fig_propose_loss_combined.png')


# =============================================================
# Versi 3-line-per-seed (untuk tesis, mirror gaya fig4_8* baseline)
# =============================================================
def plot_per_seed(metric_col, fig_path, ylabel):
    """Plot 3 line, satu per seed — gaya identik dgn fig4_8_map50 baseline."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ['#1f77b4', '#2ca02c', '#d62728']

    for (seed, df), color in zip(seed_files.items(), colors):
        ax.plot(df['epoch'], df[metric_col], color=color,
                linewidth=1.8, label=f'Seed {seed}')

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fig_path.name}')


plot_per_seed('metrics/mAP50(B)',    FIG_DIR / 'fig4_15_map50_propose.png',
              'mAP@0.5')
plot_per_seed('metrics/mAP50-95(B)', FIG_DIR / 'fig4_16_map5095_propose.png',
              'mAP@0.5:0.95')
plot_per_seed('metrics/precision(B)', FIG_DIR / 'fig4_17_precision_propose.png',
              'Precision')
plot_per_seed('metrics/recall(B)',    FIG_DIR / 'fig4_18_recall_propose.png',
              'Recall')


# =============================================================
# Loss Model Usulan (gaya sama dgn fig4_5/4_6/4_7 baseline)
# =============================================================
plot_loss('box_loss', FIG_DIR / 'fig4_19_box_loss_propose.png', 'Box loss')
plot_loss('cls_loss', FIG_DIR / 'fig4_20_cls_loss_propose.png', 'Classification loss')
plot_loss('dfl_loss', FIG_DIR / 'fig4_21_dfl_loss_propose.png', 'Distribution focal loss (DFL)')


# =============================================================
# Ringkasan best-epoch values (untuk caption / narasi)
# =============================================================
print('\n=== Best-epoch metrics per seed (Model Usulan) ===')
rows = []
for seed, df in seed_files.items():
    best_idx = df['metrics/mAP50(B)'].idxmax()
    rows.append({
        'Seed': seed,
        'Best Ep': int(df.loc[best_idx, 'epoch']),
        'mAP@0.5': df.loc[best_idx, 'metrics/mAP50(B)'],
        'mAP@0.5:0.95': df.loc[best_idx, 'metrics/mAP50-95(B)'],
        'box_loss (val)': df.loc[best_idx, 'val/box_loss'],
        'cls_loss (val)': df.loc[best_idx, 'val/cls_loss'],
        'dfl_loss (val)': df.loc[best_idx, 'val/dfl_loss'],
    })
ringkasan = pd.DataFrame(rows)
for c in ['mAP@0.5', 'mAP@0.5:0.95', 'box_loss (val)', 'cls_loss (val)', 'dfl_loss (val)']:
    ringkasan[c] = ringkasan[c].map(lambda x: f'{x:.4f}')
print(ringkasan.to_string(index=False))

print(f'\nAll outputs saved to: {FIG_DIR}')
