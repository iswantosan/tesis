"""
EDA ringkas untuk tesis — Tuberculosis6208.
Hanya generate yang dipakai di bab "Dataset Description".

Output:
  fig1_bbox_count.png      — Figure 4.1: distribusi bacilli per gambar
  fig2_bbox_size.png       — Figure 4.2: ukuran bbox + kategori COCO (small/medium/large)
  fig3_spatial.png         — Figure 4.3: spatial heatmap pusat bbox
  fig4_samples.png         — Figure 4.4: contoh gambar low/medium/high density
  table_summary.csv        — Table 4.1: ringkasan statistik dataset

Run: python tesis_eda.py
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Hilangkan border atas & kanan — hanya sumbu X dan Y yang terlihat
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# =========================================================
# CONFIG
# =========================================================
FOLDER = Path('D:/Project/yolov12/Tuberculosis6208/tuberculosis-phonecamera')
IMGSZ = 640   # YOLO training resolution → untuk klaim "small object"
OUT_DIR = Path('.')

# =========================================================
# 1. Parser XML (robust, fallback ke dimensi gambar)
# =========================================================
def parse_xml(xml_path):
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None, None, []

    img_w = img_h = None
    size_node = root.find('size')
    if size_node is not None:
        try:
            img_w = int(size_node.find('width').text)
            img_h = int(size_node.find('height').text)
        except Exception:
            pass
    if img_w is None:
        jpg = xml_path.with_suffix('.jpg')
        if jpg.exists():
            with Image.open(jpg) as im:
                img_w, img_h = im.size
    if img_w is None:
        return None, None, []

    bboxes = []
    for obj in root.findall('object'):
        bb = obj.find('bndbox') or obj.find('bbox') or obj.find('box')
        if bb is None:
            continue
        try:
            x1 = float(bb.find('xmin').text); y1 = float(bb.find('ymin').text)
            x2 = float(bb.find('xmax').text); y2 = float(bb.find('ymax').text)
        except Exception:
            continue
        if x2 > x1 and y2 > y1:
            bboxes.append((x1, y1, x2, y2))
    return img_w, img_h, bboxes


# =========================================================
# 2. Load semua data sekali
# =========================================================
xmls = sorted(FOLDER.glob('*.xml'))
assert xmls, f'No XML in {FOLDER}'

all_bboxes = []      # (xml_stem, iw, ih, x1, y1, x2, y2)
counts = []
img_dims = []
xml_data = {}        # xml -> (count, bboxes) untuk sampling

for xml in xmls:
    iw, ih, bbs = parse_xml(xml)
    if iw is None:
        continue
    counts.append(len(bbs))
    img_dims.append((iw, ih))
    xml_data[xml] = (len(bbs), bbs)
    for (x1, y1, x2, y2) in bbs:
        all_bboxes.append((xml.stem, iw, ih, x1, y1, x2, y2))

counts = np.array(counts)
print(f'Parsed {len(counts)} images | {len(all_bboxes)} bboxes')


# =========================================================
# 3. FIGURE 4.1 — Bbox count per image
# =========================================================
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.hist(counts, bins=range(0, counts.max() + 2), edgecolor='black', alpha=0.85,
        color='#1f77b4')
ax.axvline(counts.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean = {counts.mean():.2f}')
ax.axvline(np.median(counts), color='green', linestyle='--', linewidth=2,
           label=f'Median = {np.median(counts):.0f}')
ax.set_xlabel('Number of bacilli per image', fontsize=11)
ax.set_ylabel('Number of images', fontsize=11)
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig1_bbox_count.png', dpi=200, bbox_inches='tight')
plt.close()
print('Saved: fig1_bbox_count.png')


# =========================================================
# 4. FIGURE 4.2 — Bbox size at training resolution + COCO category
# =========================================================
ws_o = np.array([b[5] - b[3] for b in all_bboxes])
hs_o = np.array([b[6] - b[4] for b in all_bboxes])
areas_o = ws_o * hs_o

ws_s, hs_s = [], []
for (_, iw, ih, x1, y1, x2, y2) in all_bboxes:
    s = IMGSZ / max(iw, ih)
    ws_s.append((x2 - x1) * s)
    hs_s.append((y2 - y1) * s)
ws_s = np.array(ws_s); hs_s = np.array(hs_s)
areas_s = ws_s * hs_s

# COCO categories pada training resolution (relevant untuk klaim)
small_s = (areas_s < 32**2).sum()
medium_s = ((areas_s >= 32**2) & (areas_s < 96**2)).sum()
large_s = (areas_s >= 96**2).sum()
total = len(areas_s)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: histogram √area (training res) dengan threshold COCO
axes[0].hist(np.sqrt(areas_s), bins=50, edgecolor='black', color='#ff7f0e', alpha=0.85)
axes[0].axvline(32, color='red', linestyle='--', linewidth=2,
                label=r'COCO small threshold ($\sqrt{1024}=32$ px)')
axes[0].axvline(96, color='purple', linestyle='--', linewidth=2,
                label=r'COCO medium/large ($\sqrt{9216}=96$ px)')
axes[0].set_xlabel(r'$\sqrt{Area}$ (px) at training resolution', fontsize=11)
axes[0].set_ylabel('Number of bboxes', fontsize=11)
axes[0].legend(fontsize=10); axes[0].grid(alpha=0.3)

# Right: pie chart COCO categories
labels = [f'Small\n(<1024 px²)\n{100*small_s/total:.1f}%',
          f'Medium\n(1024-9216 px²)\n{100*medium_s/total:.1f}%',
          f'Large\n(>9216 px²)\n{100*large_s/total:.1f}%']
colors = ['#d62728', '#ff7f0e', '#2ca02c']
axes[1].pie([small_s, medium_s, large_s], labels=labels, colors=colors,
            startangle=90, textprops={'fontsize': 11},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig2_bbox_size.png', dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved: fig2_bbox_size.png  (Small={100*small_s/total:.1f}%)')


# =========================================================
# 5. FIGURE 4.3 — Spatial heatmap (justify augmentation)
# =========================================================
norm_cx = np.array([((b[3] + b[5]) / 2) / b[1] for b in all_bboxes])
norm_cy = np.array([((b[4] + b[6]) / 2) / b[2] for b in all_bboxes])

fig, ax = plt.subplots(figsize=(7, 7))
h2 = ax.hist2d(norm_cx, norm_cy, bins=40, cmap='hot')
ax.set_xlabel('Normalized X (bbox center)', fontsize=11)
ax.set_ylabel('Normalized Y (bbox center)', fontsize=11)
ax.invert_yaxis(); ax.set_aspect('equal')
plt.colorbar(h2[3], ax=ax, label='Count')
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig3_spatial.png', dpi=200, bbox_inches='tight')
plt.close()
print('Saved: fig3_spatial.png')


# =========================================================
# 6. FIGURE 4.4 — Sample images (low/medium/high density)
# =========================================================
def pick_samples(xml_data, lo, hi, n=3):
    cand = [x for x, (c, _) in xml_data.items() if lo <= c <= hi]
    if not cand:
        return []
    # spread across the range
    cand_sorted = sorted(cand, key=lambda x: xml_data[x][0])
    if len(cand_sorted) >= n:
        idx = np.linspace(0, len(cand_sorted) - 1, n, dtype=int)
        return [cand_sorted[i] for i in idx]
    return cand_sorted

samples = {
    'Low density (1-3 bacilli)':   pick_samples(xml_data, 1, 3, 3),
    'Medium density (5-8)':         pick_samples(xml_data, 5, 8, 3),
    'High density (>10)':           pick_samples(xml_data, 11, 1000, 3),
}

fig, axes = plt.subplots(3, 3, figsize=(14, 12))
for row, (cat, sl) in enumerate(samples.items()):
    for col in range(3):
        ax = axes[row, col]
        if col >= len(sl):
            ax.axis('off'); continue
        xml = sl[col]
        _, bbs = xml_data[xml]
        img = cv2.imread(str(xml.with_suffix('.jpg')))
        if img is None:
            ax.axis('off'); continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for x1, y1, x2, y2 in bbs:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                          (255, 0, 0), 3)
        ax.imshow(img)
        ax.set_xlabel(f'{cat} — {len(bbs)} bacilli', fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig4_samples.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: fig4_samples.png')


# =========================================================
# 7. TABLE 4.1 — Dataset summary
# =========================================================
ratios = ws_o / hs_o
elong = ((ratios > 2) | (ratios < 0.5)).sum()
img_w_arr = np.array([d[0] for d in img_dims])
img_h_arr = np.array([d[1] for d in img_dims])

summary = [
    ('Total images',                              len(counts)),
    ('Total bacilli annotations',                 int(counts.sum())),
    ('Images with bacilli',                       int((counts > 0).sum())),
    ('Background images (0 bacilli)',             int((counts == 0).sum())),
    ('Mean bacilli per image',                    f'{counts.mean():.2f}'),
    ('Median bacilli per image',                  f'{np.median(counts):.0f}'),
    ('Max bacilli per image',                     int(counts.max())),
    ('Image resolution (most common)',            f'{int(np.median(img_w_arr))}x{int(np.median(img_h_arr))}'),
    ('Bbox mean width (px, original)',            f'{ws_o.mean():.1f}'),
    ('Bbox mean height (px, original)',           f'{hs_o.mean():.1f}'),
    ('Bbox mean area (px², original)',            f'{areas_o.mean():.0f}'),
    (f'Bbox mean width (px, imgsz={IMGSZ})',      f'{ws_s.mean():.1f}'),
    (f'Bbox mean height (px, imgsz={IMGSZ})',     f'{hs_s.mean():.1f}'),
    (f'Bbox mean area (px², imgsz={IMGSZ})',      f'{areas_s.mean():.1f}'),
    (f'Bbox median area (px², imgsz={IMGSZ})',    f'{np.median(areas_s):.1f}'),
    (f'Small bboxes @ imgsz={IMGSZ} (<1024 px²)', f'{100*small_s/total:.1f}%'),
    (f'Medium bboxes @ imgsz={IMGSZ}',            f'{100*medium_s/total:.1f}%'),
    (f'Large bboxes @ imgsz={IMGSZ}',             f'{100*large_s/total:.1f}%'),
    ('Elongated bboxes (w/h>2 or <0.5)',          f'{100*elong/len(ratios):.1f}%'),
]
df = pd.DataFrame(summary, columns=['Metric', 'Value'])
df.to_csv(OUT_DIR / 'table_summary.csv', index=False)
print('\n=== TABLE 4.1 — Dataset summary ===')
print(df.to_string(index=False))
print('\nSaved: table_summary.csv')
