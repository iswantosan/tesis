"""
EDA all-in-one — Tuberculosis6208 dataset.
Run: python eda_run_all.py

Generates:
  eda_bbox_count.png
  eda_bbox_size.png
  eda_bbox_size_summary.csv
  eda_density_distribution.csv
  eda_aspect_ratio.png
  eda_spatial_heatmap.png
  eda_density.png
  eda_sample_images.png
  eda_color_dist.png
  eda_annotation_issues.csv
  eda_dataset_summary.csv
"""

import xml.etree.ElementTree as ET
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image


# =========================================================
# 0. SETUP
# =========================================================
folder = Path('D:/Project/yolov12/Tuberculosis6208/tuberculosis-phonecamera')
xmls = sorted(folder.glob('*.xml'))
jpgs = sorted(folder.glob('*.jpg'))

assert len(jpgs) > 0, f'No images in {folder}'
print(f'Images: {len(jpgs)} | XMLs: {len(xmls)}')


# =========================================================
# 1. ROBUST XML PARSER (helper)
# =========================================================
def parse_xml_robust(xml_path):
    """Return (img_w, img_h, list_of_bboxes). Handles non-standard schemas."""
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None, None, []

    img_w, img_h = None, None
    size_node = root.find('size')
    if size_node is not None:
        w_node, h_node = size_node.find('width'), size_node.find('height')
        if w_node is not None and h_node is not None:
            try:
                img_w, img_h = int(w_node.text), int(h_node.text)
            except (ValueError, TypeError):
                pass
    if img_w is None:
        jpg_path = xml_path.with_suffix('.jpg')
        if jpg_path.exists():
            try:
                with Image.open(jpg_path) as im:
                    img_w, img_h = im.size
            except Exception:
                pass
    if img_w is None:
        return None, None, []

    bboxes = []
    for obj in root.findall('object'):
        bb = None
        for tag in ('bndbox', 'bbox', 'box'):
            bb = obj.find(tag)
            if bb is not None:
                break
        if bb is None:
            continue
        try:
            x1 = float(bb.find('xmin').text); y1 = float(bb.find('ymin').text)
            x2 = float(bb.find('xmax').text); y2 = float(bb.find('ymax').text)
        except (AttributeError, ValueError, TypeError):
            continue
        if (x2 - x1) > 0 and (y2 - y1) > 0:
            bboxes.append((x1, y1, x2, y2))
    return img_w, img_h, bboxes


# =========================================================
# 2. OVERVIEW & BBOX COUNT PER IMAGE
# =========================================================
print('\n' + '=' * 70)
print('SECTION 2: BBOX COUNT PER IMAGE')
print('=' * 70)

all_bboxes = []
counts = []
img_dims = []
for xml in xmls:
    iw, ih, bbs = parse_xml_robust(xml)
    if iw is None:
        continue
    counts.append(len(bbs))
    img_dims.append((iw, ih))
    for x1, y1, x2, y2 in bbs:
        all_bboxes.append((xml.stem, iw, ih, x1, y1, x2, y2))

counts = np.array(counts)
print(f'Total images parsed   : {len(counts)}')
print(f'Total bacilli         : {counts.sum()}')
print(f'Mean bacilli/image    : {counts.mean():.2f}')
print(f'Median                : {np.median(counts):.0f}')
print(f'Min/Max               : {counts.min()}/{counts.max()}')
print(f'\n=== Distribution per category ===')
print(f'Background (0 bacilli) : {(counts==0).sum():>4} ({100*(counts==0).mean():>5.1f}%)')
print(f'1 bacillus             : {(counts==1).sum():>4} ({100*(counts==1).mean():>5.1f}%)')
print(f'2-5 bacilli            : {((counts>=2)&(counts<=5)).sum():>4} ({100*((counts>=2)&(counts<=5)).mean():>5.1f}%)')
print(f'6-10 bacilli           : {((counts>=6)&(counts<=10)).sum():>4} ({100*((counts>=6)&(counts<=10)).mean():>5.1f}%)')
print(f'>10 bacilli            : {(counts>10).sum():>4} ({100*(counts>10).mean():>5.1f}%)')

pd.DataFrame([
    ('Background (0)',  int((counts==0).sum()),                  f'{100*(counts==0).mean():.1f}%'),
    ('1 bacillus',      int((counts==1).sum()),                  f'{100*(counts==1).mean():.1f}%'),
    ('2-5 bacilli',     int(((counts>=2)&(counts<=5)).sum()),    f'{100*((counts>=2)&(counts<=5)).mean():.1f}%'),
    ('6-10 bacilli',    int(((counts>=6)&(counts<=10)).sum()),   f'{100*((counts>=6)&(counts<=10)).mean():.1f}%'),
    ('>10 bacilli',     int((counts>10).sum()),                  f'{100*(counts>10).mean():.1f}%'),
], columns=['Category', 'Count', 'Percentage']).to_csv('eda_density_distribution.csv', index=False)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(counts, bins=range(0, counts.max()+2), edgecolor='black', alpha=0.8)
ax.axvline(counts.mean(), color='red', linestyle='--', label=f'Mean {counts.mean():.2f}')
ax.axvline(np.median(counts), color='green', linestyle='--', label=f'Median {np.median(counts):.0f}')
ax.set_xlabel('Bacilli per image'); ax.set_ylabel('Number of images')
ax.set_title('Distribution of bacilli count per image')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('eda_bbox_count.png', dpi=150, bbox_inches='tight'); plt.close()
print('Saved: eda_bbox_count.png')


# =========================================================
# 3. BBOX SIZE + COCO CATEGORIES (original + training res)
# =========================================================
print('\n' + '=' * 70)
print('SECTION 3: BBOX SIZE DISTRIBUTION')
print('=' * 70)

IMGSZ = 640
ws_orig = np.array([b[5]-b[3] for b in all_bboxes])
hs_orig = np.array([b[6]-b[4] for b in all_bboxes])
areas_orig = ws_orig * hs_orig

ws_scaled, hs_scaled = [], []
for (name, iw, ih, x1, y1, x2, y2) in all_bboxes:
    scale = IMGSZ / max(iw, ih)
    ws_scaled.append((x2 - x1) * scale)
    hs_scaled.append((y2 - y1) * scale)
ws_scaled = np.array(ws_scaled)
hs_scaled = np.array(hs_scaled)
areas_scaled = ws_scaled * hs_scaled

print(f'Total bboxes: {len(ws_orig)}')

print(f'\n=== ORIGINAL resolution ===')
print(f'Width  : mean={ws_orig.mean():.1f},  median={np.median(ws_orig):.1f},  range=[{ws_orig.min():.0f},{ws_orig.max():.0f}]')
print(f'Height : mean={hs_orig.mean():.1f},  median={np.median(hs_orig):.1f},  range=[{hs_orig.min():.0f},{hs_orig.max():.0f}]')
print(f'Area   : mean={areas_orig.mean():.0f} px²,  median={np.median(areas_orig):.0f}')

small_o  = (areas_orig < 32**2).sum()
medium_o = ((areas_orig >= 32**2) & (areas_orig < 96**2)).sum()
large_o  = (areas_orig >= 96**2).sum()
print(f'COCO original: Small {small_o} ({100*small_o/len(areas_orig):.1f}%) | Medium {medium_o} ({100*medium_o/len(areas_orig):.1f}%) | Large {large_o} ({100*large_o/len(areas_orig):.1f}%)')

print(f'\n=== TRAINING resolution (imgsz={IMGSZ}) ===')
print(f'Width  : mean={ws_scaled.mean():.1f},  median={np.median(ws_scaled):.1f}')
print(f'Height : mean={hs_scaled.mean():.1f},  median={np.median(hs_scaled):.1f}')
print(f'Area   : mean={areas_scaled.mean():.1f} px²,  median={np.median(areas_scaled):.1f}')

small_s  = (areas_scaled < 32**2).sum()
medium_s = ((areas_scaled >= 32**2) & (areas_scaled < 96**2)).sum()
large_s  = (areas_scaled >= 96**2).sum()
print(f'COCO scaled: Small {small_s} ({100*small_s/len(areas_scaled):.1f}%) | Medium {medium_s} ({100*medium_s/len(areas_scaled):.1f}%) | Large {large_s} ({100*large_s/len(areas_scaled):.1f}%)')

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes[0,0].hist(ws_orig, bins=50, edgecolor='black', color='#1f77b4'); axes[0,0].set_title('Width @ original')
axes[0,1].hist(hs_orig, bins=50, edgecolor='black', color='#1f77b4'); axes[0,1].set_title('Height @ original')
axes[0,2].hist(np.sqrt(areas_orig), bins=50, edgecolor='black', color='#1f77b4'); axes[0,2].set_title('√Area @ original')
axes[1,0].hist(ws_scaled, bins=50, edgecolor='black', color='#ff7f0e'); axes[1,0].set_title(f'Width @ imgsz={IMGSZ}')
axes[1,1].hist(hs_scaled, bins=50, edgecolor='black', color='#ff7f0e'); axes[1,1].set_title(f'Height @ imgsz={IMGSZ}')
axes[1,2].hist(np.sqrt(areas_scaled), bins=50, edgecolor='black', color='#ff7f0e'); axes[1,2].set_title(f'√Area @ imgsz={IMGSZ}')
for ax in axes.flatten():
    ax.grid(alpha=0.3); ax.set_xlabel('pixels'); ax.set_ylabel('count')
for ax in [axes[0,2], axes[1,2]]:
    ax.axvline(32, color='red', linestyle='--', alpha=0.6, label='COCO small (√1024=32)')
    ax.legend()
plt.suptitle(f'Bbox size: ORIGINAL (top) vs TRAINING imgsz={IMGSZ} (bottom)', fontsize=14)
plt.tight_layout(); plt.savefig('eda_bbox_size.png', dpi=150, bbox_inches='tight'); plt.close()

pd.DataFrame({
    'Resolution': ['Original', f'imgsz={IMGSZ}'],
    'Mean width': [f'{ws_orig.mean():.1f}', f'{ws_scaled.mean():.1f}'],
    'Mean height': [f'{hs_orig.mean():.1f}', f'{hs_scaled.mean():.1f}'],
    'Mean area': [f'{areas_orig.mean():.0f}', f'{areas_scaled.mean():.1f}'],
    'Small %': [f'{100*small_o/len(areas_orig):.1f}%', f'{100*small_s/len(areas_scaled):.1f}%'],
    'Medium %': [f'{100*medium_o/len(areas_orig):.1f}%', f'{100*medium_s/len(areas_scaled):.1f}%'],
    'Large %': [f'{100*large_o/len(areas_orig):.1f}%', f'{100*large_s/len(areas_scaled):.1f}%'],
}).to_csv('eda_bbox_size_summary.csv', index=False)

ws, hs, areas = ws_orig, hs_orig, areas_orig   # backward-compat for sections below
small, medium, large = small_o, medium_o, large_o
print('Saved: eda_bbox_size.png, eda_bbox_size_summary.csv')


# =========================================================
# 4. ASPECT RATIO
# =========================================================
print('\n' + '=' * 70)
print('SECTION 4: ASPECT RATIO')
print('=' * 70)

ratios = ws / hs
elong_h = (ratios > 2).sum()
elong_v = (ratios < 0.5).sum()
square = ((ratios >= 0.5) & (ratios <= 2)).sum()
print(f'Mean: {ratios.mean():.2f}, Median: {np.median(ratios):.2f}')
print(f'Elongated horizontal (w/h > 2)  : {elong_h} ({100*elong_h/len(ratios):.1f}%)')
print(f'Elongated vertical   (w/h < 0.5): {elong_v} ({100*elong_v/len(ratios):.1f}%)')
print(f'Square-ish (0.5-2)               : {square} ({100*square/len(ratios):.1f}%)')

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(np.log10(ratios), bins=50, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', label='Square (1:1)')
ax.set_xlabel('log₁₀(w/h)'); ax.set_title('Aspect ratio distribution')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('eda_aspect_ratio.png', dpi=150, bbox_inches='tight'); plt.close()
print('Saved: eda_aspect_ratio.png')


# =========================================================
# 5. IMAGE RESOLUTION
# =========================================================
print('\n' + '=' * 70)
print('SECTION 5: IMAGE RESOLUTION')
print('=' * 70)
img_w_arr = np.array([d[0] for d in img_dims])
img_h_arr = np.array([d[1] for d in img_dims])
res_counter = pd.Series([(w, h) for w, h in zip(img_w_arr, img_h_arr)]).value_counts()
print(f'Unique dims: {len(res_counter)}')
print(f'Most common:\n{res_counter.head(5)}')


# =========================================================
# 6. SPATIAL HEATMAP
# =========================================================
print('\n' + '=' * 70)
print('SECTION 6: SPATIAL HEATMAP')
print('=' * 70)

norm_cx = np.array([((b[3]+b[5])/2)/b[1] for b in all_bboxes])
norm_cy = np.array([((b[4]+b[6])/2)/b[2] for b in all_bboxes])

fig, ax = plt.subplots(figsize=(8, 8))
h2 = ax.hist2d(norm_cx, norm_cy, bins=50, cmap='hot')
ax.set_xlabel('Normalized X'); ax.set_ylabel('Normalized Y')
ax.set_title('Spatial heatmap of bbox centers')
ax.invert_yaxis(); ax.set_aspect('equal')
plt.colorbar(h2[3], ax=ax, label='Count')
plt.tight_layout(); plt.savefig('eda_spatial_heatmap.png', dpi=150, bbox_inches='tight'); plt.close()
print('Saved: eda_spatial_heatmap.png')


# =========================================================
# 7. DENSITY / CLUSTERING
# =========================================================
print('\n' + '=' * 70)
print('SECTION 7: DENSITY / CLUSTERING')
print('=' * 70)

def min_dist_xml(xml):
    iw, ih, bbs = parse_xml_robust(xml)
    if len(bbs) < 2:
        return None
    centers = [((b[0]+b[2])/2/iw, (b[1]+b[3])/2/ih) for b in bbs]
    dists = []
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            dists.append(np.hypot(centers[i][0]-centers[j][0], centers[i][1]-centers[j][1]))
    return min(dists)

min_dists = [d for d in (min_dist_xml(x) for x in xmls) if d is not None]
min_dists = np.array(min_dists)
print(f'Images with >=2 bacilli: {len(min_dists)}')
print(f'Mean min-dist (normalized): {min_dists.mean():.3f}')
print(f'% with cluster (<0.05): {100*(min_dists<0.05).mean():.1f}%')

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(min_dists, bins=50, edgecolor='black')
ax.axvline(0.05, color='red', linestyle='--', label='Cluster threshold (0.05)')
ax.set_xlabel('Min normalized distance between bacilli')
ax.set_title('Bacilli clustering analysis')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('eda_density.png', dpi=150, bbox_inches='tight'); plt.close()
print('Saved: eda_density.png')


# =========================================================
# 8. SAMPLE IMAGES (stratified, self-contained)
# =========================================================
print('\n' + '=' * 70)
print('SECTION 8: SAMPLE IMAGES (stratified)')
print('=' * 70)

xml_data = {}
for xml in xmls:
    iw, ih, bbs = parse_xml_robust(xml)
    if iw is None:
        continue
    xml_data[xml] = (len(bbs), bbs)

by_count = {}
for xml, (cnt, bbs) in xml_data.items():
    by_count.setdefault(cnt, []).append(xml)

def get_stratified(by_count, count_range, n=3):
    available = sorted([c for c in count_range if c in by_count])
    if not available:
        return []
    if len(available) >= n:
        indices = sorted(set([0, len(available)//2, len(available)-1]))[:n]
        picked_counts = [available[i] for i in indices]
    else:
        picked_counts = available
    return [by_count[c][0] for c in picked_counts]

samples = {
    'Low (1-3 bacilli)':  get_stratified(by_count, [1, 2, 3], 3),
    'Medium (5-8)':       get_stratified(by_count, [5, 6, 7, 8], 3),
    'High (>10)':         get_stratified(by_count, [c for c in by_count if c > 10], 3),
}

for cat, sl in samples.items():
    print(f'{cat:<22}: {len(sl)} samples')

def draw_gt(img_path, bboxes):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for x1, y1, x2, y2 in bboxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
    return img

n_rows, n_cols = 3, 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 14))
for row, (cat, sample_list) in enumerate(samples.items()):
    for col in range(n_cols):
        ax = axes[row, col]
        if col >= len(sample_list):
            ax.axis('off'); continue
        xml = sample_list[col]
        bbs = xml_data[xml][1]
        img = draw_gt(xml.with_suffix('.jpg'), bbs)
        if img is None:
            ax.text(0.5, 0.5, f'Load failed:\n{xml.stem}', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off'); continue
        ax.imshow(img)
        ax.set_title(f'{cat}\n{xml.stem} ({len(bbs)} bacilli)', fontsize=11)
        ax.axis('off')
plt.tight_layout(); plt.savefig('eda_sample_images.png', dpi=120, bbox_inches='tight'); plt.close()
print('Saved: eda_sample_images.png')


# =========================================================
# 9. COLOR DISTRIBUTION (HSV)
# =========================================================
print('\n' + '=' * 70)
print('SECTION 9: COLOR DISTRIBUTION')
print('=' * 70)

sampled = random.sample(jpgs, min(50, len(jpgs)))
all_h, all_s, all_v = [], [], []
for p in sampled:
    img = cv2.imread(str(p))
    if img is None:
        continue
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    all_h.extend(hsv[:,:,0].flatten()[::100])
    all_s.extend(hsv[:,:,1].flatten()[::100])
    all_v.extend(hsv[:,:,2].flatten()[::100])

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
axes[0].hist(all_h, bins=180, color='purple'); axes[0].set_title('Hue (0-179)')
axes[1].hist(all_s, bins=50, color='green');   axes[1].set_title('Saturation (0-255)')
axes[2].hist(all_v, bins=50, color='gray');    axes[2].set_title('Value (0-255)')
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('eda_color_dist.png', dpi=150, bbox_inches='tight'); plt.close()
print('Saved: eda_color_dist.png')


# =========================================================
# 10. ANNOTATION QUALITY CHECK
# =========================================================
print('\n' + '=' * 70)
print('SECTION 10: ANNOTATION QUALITY CHECK')
print('=' * 70)

issues = []
for xml in xmls:
    iw, ih, bbs = parse_xml_robust(xml)
    if iw is None:
        continue
    for x1, y1, x2, y2 in bbs:
        w, h = x2-x1, y2-y1
        if w < 3 or h < 3:
            issues.append((xml.stem, 'too_small', f'{w:.0f}x{h:.0f}'))
        if w > iw*0.5 or h > ih*0.5:
            issues.append((xml.stem, 'too_large', f'{w:.0f}x{h:.0f} / {iw}x{ih}'))
        if x1 < 0 or y1 < 0 or x2 > iw or y2 > ih:
            issues.append((xml.stem, 'out_of_bounds', f'[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]'))
        if w/h > 10 or h/w > 10:
            issues.append((xml.stem, 'extreme_ratio', f'{w:.0f}x{h:.0f}'))

print(f'Total potential issues: {len(issues)}')
if issues:
    df_issues = pd.DataFrame(issues, columns=['image', 'issue_type', 'detail'])
    print(df_issues['issue_type'].value_counts())
    df_issues.to_csv('eda_annotation_issues.csv', index=False)
    print('Saved: eda_annotation_issues.csv')


# =========================================================
# 11. SUMMARY TABLE
# =========================================================
print('\n' + '=' * 70)
print('SECTION 11: DATASET SUMMARY TABLE')
print('=' * 70)

summary = {
    'Total images':                          len(counts),
    'Total bacilli annotations':             int(counts.sum()),
    'Images with bacilli':                   int((counts > 0).sum()),
    'Background images (no bacilli)':        int((counts == 0).sum()),
    'Mean bacilli per image':                f'{counts.mean():.2f}',
    'Median bacilli per image':              f'{np.median(counts):.0f}',
    'Max bacilli per image':                 int(counts.max()),
    'Image dim (most common)':               f'{img_w_arr[0]}x{img_h_arr[0]}',
    'Bbox mean width (px, original)':        f'{ws_orig.mean():.1f}',
    'Bbox mean height (px, original)':       f'{hs_orig.mean():.1f}',
    'Bbox mean area (px², original)':        f'{areas_orig.mean():.0f}',
    'Small bboxes @ original (<1024px²)':    f'{100*small_o/len(areas_orig):.1f}%',
    'Bbox mean width (px, imgsz=640)':       f'{ws_scaled.mean():.1f}',
    'Bbox mean height (px, imgsz=640)':      f'{hs_scaled.mean():.1f}',
    'Bbox mean area (px², imgsz=640)':       f'{areas_scaled.mean():.1f}',
    'Small bboxes @ imgsz=640 (<1024px²)':   f'{100*small_s/len(areas_scaled):.1f}%',
    'Elongated bboxes (w/h>2 or <0.5)':      f'{100*(elong_h+elong_v)/len(ratios):.1f}%',
    'Mean min-distance (clusters, norm)':    f'{min_dists.mean():.3f}',
    'Annotation issues found':               len(issues),
}
df_sum = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
print(df_sum.to_string(index=False))
df_sum.to_csv('eda_dataset_summary.csv', index=False)

print('\n' + '=' * 70)
print('DONE. Files generated:')
print('  eda_bbox_count.png')
print('  eda_bbox_size.png + eda_bbox_size_summary.csv')
print('  eda_density_distribution.csv')
print('  eda_aspect_ratio.png')
print('  eda_spatial_heatmap.png')
print('  eda_density.png')
print('  eda_sample_images.png')
print('  eda_color_dist.png')
print('  eda_annotation_issues.csv')
print('  eda_dataset_summary.csv')
print('=' * 70)
