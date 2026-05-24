"""
Generate Gambar 4.15 dan 4.16 untuk Section 4.4 — Visualisasi Hasil Deteksi.

Output:
  fig4_15_detection_success.png  — 2 panel: low density + high density
  fig4_16_detection_errors.png   — 1 panel: error case (FP/FN/box drift)

REQUIREMENT:
  - WEIGHTS_PATH: path ke best.pt Model Usulan (atau weights apa pun yang lo
    mau visualisasikan). Kalau belum ada, gw rekomen pakai baseline best.pt
    sementara sambil training Model Usulan asli.
  - Ultralytics package terinstall: pip install ultralytics

Color legend (consistent dgn FPS test screenshot lo):
  Kuning  = Ground Truth (GT)
  Hijau   = True Positive (TP)
  Merah   = False Positive (FP)
  Magenta = False Negative (FN missed)
"""
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =============================================================
# CONFIG — edit sesuai setup lo
# =============================================================
WEIGHTS_PATH = 'D:/Project/yolov12/RESULT/content/tb_detections_chen_yolov12s_project/seed42_freeze2/weights/best.pt'
SOURCE_DIR   = Path('D:/Project/yolov12/Tuberculosis6208/tuberculosis-phonecamera')
OUT_DIR      = Path('D:/Project/yolov12/RESULT/figures')
IMG_SIZE     = 640
CONF_THRES   = 0.25
IOU_MATCH    = 0.5         # IoU threshold untuk TP/FP klasifikasi

# Pilihan image (dari list lo)
SUCCESS_IMAGES = [
    'tuberculosis-phone-0052.jpg',   # low density (~2 bacilli)
    'tuberculosis-phone-0066.jpg',   # high density (~16 bacilli)
]
ERROR_IMAGE = 'tuberculosis-phone-0062.jpg'   # high density, more challenging

# =============================================================
# Helpers
# =============================================================
def parse_gt_xml(img_path):
    """Read PASCAL VOC XML, return list of (x1, y1, x2, y2)."""
    xml_path = img_path.with_suffix('.xml')
    if not xml_path.exists():
        return []
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall('object'):
        bb = obj.find('bndbox') or obj.find('bbox') or obj.find('box')
        if bb is None: continue
        try:
            x1 = float(bb.find('xmin').text); y1 = float(bb.find('ymin').text)
            x2 = float(bb.find('xmax').text); y2 = float(bb.find('ymax').text)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
        except (AttributeError, ValueError, TypeError):
            continue
    return boxes


def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def match_predictions(gt_boxes, pred_boxes, iou_thres=0.5):
    """Greedy IoU matching. Returns (tp_idx, fp_idx, fn_idx)."""
    matched_gt = set()
    tp_idx, fp_idx = [], []
    # Sort preds by area (larger first; or use confidence if available)
    for i, p in enumerate(pred_boxes):
        best_iou, best_j = 0, -1
        for j, g in enumerate(gt_boxes):
            if j in matched_gt: continue
            iou = compute_iou(p, g)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thres:
            matched_gt.add(best_j)
            tp_idx.append(i)
        else:
            fp_idx.append(i)
    fn_idx = [j for j in range(len(gt_boxes)) if j not in matched_gt]
    return tp_idx, fp_idx, fn_idx


def draw_boxes_on_axis(ax, img_rgb, gt_boxes, pred_boxes, tp_idx, fp_idx, fn_idx):
    """Draw GT (yellow), TP (green), FP (red), FN (magenta) bboxes."""
    ax.imshow(img_rgb)
    # GT (yellow, thin)
    for g in gt_boxes:
        ax.add_patch(Rectangle((g[0], g[1]), g[2]-g[0], g[3]-g[1],
                               fill=False, edgecolor='yellow', linewidth=1.2))
    # TP (green)
    for i in tp_idx:
        p = pred_boxes[i]
        ax.add_patch(Rectangle((p[0], p[1]), p[2]-p[0], p[3]-p[1],
                               fill=False, edgecolor='lime', linewidth=2.0))
    # FP (red)
    for i in fp_idx:
        p = pred_boxes[i]
        ax.add_patch(Rectangle((p[0], p[1]), p[2]-p[0], p[3]-p[1],
                               fill=False, edgecolor='red', linewidth=2.0))
    # FN (magenta)
    for j in fn_idx:
        g = gt_boxes[j]
        ax.add_patch(Rectangle((g[0], g[1]), g[2]-g[0], g[3]-g[1],
                               fill=False, edgecolor='magenta', linewidth=2.0))
    ax.set_xticks([]); ax.set_yticks([])


def run_inference(model, img_path):
    """Returns list of pred bboxes in (x1, y1, x2, y2) format."""
    results = model(str(img_path), imgsz=IMG_SIZE, conf=CONF_THRES, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) else []
    return [tuple(b) for b in boxes]


# =============================================================
# Main
# =============================================================
from ultralytics import YOLO
print(f'Loading model: {WEIGHTS_PATH}')
model = YOLO(WEIGHTS_PATH)

OUT_DIR.mkdir(parents=True, exist_ok=True)


def process_image(img_name):
    img_path = SOURCE_DIR / img_name
    assert img_path.exists(), f'Image not found: {img_path}'
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gt = parse_gt_xml(img_path)
    pred = run_inference(model, img_path)
    tp_idx, fp_idx, fn_idx = match_predictions(gt, pred, IOU_MATCH)
    return img_rgb, gt, pred, tp_idx, fp_idx, fn_idx


# ----- Gambar 4.15: SUCCESS cases (low + high density) -----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
labels = ['(a) Kepadatan rendah', '(b) Kepadatan tinggi']
for ax, img_name, label in zip(axes, SUCCESS_IMAGES, labels):
    img_rgb, gt, pred, tp, fp, fn = process_image(img_name)
    draw_boxes_on_axis(ax, img_rgb, gt, pred, tp, fp, fn)
    ax.set_xlabel(f'{label} — GT={len(gt)}  TP={len(tp)}  FP={len(fp)}  FN={len(fn)}',
                  fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig4_15_detection_success.png',
            dpi=200, bbox_inches='tight')
plt.close()
print('Saved: fig4_15_detection_success.png')

# ----- Gambar 4.16: ERROR case (single image, larger) -----
fig, ax = plt.subplots(figsize=(10, 7))
img_rgb, gt, pred, tp, fp, fn = process_image(ERROR_IMAGE)
draw_boxes_on_axis(ax, img_rgb, gt, pred, tp, fp, fn)
ax.set_xlabel(
    f'{ERROR_IMAGE} — GT={len(gt)}  TP={len(tp)}  FP={len(fp)}  FN={len(fn)}',
    fontsize=11
)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig4_16_detection_errors.png',
            dpi=200, bbox_inches='tight')
plt.close()
print('Saved: fig4_16_detection_errors.png')

print('\nLegend reminder:')
print('  Kuning  = Ground Truth')
print('  Hijau   = True Positive')
print('  Merah   = False Positive')
print('  Magenta = False Negative (missed)')
