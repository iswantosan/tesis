# P3-Weighted Loss Usage Guide

## Konsep

P3-weighted loss memberikan bobot lebih besar pada loss dari P3 (scale terkecil) dibanding P4/P5. Ini powerful untuk small object detection seperti BTA karena:

- **BTA hampir selalu terdeteksi di P3** (resolusi tertinggi)
- **Loss YOLO standar treat semua scale "sama penting"**
- **Model jadi "terlalu nyaman" belajar dari objek besar / easy negative**
- **Ini mengubah prioritas belajar tanpa sentuh backbone**

## Cara Menggunakan

### 1. Via Hyperparameters (default.yaml atau training args)

Tambahkan hyperparameter berikut ke config training:

```yaml
# P3-weighted loss parameters
p3_weight: 1.5  # Bobot untuk P3 loss (default: 1.5)
p4_weight: 1.0  # Bobot untuk P4 loss (default: 1.0)
p5_weight: 1.0  # Bobot untuk P5 loss (default: 1.0)
```

### 2. Via Python Code

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov12.yaml')

# Set hyperparameters untuk P3-weighted loss
model.args.p3_weight = 1.5  # P3 gets 1.5x weight
model.args.p4_weight = 1.0  # P4 normal
model.args.p5_weight = 1.0  # P5 normal

# Train
model.train(data='your_dataset.yaml', epochs=100)
```

### 3. Via Command Line

```bash
# Training dengan P3-weighted loss
yolo detect train data=your_dataset.yaml model=yolov12.yaml epochs=100 p3_weight=1.5 p4_weight=1.0 p5_weight=1.0
```

## Parameter Recommendations

### Conservative (Recommended Start)
- `p3_weight: 1.2` - Peningkatan ringan, aman untuk training awal
- `p4_weight: 1.0`
- `p5_weight: 1.0`

### Moderate (Good for BTA)
- `p3_weight: 1.5` - **Recommended untuk BTA**
- `p4_weight: 1.0`
- `p5_weight: 1.0`

### Aggressive (Use with Caution)
- `p3_weight: 2.0` - Hanya jika P3 benar-benar dominan
- `p4_weight: 1.0`
- `p5_weight: 1.0`

## Implementation Details

Loss dihitung dengan memberikan bobot berbeda pada setiap anchor berdasarkan stride-nya:

- **P3 anchors** (stride=8): Dapat bobot `p3_weight`
- **P4 anchors** (stride=16): Dapat bobot `p4_weight`
- **P5 anchors** (stride=32): Dapat bobot `p5_weight`

Bobot diterapkan pada:
- **Classification loss**: Per-anchor weighting
- **Box/IoU loss**: Per-anchor weighting
- **DFL loss**: Per-anchor weighting

## Expected Effects

### Positive Effects
- ✅ **Recall small objects naik** (BTA lebih mudah terdeteksi)
- ✅ **Confidence small objects naik** (model lebih yakin)
- ✅ **FP di background turun** (model fokus ke detail kecil)
- ✅ **mAP50 small objects naik**

### Potential Issues
- ⚠️ **Large object detection mungkin sedikit turun** (trade-off)
- ⚠️ **Training mungkin lebih lama converge** (jika bobot terlalu besar)

## Tips

1. **Start dengan bobot kecil** (1.2-1.3) dan naikkan secara bertahap
2. **Monitor loss curves** - pastikan loss tidak explode
3. **Compare dengan baseline** - pastikan overall mAP tidak turun drastis
4. **Fine-tune berdasarkan dataset** - setiap dataset punya karakteristik berbeda

## Example Training Script

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov12.yaml')

# Configure P3-weighted loss
model.args.p3_weight = 1.5
model.args.p4_weight = 1.0
model.args.p5_weight = 1.0

# Train
results = model.train(
    data='bta_dataset.yaml',
    epochs=300,
    imgsz=640,
    batch=16,
    device=0,
)

# Validate
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
```

## Notes

- **Tidak perlu ubah arsitektur** - ini pure loss weighting
- **Compatible dengan semua YOLOv12 variants** (n/s/m/l/x)
- **Bisa dikombinasikan dengan blocks lain** (BGSuppressP3, DecoupledP3Detect, dll)
- **Auto-enabled** jika `p3_weight > 1.0` di hyperparameters
























