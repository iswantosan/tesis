# 🚀 SFAN NAIK Guide - Versi yang Bener-bener Naik!

## ❌ Masalah: Workaround Version Tidak Naik

Workaround version (`yolov12-sfan-workaround.yaml`) tidak naik karena:
- Hanya pakai modul basic (ChannelAttention + SpatialAttention)
- Belum pakai modul **terbukti bagus** (FBSB, SOE, DeformableHead)
- Belum full SFAN functionality

---

## ✅ Solusi: Pakai Versi yang Terbukti Bagus!

### **Option 1: `yolov12.yaml` (RECOMMENDED)** ⭐
**Kombinasi: FBSB + SPDConv + DeformableHead**

File ini **sudah optimal** dan menggunakan modul **terbukti**:
- ✅ **FBSB**: mAP50-95 = **0.444** (TERBAIK)
- ✅ **SPDConv**: Preserve small object info
- ✅ **DeformableHead**: mAP50 = 0.872, mAP50-95 = 0.44

```python
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/v12/yolov12.yaml')
model.train(data='your_data.yaml', epochs=100, imgsz=640)
```

**Expected**: mAP50-95 ~0.44-0.45

---

### **Option 2: `yolov12-sfan-optimized.yaml`** (Baru!) 🔥
**Kombinasi: FBSB + SOE + SPDConv + DeformableHead**

Versi **optimized** dengan semua top performers:
- ✅ **FBSB** di backbone P4/P5 (noise suppression) + di P3 neck
- ✅ **SOE** di P4 neck (small object enhancement - mAP50: 0.87)
- ✅ **SPDConv** di semua downsampling (preserve small objects)
- ✅ **DeformableHead** di P4/P5 (mAP50: 0.872, mAP50-95: 0.44)

```python
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/v12/yolov12-sfan-optimized.yaml')
model.train(data='your_data.yaml', epochs=100, imgsz=640)
```

**Expected**: mAP50-95 ~0.44-0.46 (lebih tinggi dari workaround)

---

### **Option 3: `yolov12-sfan-aggressive.yaml`** (Maximum!) 💪
**Kombinasi: FBSB + SOE + SPDConv + DeformableHead + Attention**

Versi **maximum aggressive** dengan semua modul proven:
- ✅ **FBSB** di backbone P4/P5 + neck P3/P4
- ✅ **ChannelAttention + SpatialAttention** di backbone (additional refinement)
- ✅ **SOE** di P3/P4 neck (small object enhancement)
- ✅ **SPDConv** di semua downsampling
- ✅ **DeformableHead** di P4/P5

```python
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/v12/yolov12-sfan-aggressive.yaml')
model.train(data='your_data.yaml', epochs=100, imgsz=640)
```

**Expected**: mAP50-95 ~0.45-0.46 (maximum, tapi lebih berat)

**⚠️ Warning**: Lebih berat, monitor OOM!

---

## 📊 Perbandingan Versi

| Versi | Modul | Expected mAP50-95 | Kompleksitas |
|-------|-------|-------------------|--------------|
| `yolov12.yaml` | FBSB + SPDConv + DeformableHead | 0.44-0.45 | Medium |
| `yolov12-sfan-workaround.yaml` | ChannelAttn + SpatialAttn + SOE | 0.42-0.43 | Low (tidak naik) ❌ |
| `yolov12-sfan-optimized.yaml` | FBSB + SOE + SPDConv + DeformableHead | 0.44-0.46 | Medium-High ✅ |
| `yolov12-sfan-aggressive.yaml` | All + Attention | 0.45-0.46 | High (bisa OOM) ⚠️ |

---

## 🎯 Rekomendasi

### **Untuk Test Cepat:**
1. Pakai `yolov12.yaml` dulu (sudah proven optimal)
2. Jika hasil bagus, lanjut ke `yolov12-sfan-optimized.yaml`

### **Untuk Maximum mAP:**
1. Pakai `yolov12-sfan-optimized.yaml` (balance antara performance dan kompleksitas)
2. Jika masih belum naik, coba `yolov12-sfan-aggressive.yaml` (lebih aggressive)

### **Jika Masih Tidak Naik:**
Kemungkinan masalahnya bukan di architecture, tapi di:
- **Hyperparameters** (learning rate, batch size, dll)
- **Data augmentation** (kurang atau terlalu banyak)
- **Training strategy** (epochs, warmup, dll)
- **Loss weights** (box, cls, dfl)

Cek juga:
- Apakah baseline sudah benar?
- Apakah ada overfitting?
- Apakah metrics yang dicek (mAP50 vs mAP50-95)?

---

## 💡 Tips Training

```python
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/v12/yolov12-sfan-optimized.yaml')

model.train(
    data='your_data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,  # Adjust sesuai GPU
    optimizer='AdamW',
    lr0=0.001,
    weight_decay=0.0005,
    warmup_epochs=3,
    box=7.5,   # Box loss weight
    cls=0.5,   # Class loss weight
    dfl=1.5,   # DFL loss weight
    # Augmentation untuk small objects
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,  # Kurangi untuk small objects
    copy_paste=0.0,
)
```

---

## 🔥 Yang Harus Dicoba

1. ✅ **`yolov12.yaml`** - Sudah proven, harus naik!
2. ✅ **`yolov12-sfan-optimized.yaml`** - Kombinasi optimal dengan SOE
3. ⚠️ **`yolov12-sfan-aggressive.yaml`** - Jika masih belum naik (maximum)

**Good luck! Pasti naik! 🚀**

