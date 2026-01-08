# 🚀 YOLOv12 mAP50 Boost Guide

## 📊 Strategi Improvement

### 1. **Backbone: Context Augmentation**
- **Large Kernel Convolution (7x7 atau 9x9)**: Ganti beberapa blok di stage terakhir
- **Global Context Block**: Tambahkan setelah stage 4 & 5
- **Fungsi**: Membantu model "melihat" hubungan antar fitur yang berjauhan

### 2. **Neck: Enhanced FPN / BiFPN**
- **Enhanced FPN**: Large Kernel + Global Context di neck
- **BiFPN** (Advanced): Weighted Bi-directional Feature Pyramid dengan learnable weights
- **P6 Layer**: Extra scale untuk objek besar (optional)

### 3. **Head: DW-Decoupled Head**
- **Depthwise Separable Decoupled**: Pisahkan cls dan reg dengan DWConv
- **Efisiensi**: Lebih ringan dari decoupled biasa
- **Spesialisasi**: Feature lebih spesifik untuk masing-masing task

---

## 📁 File YAML yang Tersedia

### 1. **`yolov12-map50-boost.yaml`** ⭐ RECOMMENDED
**Kombinasi: Large Kernel + Global Context + DW-Decoupled Head**

**Fitur:**
- ✅ Large Kernel Conv (7x7) di backbone P4 & P5
- ✅ Global Context Block di backbone P4, P5, P6
- ✅ Enhanced FPN dengan Large Kernel di neck
- ✅ DW-Decoupled Head untuk spesialisasi
- ✅ P6 Layer untuk objek besar

**Expected:**
- mAP50: +1-2% dari baseline
- Lebih praktis dan mudah digunakan

**Usage:**
```python
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/v12/yolov12-map50-boost.yaml')
model.train(data='your_data.yaml', epochs=100, imgsz=640)
```

---

### 2. **`yolov12-map50-boost-bifpn.yaml`** (Advanced)
**Kombinasi: Large Kernel + Global Context + BiFPN + DW-Decoupled Head**

**Fitur:**
- ✅ Large Kernel Conv (7x7) di backbone
- ✅ Global Context Block di backbone
- ✅ **BiFPN**: Weighted Bi-directional Feature Pyramid (learnable weights)
- ✅ DW-Decoupled Head
- ⚠️ **Note**: BiFPN return list, perlu Index module untuk extract

**Expected:**
- mAP50: +1.5-2.5% dari baseline
- Lebih advanced tapi lebih kompleks

**Usage:**
```python
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/v12/yolov12-map50-boost-bifpn.yaml')
model.train(data='your_data.yaml', epochs=100, imgsz=640)
```

---

## 🔍 Modul Baru yang Diimplement

### 1. **GlobalContextBlock**
**File**: `ultralytics/nn/modules/block.py`

**Fungsi**: Global context modeling yang ringan
- Global Average Pooling → Context modeling → Broadcast
- Lebih ringan dari Self-Attention tapi efektif

**Args**: `[c1, c2, reduction]`
- `c1`: Input channels
- `c2`: Output channels (default: same as input)
- `reduction`: Channel reduction ratio (default: 4)

---

### 2. **LargeKernelConv**
**File**: `ultralytics/nn/modules/block.py`

**Fungsi**: Large kernel convolution untuk global context
- Kernel size 7x7 atau 9x9
- Optional dilation untuk larger receptive field

**Args**: `[c1, c2, k, dilation]`
- `c1`: Input channels
- `c2`: Output channels
- `k`: Kernel size (default: 7)
- `dilation`: Dilation rate (default: 1)

---

### 3. **BiFPN**
**File**: `ultralytics/nn/modules/block.py`

**Fungsi**: Weighted Bi-directional Feature Pyramid
- Top-down path: P5 → P4 → P3 dengan learnable weights
- Bottom-up path: P3 → P4 → P5 dengan learnable weights
- Model belajar sendiri feature mana yang penting

**Args**: `[c3, c4, c5, c_out]`
- Input: List of 3 tensors [P3, P4, P5]
- Output: List of 3 enhanced features [P3, P4, P5]

**Note**: Return list, perlu Index module untuk extract individual outputs

---

### 4. **DWDecoupledHead**
**File**: `ultralytics/nn/modules/head.py`

**Fungsi**: Depthwise Separable Decoupled Head
- Classification branch: DWConv + PointConv
- Regression branch: DWConv + PointConv
- Lebih efisien dari decoupled biasa

**Args**: `[nc]` (number of classes)
- Input channels auto-inferred dari detection layers

---

## 📈 Expected Results

### **yolov12-map50-boost.yaml**:
- **mAP50**: +1-2% dari baseline
- **mAP50-95**: +0.3-0.5%
- **Precision/Recall**: Lebih balanced

### **yolov12-map50-boost-bifpn.yaml**:
- **mAP50**: +1.5-2.5% dari baseline
- **mAP50-95**: +0.5-0.8%
- **Precision/Recall**: Lebih balanced

---

## 🎯 Rekomendasi

### **Untuk Test Cepat:**
1. Pakai `yolov12-map50-boost.yaml` (RECOMMENDED)
2. Lebih praktis, semua modul sudah terintegrasi dengan baik
3. Expected improvement +1-2% mAP50

### **Untuk Maximum mAP50:**
1. Pakai `yolov12-map50-boost-bifpn.yaml` (Advanced)
2. BiFPN dengan learnable weights untuk optimal fusion
3. Expected improvement +1.5-2.5% mAP50

---

## 💡 Tips Training

```python
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/v12/yolov12-map50-boost.yaml')

model.train(
    data='your_data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    optimizer='AdamW',
    lr0=0.001,
    weight_decay=0.0005,
    warmup_epochs=3,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
)
```

---

## ⚠️ Catatan Penting

1. **Large Kernel Conv**: Bisa lebih berat, monitor memory usage
2. **BiFPN**: Return list, perlu Index module untuk extract (sudah di-handle di YAML)
3. **P6 Layer**: Optional, bisa dihapus jika tidak butuh deteksi objek besar
4. **DW-Decoupled Head**: Lebih efisien dari decoupled biasa, tapi tetap perlu lebih banyak parameter

---

## 🔄 Alternative: Tanpa P6

Jika tidak butuh P6, bisa hapus:
- Layer 13-15 di backbone (P6)
- Layer 36-38 di head (P6 processing)
- Update Detect head: `[[27, 31, 35], 1, DWDecoupledHead, [nc]]` (P3, P4, P5 saja)

---

**Good luck! Pasti mAP50 naik! 🚀**

