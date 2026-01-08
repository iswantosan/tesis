# 🚀 YOLOv12 Optimization Guide - Naikkan mAP!

Berdasarkan hasil testing 100 epoch, berikut **3 variasi optimal** yang sudah dibuat:

## 📊 Analisis Hasil Testing

**Top Performers:**
- **FBSB**: mAP50-95 = **0.444** ⭐ (TERBAIK)
- **FBSBMS**: Precision = **0.815** ⭐ (TERBAIK)
- **DPRB**: Recall = **0.821** ⭐ (TERBAIK)
- **DeformableHead**: mAP50 = **0.872**, mAP50-95 = **0.44** ⭐
- **SOE**: mAP50 = **0.87**, mAP50-95 = **0.437** (bagus untuk small objects)

---

## 🎯 3 Variasi Konfigurasi

### 1. **yolov12.yaml** (File Utama) - ⭐ RECOMMENDED
**Kombinasi: FBSB + SPDConv + DeformableHead**

**Target:** Maximum mAP50-95
- ✅ **FBSB** di P3 & P4 (mAP50-95: 0.444)
- ✅ **SPDConv** untuk downsampling (preserve small objects)
- ✅ **DeformableHead** di P4 & P5 (mAP50: 0.872, mAP50-95: 0.44)

**Expected Performance:**
- Precision: ~0.80
- Recall: ~0.80
- mAP50: ~0.87-0.88
- **mAP50-95: ~0.44-0.45** 🎯

**Usage:**
```python
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/v12/yolov12.yaml')
model.train(data='your_data.yaml', epochs=100, imgsz=640)
```

---

### 2. **yolov12_optimal.yaml** - Maximum mAP
**Kombinasi: FBSB + SPDConv + DeformableHead** (sama dengan file utama)

**Best for:** Mencapai mAP50-95 tertinggi
- FBSB: Best mAP50-95 (0.444)
- DeformableHead: Adaptive geometric transformation
- SPDConv: Preserve spatial info untuk small objects

---

### 3. **yolov12_balanced.yaml** - Precision Focus
**Kombinasi: FBSBMS + SPDConv + SOE**

**Target:** High Precision + Small Object Detection
- ✅ **FBSBMS**: Best Precision (0.815) - Multi-scale mask
- ✅ **SPDConv**: Preserve small object info
- ✅ **SOE**: Small Object Enhancement (0.87 mAP50, 0.437 mAP50-95)

**Expected Performance:**
- **Precision: ~0.81-0.82** 🎯
- Recall: ~0.78-0.79
- mAP50: ~0.87
- mAP50-95: ~0.44

**Best for:** Dataset dengan banyak false positives, butuh precision tinggi

---

### 4. **yolov12_lightweight.yaml** - Recall Focus (Hati-hati OOM)
**Kombinasi: FBSB + SPDConv + DPRB**

**Target:** High Recall + Lightweight
- ✅ **FBSB**: Best mAP50-95 (0.444)
- ✅ **DPRB**: Best Recall (0.821) - Dense refinement
- ✅ **SPDConv**: Preserve small object info

**Expected Performance:**
- Precision: ~0.76-0.77
- **Recall: ~0.82-0.83** 🎯
- mAP50: ~0.86-0.87
- mAP50-95: ~0.44

**Best for:** Dataset dengan banyak false negatives, butuh recall tinggi

**⚠️ Warning:** DPRB menggunakan dense connections, bisa lebih berat. Monitor OOM!

---

## 🔍 Penjelasan Modules

### **FBSB** (Foreground-Background Separation Block)
- **Hasil:** mAP50-95 = 0.444 (TERBAIK)
- **Cara kerja:** Memisahkan foreground dan background features
- **Best for:** General object detection

### **FBSBMS** (FBSB-MultiScale)
- **Hasil:** Precision = 0.815 (TERBAIK)
- **Cara kerja:** Multi-scale mask generation (1x1, 3x3, 5x5)
- **Best for:** High precision requirements

### **DPRB** (Dense Prediction Refinement Block)
- **Hasil:** Recall = 0.821 (TERBAIK)
- **Cara kerja:** Dense connections dengan 3 cascade conv layers
- **Best for:** High recall requirements
- **⚠️ Warning:** Bisa lebih berat, hati-hati OOM!

### **DeformableHead**
- **Hasil:** mAP50 = 0.872, mAP50-95 = 0.44
- **Cara kerja:** Adaptive geometric transformation dengan offset learning
- **Best for:** Complex object shapes

### **SOE** (Small Object Enhancement)
- **Hasil:** mAP50 = 0.87, mAP50-95 = 0.437
- **Cara kerja:** DepthWise Conv 5x5 + Channel Attention
- **Best for:** Small object detection

### **SPDConv** (Spatial-to-Depth Convolution)
- **Cara kerja:** Preserve spatial information saat downsampling
- **Best for:** Small object preservation
- **Posisi:** Di bottom-up path (PAN) untuk P3->P4 dan P4->P5

---

## 📈 Strategi Training

### **100 Epochs - Recommended Settings:**

```python
from ultralytics import YOLO

# Pilih salah satu:
model = YOLO('ultralytics/cfg/models/v12/yolov12.yaml')  # RECOMMENDED
# model = YOLO('ultralytics/cfg/models/v12/yolov12_balanced.yaml')  # Precision focus
# model = YOLO('ultralytics/cfg/models/v12/yolov12_lightweight.yaml')  # Recall focus

# Training dengan optimal settings
model.train(
    data='your_data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,  # Adjust sesuai GPU memory
    optimizer='AdamW',  # Atau 'SGD'
    lr0=0.001,
    weight_decay=0.0005,
    warmup_epochs=3,
    box=7.5,  # Box loss weight
    cls=0.5,  # Class loss weight
    dfl=1.5,  # DFL loss weight
    hsv_h=0.015,  # HSV-Hue augmentation
    hsv_s=0.7,  # HSV-Saturation augmentation
    hsv_v=0.4,  # HSV-Value augmentation
    degrees=0.0,  # Rotation augmentation
    translate=0.1,  # Translation augmentation
    scale=0.5,  # Scale augmentation
    shear=0.0,  # Shear augmentation
    perspective=0.0,  # Perspective augmentation
    flipud=0.0,  # Vertical flip
    fliplr=0.5,  # Horizontal flip
    mosaic=1.0,  # Mosaic augmentation
    mixup=0.0,  # Mixup augmentation
    copy_paste=0.0,  # Copy-paste augmentation
)
```

---

## 🎯 Rekomendasi Berdasarkan Use Case

### **1. Maximum mAP50-95** → `yolov12.yaml` atau `yolov12_optimal.yaml`
- FBSB + SPDConv + DeformableHead
- **Expected:** mAP50-95 ~0.44-0.45

### **2. High Precision** → `yolov12_balanced.yaml`
- FBSBMS + SPDConv + SOE
- **Expected:** Precision ~0.81-0.82

### **3. High Recall** → `yolov12_lightweight.yaml`
- FBSB + SPDConv + DPRB
- **Expected:** Recall ~0.82-0.83
- ⚠️ **Hati-hati OOM!**

### **4. Small Objects** → `yolov12.yaml` (sudah ada SPDConv + SOE bisa ditambah)
- SPDConv sudah ada di semua variasi
- Bisa tambahkan SOE di P3 jika perlu

---

## 🔧 Tips Optimasi Lebih Lanjut

1. **Learning Rate Scheduling:**
   - Gunakan cosine annealing atau reduce on plateau
   - Start dengan lr0=0.001, adjust berdasarkan loss curve

2. **Data Augmentation:**
   - Untuk small objects: kurangi mosaic, tingkatkan copy_paste
   - Untuk general: gunakan default settings

3. **Loss Weights:**
   - Jika banyak false positives: tingkatkan cls weight
   - Jika banyak false negatives: tingkatkan box weight

4. **Multi-Scale Training:**
   - Coba imgsz=[640, 800] untuk multi-scale
   - Atau gunakan imgsz=640 untuk konsistensi

5. **Ensemble:**
   - Train beberapa variasi, ensemble untuk hasil terbaik
   - Combine predictions dari FBSB, FBSBMS, dan DeformableHead

---

## 📝 Catatan Penting

- ✅ **SPDConv** sudah ditambahkan di semua variasi untuk preserve small objects
- ✅ **FBSB** memberikan mAP50-95 terbaik (0.444)
- ✅ **DeformableHead** memberikan mAP50 terbaik (0.872)
- ⚠️ **DPRB** bisa menyebabkan OOM, monitor memory usage
- ⚠️ **FBSBMS** lebih berat dari FBSB, tapi precision lebih tinggi

---

## 🚀 Next Steps

1. **Test `yolov12.yaml`** (RECOMMENDED) - 100 epochs
2. **Compare dengan baseline** (backbone 84, neck/head 87)
3. **Monitor metrics:** Precision, Recall, mAP50, mAP50-95
4. **Adjust hyperparameters** berdasarkan loss curve
5. **Fine-tune** jika perlu (tambah SOE, adjust loss weights, dll)

**Good luck! 🎯**


