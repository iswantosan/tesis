# 📊 Faktor-Faktor yang Mempengaruhi mAP50

## 🎯 Definisi mAP50

**mAP50** = Mean Average Precision pada IoU threshold **0.5**

- **AP (Average Precision)**: Area di bawah Precision-Recall curve untuk setiap kelas
- **mAP50**: Rata-rata AP@0.5 untuk semua kelas
- **IoU 0.5**: Deteksi dianggap benar jika IoU ≥ 0.5 dengan ground truth

---

## 📈 Faktor Utama yang Mempengaruhi mAP50

### 1. **Precision & Recall** ⭐⭐⭐⭐⭐ (PALING PENTING!)

mAP50 dihitung dari **Precision-Recall curve**, jadi kedua metrik ini adalah faktor utama:

#### **Precision** (Akurasi Deteksi)
```
Precision = TP / (TP + FP)
```
**Dipengaruhi oleh:**
- ✅ **False Positives (FP)**: Deteksi yang salah
  - Model mendeteksi objek yang tidak ada
  - Class misclassification
  - Background noise terdeteksi sebagai objek

**Cara Naikkan Precision:**
- Tingkatkan `cls` loss weight (1.0 → 1.5)
- Gunakan architecture yang mengurangi FP (FBSB, Texture Punish)
- Naikkan NMS threshold (iou: 0.6 → 0.7)
- Perbaiki kualitas data (kurangi mislabeling)

#### **Recall** (Kemampuan Deteksi)
```
Recall = TP / (TP + FN)
```
**Dipengaruhi oleh:**
- ✅ **False Negatives (FN)**: Objek yang terlewat
  - Objek terlalu kecil tidak terdeteksi
  - Objek terhalang/occluded
  - Confidence terlalu rendah

**Cara Naikkan Recall:**
- Naikkan image size (640 → 832 atau 1024)
- Gunakan P3-weighted loss untuk small objects
- Turunkan confidence threshold saat validation
- Gunakan architecture untuk small objects (SOE, SPDConv)
- Tambah lebih banyak data

---

### 2. **Data Quality & Quantity** ⭐⭐⭐⭐⭐

#### **Kuantitas Data**
- ✅ **Minimal 100-200 images per class** (lebih banyak = lebih baik)
- ✅ **Balanced classes**: Semua kelas punya jumlah data yang seimbang
- ✅ **Train/Val split**: 80/20 atau 70/30

#### **Kualitas Data**
- ✅ **Annotation akurat**: Bounding box tepat dan konsisten
- ✅ **Tidak ada mislabeling**: Class label benar
- ✅ **Data representatif**: Mencakup variasi kondisi (cahaya, angle, scale)
- ✅ **Data cleaning**: Hapus duplikat dan gambar buruk

**Impact**: Data buruk = mAP50 rendah, tidak peduli seberapa bagus modelnya!

---

### 3. **Hyperparameter Training** ⭐⭐⭐⭐

#### **A. Learning Rate (LR)**
```python
# Dataset kecil (< 1000 images):
lr0: 0.001  # Turunkan dari default 0.01
lrf: 0.1    # Final LR = lr0 * lrf

# Dataset besar:
lr0: 0.01   # Default
lrf: 0.01   # Default
```
**Impact**: LR terlalu tinggi → training tidak stabil → mAP50 turun
**Impact**: LR terlalu rendah → training lambat → mAP50 stuck

#### **B. Loss Weights**
```yaml
box: 7.5   # Box regression loss (akurasi bbox)
cls: 0.5   # Classification loss (kurangi FP)
dfl: 1.5   # Distribution Focal Loss

# Jika banyak false positives:
cls: 1.0   # Tingkatkan → Precision naik → mAP50 naik

# Jika bbox tidak akurat:
box: 10.0  # Tingkatkan → IoU naik → mAP50 naik
```

#### **C. Image Size**
```python
imgsz: 640   # Default
imgsz: 832   # Lebih baik untuk small objects (+1-2% mAP50)
imgsz: 1024  # Maximum (jika GPU cukup)
```
**Impact**: Image lebih besar → Small objects lebih jelas → Recall naik → mAP50 naik

#### **D. Batch Size**
```python
batch: 16   # Default
batch: 32   # Lebih stabil (jika GPU cukup)
```
**Impact**: Batch lebih besar → Gradient lebih stabil → Training lebih baik

#### **E. Epochs**
```python
epochs: 100   # Default
epochs: 200-300  # Lebih banyak belajar (+0.5-1.5% mAP50)
```
**Impact**: Lebih banyak epochs → Model belajar lebih baik → mAP50 naik

---

### 4. **Model Architecture** ⭐⭐⭐

#### **A. Backbone (Feature Extraction)**
- ✅ **Large Kernel Conv (7x7, 9x9)**: Global context → mAP50 +1-2%
- ✅ **Global Context Block**: Hubungan antar fitur → mAP50 +1-2%
- ✅ **SPDConv**: Preserve small object info → mAP50 +1-2% untuk small objects

#### **B. Neck (Feature Fusion)**
- ✅ **Enhanced FPN**: Large Kernel + Global Context
- ✅ **BiFPN**: Weighted fusion → mAP50 +1.5-2.5%
- ✅ **FBSB**: Foreground-Background Separation → mAP50-95 = 0.444 (TERBAIK)

#### **C. Head (Detection)**
- ✅ **DW-Decoupled Head**: Spesialisasi cls & reg → mAP50 +1-2%
- ✅ **DeformableHead**: Adaptive geometry → mAP50 = 0.872
- ✅ **SOE**: Small Object Enhancement → mAP50 = 0.87

**Rekomendasi Architecture:**
```python
# Maximum mAP50:
model = YOLO('ultralytics/cfg/models/v12/yolov12-map50-boost.yaml')
# Expected: +1-2% mAP50 dari baseline

# Advanced (BiFPN):
model = YOLO('ultralytics/cfg/models/v12/yolov12-map50-boost-bifpn.yaml')
# Expected: +1.5-2.5% mAP50 dari baseline
```

---

### 5. **Data Augmentation** ⭐⭐⭐

#### **Basic Augmentation**
```yaml
# Untuk small objects:
hsv_h: 0.015   # Hue (kecil untuk medical)
hsv_s: 0.5     # Saturation
hsv_v: 0.4     # Brightness
degrees: 0.0   # Rotation (hati-hati!)
translate: 0.1 # Translation
scale: 0.5     # Scale augmentation
fliplr: 0.5    # Flip left-right
mosaic: 1.0    # Mosaic (sangat efektif!)
mixup: 0.1     # Mixup (hati-hati untuk medical)
```

#### **Impact Augmentation:**
- ✅ **Mosaic**: Sangat efektif untuk meningkatkan mAP50
- ⚠️ **Augmentation berlebihan**: Bisa menurunkan mAP50 (overfitting ke augmented data)
- ⚠️ **Augmentation tidak sesuai**: Medical images sensitif terhadap rotation/perspective

---

### 6. **Training Strategy** ⭐⭐⭐

#### **A. Freeze & Unfreeze**
```python
# Strategy 1: Freeze backbone dulu
freeze: 20  # Freeze 20 layers pertama
# Lalu unfreeze semua setelah beberapa epochs

# Strategy 2: Progressive unfreezing
# Epoch 1-30: freeze=20
# Epoch 31-60: freeze=10
# Epoch 61+: freeze=None
```

#### **B. Optimizer**
```python
optimizer: AdamW   # Biasanya lebih baik dari SGD
# atau
optimizer: SGD
momentum: 0.937
weight_decay: 0.0005
```

#### **C. Learning Rate Schedule**
```python
warmup_epochs: 3.0  # Warmup untuk stabilisasi
close_mosaic: 10    # Disable mosaic di 10 epochs terakhir
```

---

### 7. **P3-Weighted Loss** ⭐⭐⭐⭐ (Khusus Small Objects)

```python
model.train(
    data='your_data.yaml',
    p3_weight=1.5,  # P3 (stride=8) dapat bobot 1.5x
    p4_weight=1.0,
    p5_weight=1.0,
)
```

**Impact**: 
- ✅ Recall small objects naik
- ✅ mAP50 small objects naik (+1-3%)
- ⚠️ Large object detection mungkin sedikit turun (trade-off)

---

### 8. **Post-Training Optimization** ⭐⭐

#### **A. Test Time Augmentation (TTA)**
```python
results = model.predict(source='test/', augment=True)
```
**Impact**: +0.5-1% mAP50

#### **B. NMS Tuning**
```yaml
iou: 0.6    # Turunkan jika banyak overlapping detections
conf: 0.25  # Confidence threshold (tune berdasarkan val set)
max_det: 300  # Maximum detections per image
```

#### **C. Ensemble**
```python
# Gunakan multiple models dan ensemble predictions
# Impact: +2-5% mAP50
```

---

## 🎯 Priority Actions untuk Naikkan mAP50

### **Tingkat 1 (HIGH IMPACT - +2-5% mAP50):**
1. ✅ **Data quality & quantity** ⭐⭐⭐⭐⭐
2. ✅ **Learning rate tuning** ⭐⭐⭐⭐⭐
3. ✅ **Image size** (naikkan jika small objects) ⭐⭐⭐⭐
4. ✅ **More epochs** (100 → 200-300) ⭐⭐⭐⭐

### **Tingkat 2 (MEDIUM IMPACT - +1-2% mAP50):**
5. ✅ **Loss weights tuning** ⭐⭐⭐
6. ✅ **P3-weighted loss** (untuk small objects) ⭐⭐⭐
7. ✅ **Architecture improvements** (mAP50-boost variants) ⭐⭐⭐
8. ✅ **Augmentation strategy** ⭐⭐⭐

### **Tingkat 3 (LOW-MEDIUM IMPACT - +0.5-1% mAP50):**
9. ✅ **Batch size** ⭐⭐
10. ✅ **Optimizer selection** ⭐⭐
11. ✅ **NMS tuning** ⭐
12. ✅ **TTA** ⭐

---

## 📊 Formula mAP50

```
1. Untuk setiap deteksi:
   - Hitung IoU dengan ground truth
   - Jika IoU ≥ 0.5 → True Positive (TP)
   - Jika IoU < 0.5 atau salah class → False Positive (FP)
   - Ground truth yang tidak terdeteksi → False Negative (FN)

2. Hitung Precision & Recall:
   Precision = TP / (TP + FP)
   Recall = TP / (TP + FN)

3. Buat Precision-Recall curve (variasi confidence threshold)

4. Hitung AP@0.5 = Area di bawah PR curve untuk setiap kelas

5. mAP50 = Rata-rata AP@0.5 untuk semua kelas
```

---

## 🔍 Troubleshooting mAP50 Rendah

### **Low Precision (Banyak FP):**
```yaml
# Solusi:
cls: 1.0    # Tingkatkan class loss
iou: 0.7    # Naikkan NMS threshold
# Periksa data: mungkin ada mislabeling
```

### **Low Recall (Banyak FN):**
```yaml
# Solusi:
imgsz: 832  # Naikkan image size
p3_weight: 1.5  # P3-weighted loss
conf: 0.2   # Turunkan confidence threshold
# Tambah lebih banyak data
```

### **Overfitting (Val mAP50 << Train mAP50):**
```yaml
# Solusi:
- Tingkatkan data augmentation
- Turunkan learning rate
- Gunakan weight decay yang lebih tinggi
- Freeze lebih banyak layers
```

### **Underfitting (Train & Val mAP50 sama-sama rendah):**
```yaml
# Solusi:
- Naikkan model size (yolov12n → yolov12s)
- Turunkan augmentation
- Naikkan learning rate (hati-hati!)
- Tambah epochs
- Periksa data quality
```

---

## 💡 Quick Wins (Paling Mudah & Efektif)

1. ✅ **Naikkan image size**: 640 → 832 atau 1024 (+1-2% mAP50)
2. ✅ **Lebih banyak epochs**: 100 → 200-300 (+0.5-1.5% mAP50)
3. ✅ **Tune learning rate**: lr0: 0.001 untuk dataset kecil (+0.5-1% mAP50)
4. ✅ **P3-weighted loss**: p3_weight=1.5 untuk small objects (+1-3% mAP50)
5. ✅ **Pakai mAP50-boost architecture**: +1-2% mAP50
6. ✅ **Periksa annotation quality**: Pastikan tidak ada mislabeling
7. ✅ **Class balancing**: Pastikan semua class punya cukup samples

---

## 📚 Referensi

- `MAP50_BOOST_GUIDE.md`: Panduan lengkap untuk boost mAP50
- `MAP_IMPROVEMENT_GUIDE.md`: Strategi umum improve mAP
- `MAP_BOOST_TIPS.md`: Quick tips untuk naikkan mAP
- `P3_WEIGHTED_LOSS_USAGE.md`: Panduan P3-weighted loss

---

**Kesimpulan**: mAP50 dipengaruhi oleh **Precision & Recall**, yang dipengaruhi oleh **data quality**, **hyperparameters**, **architecture**, dan **training strategy**. Fokus pada faktor-faktor HIGH IMPACT terlebih dahulu!

























