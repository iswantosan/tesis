# 📈 Panduan Meningkatkan mAP (mean Average Precision)

## 🎯 Strategi Utama untuk Improve mAP

### 1. **Data Quality & Quantity** (PALING PENTING!)
```yaml
# ✅ Yang Harus Dilakukan:
- Pastikan dataset BALANCED (tidak ada class yang terlalu sedikit)
- Minimal 100-200 images per class (lebih banyak = lebih baik)
- Annotation HARUS AKURAT dan konsisten
- Data augmentation yang sesuai dengan use case
- Train/Val split: 80/20 atau 70/30
```

**Tips:**
- Gunakan **data cleaning**: Hapus image buruk/duplikat
- **Review annotations** secara manual untuk accuracy
- **Balancing classes**: Jika ada class imbalance, gunakan class weights atau oversample

---

### 2. **Hyperparameter Tuning**

#### A. Learning Rate (LR) - Sangat Kritis!
```python
# Untuk dataset kecil (< 1000 images):
lr0: 0.001  # Turunkan dari default 0.01
lrf: 0.1    # Final LR = lr0 * lrf

# Untuk dataset besar:
lr0: 0.01   # Default
lrf: 0.01   # Default

# Jika overfitting:
lr0: 0.0005  # Turunkan lebih rendah
warmup_epochs: 5.0  # Tingkatkan warmup
```

#### B. Loss Weights
```yaml
box: 7.5   # Box regression loss (default)
cls: 0.5   # Classification loss
dfl: 1.5   # Distribution Focal Loss

# Jika banyak false positives:
cls: 1.0   # Tingkatkan class loss

# Jika bbox tidak akurat:
box: 10.0  # Tingkatkan box loss
```

#### C. Batch Size & Image Size
```yaml
batch: 16      # Jika GPU memory cukup, naikkan ke 32/64
imgsz: 640     # Untuk objek kecil, coba 832 atau 1024
multi_scale: True  # Multi-scale training (lebih lambat tapi lebih robust)
```

---

### 3. **Data Augmentation Strategy**

#### A. Basic Augmentation (Sesuaikan dengan Dataset)
```yaml
# Untuk objek kecil (medical/biology):
hsv_h: 0.015   # Hue (warna) - kecil untuk medical
hsv_s: 0.5     # Saturation - turunkan jika warna penting
hsv_v: 0.4     # Value (brightness)
degrees: 0.0   # Rotation - HATI-HATI untuk medical!
translate: 0.1 # Translation
scale: 0.5     # Scale augmentation
shear: 0.0     # Shear - hati-hati
perspective: 0.0  # Perspective - hati-hati
flipud: 0.0    # Flip up-down
fliplr: 0.5    # Flip left-right
mosaic: 1.0    # Mosaic augmentation (sangat efektif!)
mixup: 0.1     # Mixup (hati-hati untuk medical)
copy_paste: 0.0  # Copy-paste (baik untuk object detection)
```

#### B. Advanced Augmentation
```yaml
# Jika overfitting:
mosaic: 1.0
mixup: 0.15
copy_paste: 0.1

# Jika underfitting (mAP rendah):
# Kurangi augmentation
mosaic: 0.8
mixup: 0.05
```

---

### 4. **Training Strategy**

#### A. Epochs & Early Stopping
```yaml
epochs: 300        # Lebih banyak epochs = lebih baik (jika dataset besar)
patience: 50       # Early stopping patience
close_mosaic: 10   # Disable mosaic di 10 epochs terakhir
```

#### B. Freezing & Fine-tuning
```python
# Strategy 1: Freeze backbone dulu, lalu unfreeze
# Epoch 1-50: Freeze backbone
freeze: 20

# Epoch 51-150: Unfreeze semua
freeze: None

# Strategy 2: Progressive unfreezing
# Epoch 1-30: freeze=20
# Epoch 31-60: freeze=10
# Epoch 61+: freeze=None
```

#### C. Optimizer
```yaml
optimizer: AdamW   # Biasanya lebih baik dari SGD
# atau
optimizer: SGD     # Dengan momentum tinggi
momentum: 0.937
weight_decay: 0.0005
```

---

### 5. **Model Architecture Improvements**

Untuk model Anda (P2-P4 detection):

#### A. Test dengan Scale yang Lebih Besar
```python
# Coba model yang lebih besar:
# yolov12s.yaml atau yolov12m.yaml (bukan 'n')
# Lebih banyak parameters = lebih baik (jika data cukup)
```

#### B. Test Focal Loss (jika banyak hard examples)
```python
# Biasanya sudah ada di YOLOv12, tapi bisa di-tune
```

#### C. Anchor Tuning (jika menggunakan anchor-based)
```yaml
# Pastikan anchor size sesuai dengan objek Anda
# Untuk objek kecil, butuh anchor yang lebih kecil
```

---

### 6. **Post-Training Optimization**

#### A. Test Time Augmentation (TTA)
```python
from ultralytics import YOLO
model = YOLO('best.pt')
results = model.predict(source='test/', augment=True)  # TTA enabled
```

#### B. NMS Tuning
```yaml
iou: 0.6    # Turunkan jika banyak overlapping detections
conf: 0.25  # Confidence threshold (tune berdasarkan val set)
max_det: 300  # Maximum detections per image
```

#### C. Ensemble
```python
# Gunakan multiple models dan ensemble predictions
# Bisa increase mAP 2-5%
```

---

### 7. **Troubleshooting Specific Issues**

#### A. Low mAP@0.5 (Recall rendah = banyak miss detections)
```yaml
# Solusi:
- Tingkatkan data augmentation
- Turunkan confidence threshold saat validation
- Periksa apakah objek terlalu kecil (coba imgsz lebih besar)
- Tambah lebih banyak data
- Check annotation quality
```

#### B. High Recall tapi Low Precision (Banyak false positives)
```yaml
# Solusi:
cls: 1.0    # Tingkatkan class loss weight
- Tingkatkan NMS threshold (iou: 0.7 → 0.75)
- Periksa data: mungkin ada mislabeling
- Gunakan lebih banyak negative samples
```

#### C. Overfitting (Val mAP jauh lebih rendah dari Train mAP)
```yaml
# Solusi:
- Tingkatkan data augmentation
- Turunkan learning rate
- Gunakan dropout (jika ada)
- Freeze lebih banyak layers
- Kurangi model complexity
- Gunakan weight decay yang lebih tinggi
```

#### D. Underfitting (Train & Val mAP sama-sama rendah)
```yaml
# Solusi:
- Tingkatkan model size (yolov12s → yolov12m)
- Turunkan augmentation
- Tingkatkan learning rate (hati-hati!)
- Tambah epochs
- Periksa data quality
```

---

### 8. **Recommended Training Workflow**

```python
# Step 1: Quick test dengan config dasar
# Train 50 epochs dengan default settings

# Step 2: Analyze results
# - Lihat confusion matrix
# - Periksa PR curve
# - Identifikasi class yang performa buruk

# Step 3: Hyperparameter tuning
# - Tune learning rate (paling penting!)
# - Tune loss weights
# - Tune augmentation

# Step 4: Longer training
# - Train 200-300 epochs dengan best hyperparameters
# - Use early stopping

# Step 5: Fine-tuning
# - Lower learning rate untuk 50 epochs terakhir
# - Disable mosaic untuk epochs terakhir
```

---

### 9. **Quick Wins (Paling Mudah & Efektif)**

1. ✅ **Naikkan image size**: 640 → 832 atau 1024 (jika GPU cukup)
2. ✅ **Lebih banyak epochs**: 100 → 200-300
3. ✅ **Tune learning rate**: Coba lr0: 0.001 untuk dataset kecil
4. ✅ **Enable mosaic**: mosaic: 1.0 (sudah default)
5. ✅ **Naikkan batch size**: 16 → 32 (jika memory cukup)
6. ✅ **Periksa annotation quality**: Pastikan tidak ada mislabeling
7. ✅ **Class balancing**: Pastikan semua class punya cukup samples

---

### 10. **Monitoring & Debugging**

```python
# Gunakan tensorboard atau wandb untuk monitoring:
# - Loss curves (train vs val)
# - mAP curves
# - Learning rate schedule
# - Confusion matrix

# Red flags:
# - Train loss turun tapi val loss naik → Overfitting
# - Keduanya tidak turun → Underfitting atau LR terlalu kecil
# - mAP stuck → Perlu tune hyperparameters atau lebih banyak data
```

---

## 🎯 Priority Actions untuk Improve mAP

**Tingkat 1 (HIGH IMPACT):**
1. Data quality & quantity ⭐⭐⭐⭐⭐
2. Learning rate tuning ⭐⭐⭐⭐⭐
3. Image size (naikkan jika objek kecil) ⭐⭐⭐⭐
4. More epochs ⭐⭐⭐⭐

**Tingkat 2 (MEDIUM IMPACT):**
5. Loss weights tuning ⭐⭐⭐
6. Augmentation strategy ⭐⭐⭐
7. Batch size ⭐⭐⭐
8. Optimizer selection ⭐⭐

**Tingkat 3 (LOW-MEDIUM IMPACT):**
9. Architecture changes ⭐⭐
10. NMS tuning ⭐
11. TTA ⭐
12. Ensemble ⭐

---

## 💡 Tips Khusus untuk Medical/Microscopic Detection

Karena Anda menggunakan 1 class (nc: 1) dan P2-P4 detection:

1. **Objek kecil**: Pastikan `imgsz` cukup besar (640+)
2. **Augmentation hati-hati**: Medical images sensitif terhadap rotation/perspective
3. **Focus pada precision**: Untuk medical, false positive lebih berbahaya
4. **Multi-scale training**: Sangat membantu untuk objek kecil
5. **Test dengan P2 focus**: Layer P2 untuk small object detection

---

Good luck! 🚀


