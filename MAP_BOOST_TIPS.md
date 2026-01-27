# 🚀 Tips Naikkan mAP - Quick Reference

## ⚡ Quick Wins (Paling Mudah & Efektif)

### 1. **P3 Weighted Loss** ⭐⭐⭐⭐⭐
**Cara Pakai:**
```python
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/v12/yolov12-ide1-local-contrast.yaml')

# Set P3-weighted loss (WAJIB untuk small objects!)
model.train(
    data='your_data.yaml',
    epochs=300,
    p3_weight=1.5,  # P3 dapat bobot 1.5x (RECOMMENDED)
    p4_weight=1.0,
    p5_weight=1.0,
    imgsz=640,
    batch=16,
)
```

**Expected:** +1-3% mAP50 untuk small objects

---

### 2. **Naikkan Image Size** ⭐⭐⭐⭐⭐
```python
model.train(
    data='your_data.yaml',
    imgsz=832,  # Naikkan dari 640 ke 832 atau 1024
    # atau
    imgsz=1024,  # Lebih besar = lebih baik (jika GPU cukup)
    batch=8,  # Kurangi batch size kalau memory kurang
)
```

**Expected:** +1-2% mAP50 untuk small objects

---

### 3. **Tune Learning Rate** ⭐⭐⭐⭐⭐
```python
model.train(
    data='your_data.yaml',
    lr0=0.001,  # Turunkan dari 0.01 untuk dataset kecil
    lrf=0.1,  # Final LR = lr0 * lrf (0.001 * 0.1 = 0.0001)
    warmup_epochs=5.0,  # Tingkatkan warmup
)
```

**Expected:** +0.5-1% mAP50 (stabil training)

---

### 4. **Lebih Banyak Epochs** ⭐⭐⭐⭐
```python
model.train(
    data='your_data.yaml',
    epochs=300,  # Naikkan dari 100 ke 200-300
    patience=50,  # Early stopping
    close_mosaic=10,  # Disable mosaic di 10 epochs terakhir
)
```

**Expected:** +0.5-1.5% mAP50 (lebih banyak belajar)

---

### 5. **Tune Loss Weights** ⭐⭐⭐⭐
```python
model.train(
    data='your_data.yaml',
    box=7.5,  # Box loss (default)
    cls=1.0,  # Naikkan dari 0.5 ke 1.0 jika banyak false positives
    dfl=1.5,  # DFL loss (default)
)
```

**Expected:** +0.3-0.8% mAP50 (kurangi FP)

---

## 🎯 Advanced: Pakai Model Architecture yang Lebih Bagus

### Option 1: Pakai mAP50-Boost Variant
```python
from ultralytics import YOLO

# Pakai model dengan Global Context + Large Kernel
model = YOLO('ultralytics/cfg/models/v12/yolov12-map50-boost.yaml')

model.train(
    data='your_data.yaml',
    epochs=300,
    imgsz=640,
    # Tambahkan P3-weighted loss juga!
    p3_weight=1.5,
    p4_weight=1.0,
    p5_weight=1.0,
)
```

**Expected:** +1-2% mAP50 dari baseline

---

### Option 2: Kombinasi Local Contrast + mAP50-Boost
Bisa modifikasi model YAML Anda untuk tambahkan Global Context Block:

```yaml
# Tambahkan di backbone setelah A2C2f
- [-1, 1, GlobalContextBlock, [512, 512, 4]] # P4
- [-1, 1, GlobalContextBlock, [1024, 1024, 4]] # P5

# Tambahkan di head P3 setelah Local Contrast Gate
- [-1, 1, GlobalContextBlock, [256, 256, 4]] # P3 head
```

---

## 📊 Training Configuration Lengkap (Recommended)

```python
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/v12/yolov12-ide1-local-contrast.yaml')

model.train(
    # Data & Model
    data='your_data.yaml',
    epochs=300,
    patience=50,
    
    # Image & Batch
    imgsz=832,  # Naikkan dari 640
    batch=16,  # Naikkan ke 32 jika GPU cukup
    
    # Optimizer & LR
    optimizer='AdamW',  # Biasanya lebih baik dari SGD
    lr0=0.001,  # Turunkan untuk dataset kecil
    lrf=0.1,
    warmup_epochs=5.0,
    weight_decay=0.0005,
    
    # Loss Weights
    box=7.5,
    cls=1.0,  # Naikkan jika banyak FP
    dfl=1.5,
    
    # P3-Weighted Loss (PENTING!)
    p3_weight=1.5,  # Fokus ke small objects
    p4_weight=1.0,
    p5_weight=1.0,
    
    # Augmentation
    hsv_h=0.015,  # Medical: kecil
    hsv_s=0.5,
    hsv_v=0.4,
    degrees=0.0,  # Hati-hati untuk medical!
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,  # Sangat efektif!
    mixup=0.1,
    
    # Training Strategy
    close_mosaic=10,  # Disable mosaic di 10 epochs terakhir
    multi_scale=False,  # True untuk multi-scale (lebih lambat)
    amp=True,  # Mixed precision
    cache=False,  # True untuk faster (butuh RAM)
    
    # Device
    device=0,
    workers=8,
)
```

---

## 🔍 Troubleshooting: mAP Stuck atau Turun?

### Problem: mAP Stuck di angka tertentu
**Solusi:**
1. ✅ Naikkan image size: 640 → 832 atau 1024
2. ✅ Tambah epochs: 100 → 200-300
3. ✅ Tune learning rate: coba lr0=0.0005 (lebih rendah)
4. ✅ Periksa data quality: pastikan annotation akurat
5. ✅ Tambah data: lebih banyak data = lebih baik

---

### Problem: Banyak False Positives
**Solusi:**
```python
model.train(
    cls=1.0,  # Naikkan dari 0.5 ke 1.0
    p3_weight=1.5,  # P3-weighted loss
    # Di inference:
    conf=0.3,  # Naikkan confidence threshold
    iou=0.7,  # Turunkan NMS threshold
)
```

---

### Problem: Banyak Miss Detection (Recall rendah)
**Solusi:**
```python
model.train(
    imgsz=1024,  # Naikkan image size
    p3_weight=2.0,  # Lebih agresif untuk small objects
    cls=0.5,  # Turunkan class loss (lebih permissive)
    # Di inference:
    conf=0.15,  # Turunkan confidence threshold
    max_det=500,  # Naikkan max detections
)
```

---

### Problem: Overfitting (Train mAP tinggi, Val mAP rendah)
**Solusi:**
```python
model.train(
    lr0=0.0005,  # Turunkan LR
    weight_decay=0.001,  # Naikkan weight decay
    mosaic=1.0,  # Pastikan augmentation aktif
    mixup=0.15,  # Naikkan mixup
    # Freeze backbone dulu:
    freeze=20,  # Freeze 20 layers pertama
    epochs=200,  # Kurangi epochs
)
```

---

## 📈 Priority Action Items

**Tingkat 1 (HIGH IMPACT - Lakukan DULU!):**
1. ✅ **P3 Weighted Loss** (p3_weight=1.5) ⭐⭐⭐⭐⭐
2. ✅ **Naikkan Image Size** (640 → 832/1024) ⭐⭐⭐⭐⭐
3. ✅ **Tune Learning Rate** (lr0=0.001 untuk dataset kecil) ⭐⭐⭐⭐⭐
4. ✅ **Lebih Banyak Epochs** (200-300 epochs) ⭐⭐⭐⭐

**Tingkat 2 (MEDIUM IMPACT):**
5. ✅ **Tune Loss Weights** (cls=1.0 jika banyak FP) ⭐⭐⭐⭐
6. ✅ **Naikkan Batch Size** (16 → 32 jika GPU cukup) ⭐⭐⭐
7. ✅ **Pakai Optimizer AdamW** ⭐⭐⭐

**Tingkat 3 (ADVANCED):**
8. ✅ **Pakai mAP50-Boost Architecture** ⭐⭐
9. ✅ **Multi-scale Training** ⭐⭐
10. ✅ **TTA (Test Time Augmentation)** ⭐

---

## 💡 Tips Khusus untuk Model Anda (Local Contrast Gate)

Karena Anda sudah pakai **Local Contrast Gate** di P3:

1. **Kombinasi dengan P3 Weighted Loss** → Double boost untuk small objects!
   ```python
   p3_weight=1.5  # Combine dengan Local Contrast Gate
   ```

2. **Tune Local Contrast Parameters** di YAML:
   ```yaml
   # Di model YAML Anda:
   - [-1, 1, LocalContrastGate, [256, 5, 0.7]]  # k=5, alpha=0.7
   
   # Coba naikkan alpha untuk lebih agresif:
   - [-1, 1, LocalContrastGate, [256, 5, 0.8]]  # alpha=0.8 (lebih kuat)
   ```

3. **Tambahkan Global Context Block** di head P3 juga:
   ```yaml
   - [-1, 1, LocalContrastGate, [256, 5, 0.7]]  # 15-Local Contrast
   - [-1, 1, GlobalContextBlock, [256, 256, 4]] # 16-GC Block (tambahkan ini!)
   ```

---

## 🎯 Expected Results

Dengan kombinasi semua tips di atas:
- **Baseline mAP50**: X%
- **+ P3 Weighted Loss**: +1-3%
- **+ Image Size 832**: +1-2%
- **+ Tuned LR**: +0.5-1%
- **+ More Epochs**: +0.5-1.5%
- **+ Loss Weights**: +0.3-0.8%
- **Total Expected**: +3-8% mAP50! 🚀

---

## ⚠️ Catatan Penting

1. **Jangan lakukan semua sekaligus!** → Coba satu per satu, monitor hasilnya
2. **Start dengan Quick Wins dulu** → P3 Weighted Loss + Image Size
3. **Monitor loss curves** → Pastikan tidak overfitting
4. **Validate hasil** → Bandingkan dengan baseline
5. **Dataset quality penting!** → Pastikan annotation akurat

---

**Good luck! Semoga mAP naik terus! 🚀📈**





























