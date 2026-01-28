# 📊 Detail: Bagaimana Precision & Recall Mempengaruhi mAP50

## 🎯 Overview

**mAP50** dihitung dari **Precision-Recall (PR) curve** yang dibuat dari hasil matching antara **prediksi** dan **ground truth** menggunakan **IoU threshold 0.5**.

---

## 📐 Step 1: Menghitung IoU (Intersection over Union)

### Formula IoU:
```
IoU = (Area of Intersection) / (Area of Union)

dimana:
- Intersection = area yang overlap antara predicted box dan ground truth box
- Union = total area dari kedua box dikurangi intersection
```

### Contoh Visual:
```
Ground Truth Box:  [x1=10, y1=10, x2=50, y2=50]  (area = 1600)
Predicted Box:     [x1=15, y1=15, x2=55, y2=55]  (area = 1600)

Intersection:      [x1=15, y1=15, x2=50, y2=50]  (area = 1225)
Union:             (1600 + 1600 - 1225) = 1975

IoU = 1225 / 1975 = 0.62
```

**Kode dari `ultralytics/utils/metrics.py`:**
```python
def box_iou(box1, box2, eps=1e-7):
    # Calculate intersection
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)
    # Calculate union
    union = (a2 - a1).prod(2) + (b2 - b1).prod(2) - inter
    return inter / (union + eps)
```

---

## ✅ Step 2: Menentukan TP, FP, FN (untuk IoU threshold 0.5)

### Proses Matching (dari `ultralytics/engine/validator.py`):

```python
def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
    """
    Matches predictions to ground truth menggunakan IoU threshold.
    
    Untuk mAP50, threshold = 0.5
    """
    # 1. Cek apakah class cocok
    correct_class = true_classes[:, None] == pred_classes
    
    # 2. Zero out wrong classes (hanya hitung IoU jika class sama)
    iou = iou * correct_class
    
    # 3. Untuk mAP50, threshold = 0.5
    threshold = 0.5
    matches = np.nonzero(iou >= threshold)  # IoU >= 0.5 dan class match
    
    # 4. Satu ground truth hanya bisa match dengan satu prediction (highest IoU)
    # 5. Satu prediction hanya bisa match dengan satu ground truth
```

### Kategori Deteksi:

#### **True Positive (TP)** ✅
- **Kondisi**: 
  - IoU ≥ 0.5 **DAN**
  - Predicted class = Ground truth class **DAN**
  - Ground truth belum di-match dengan prediction lain
- **Artinya**: Deteksi **BENAR** - model mendeteksi objek yang ada dengan benar

**Contoh:**
```
Ground Truth:  [x1=10, y1=10, x2=50, y2=50, class=0]  (person)
Prediction:    [x1=12, y1=12, x2=52, y2=52, class=0, conf=0.9]
IoU = 0.65 → TP ✅ (karena IoU > 0.5 dan class sama)
```

#### **False Positive (FP)** ❌
- **Kondisi**:
  - IoU < 0.5 **ATAU**
  - Predicted class ≠ Ground truth class **ATAU**
  - Tidak ada ground truth yang cocok (deteksi objek yang tidak ada)
- **Artinya**: Deteksi **SALAH** - model mendeteksi sesuatu yang tidak benar

**Contoh 1: IoU terlalu rendah**
```
Ground Truth:  [x1=10, y1=10, x2=50, y2=50, class=0]
Prediction:    [x1=60, y1=60, x2=100, y2=100, class=0, conf=0.8]
IoU = 0.0 → FP ❌ (karena IoU < 0.5)
```

**Contoh 2: Class salah**
```
Ground Truth:  [x1=10, y1=10, x2=50, y2=50, class=0]  (person)
Prediction:    [x1=12, y1=12, x2=52, y2=52, class=1, conf=0.9]  (car)
IoU = 0.65 → FP ❌ (karena class berbeda, meskipun IoU tinggi)
```

**Contoh 3: Deteksi objek yang tidak ada**
```
Ground Truth:  [x1=10, y1=10, x2=50, y2=50, class=0]  (hanya 1 objek)
Prediction 1:  [x1=12, y1=12, x2=52, y2=52, class=0, conf=0.9]  → TP ✅
Prediction 2:  [x1=200, y1=200, x2=250, y2=250, class=0, conf=0.7]  → FP ❌
                (tidak ada ground truth di lokasi ini)
```

#### **False Negative (FN)** ❌
- **Kondisi**: Ground truth **TIDAK** di-match dengan prediction manapun
- **Artinya**: Objek yang **ADA** tapi **TIDAK TERDETEKSI** oleh model

**Contoh:**
```
Ground Truth 1:  [x1=10, y1=10, x2=50, y2=50, class=0]  → Match dengan Prediction 1 → TP ✅
Ground Truth 2:  [x1=100, y1=100, x2=150, y2=150, class=0]  → TIDAK ada prediction → FN ❌
```

---

## 📊 Step 3: Menghitung Precision & Recall

### Formula:

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```

### Contoh Perhitungan:

**Skenario:**
- Total Ground Truth: 10 objek
- Total Predictions: 12 deteksi
- TP = 7, FP = 5, FN = 3

**Perhitungan:**
```
Precision = 7 / (7 + 5) = 7/12 = 0.583 (58.3%)
Recall    = 7 / (7 + 3) = 7/10 = 0.700 (70.0%)
```

**Interpretasi:**
- **Precision 58.3%**: Dari 12 deteksi, hanya 7 yang benar (5 salah)
- **Recall 70%**: Dari 10 objek yang ada, model berhasil mendeteksi 7 (3 terlewat)

---

## 📈 Step 4: Membuat Precision-Recall Curve

### Proses (dari `ultralytics/utils/metrics.py`):

```python
def ap_per_class(tp, conf, pred_cls, target_cls, ...):
    """
    Membuat PR curve dengan variasi confidence threshold.
    """
    # 1. Sort predictions berdasarkan confidence (dari tinggi ke rendah)
    i = np.argsort(-conf)  # Sort descending
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    
    # 2. Untuk setiap confidence threshold (0.0 sampai 1.0, step 0.001):
    x = np.linspace(0, 1, 1000)  # 1000 titik
    
    for threshold in x:  # threshold = 0.0, 0.001, 0.002, ..., 1.0
        # Filter predictions dengan confidence >= threshold
        valid = conf >= threshold
        
        # Hitung TP dan FP untuk threshold ini
        tpc = tp[valid].cumsum(0)  # Cumulative TP
        fpc = (1 - tp[valid]).cumsum(0)  # Cumulative FP
        
        # Hitung Precision & Recall
        precision = tpc / (tpc + fpc)
        recall = tpc / (n_labels + eps)  # n_labels = total ground truth
        
        # Simpan ke curve
        p_curve.append(precision)
        r_curve.append(recall)
```

### Contoh PR Curve:

**Data:**
```
Predictions (sorted by confidence):
1. [conf=0.95, TP=True]   → TP=1, FP=0, Precision=1.0, Recall=0.1 (1/10)
2. [conf=0.90, TP=True]   → TP=2, FP=0, Precision=1.0, Recall=0.2 (2/10)
3. [conf=0.85, TP=True]   → TP=3, FP=0, Precision=1.0, Recall=0.3 (3/10)
4. [conf=0.80, TP=False]  → TP=3, FP=1, Precision=0.75, Recall=0.3 (3/10)
5. [conf=0.75, TP=True]   → TP=4, FP=1, Precision=0.80, Recall=0.4 (4/10)
6. [conf=0.70, TP=False]  → TP=4, FP=2, Precision=0.67, Recall=0.4 (4/10)
7. [conf=0.65, TP=True]   → TP=5, FP=2, Precision=0.71, Recall=0.5 (5/10)
...
```

**PR Curve:**
```
Precision
  1.0 |     ●─────●─────●
      |              ╲
  0.8 |               ●───●
      |                    ╲
  0.6 |                     ●───●
      |                          ╲
  0.4 |                           ●───●───●───●
      |
  0.2 |
      |
  0.0 └─────────────────────────────────────────── Recall
      0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0
```

**Karakteristik PR Curve:**
- **X-axis**: Recall (0.0 - 1.0)
- **Y-axis**: Precision (0.0 - 1.0)
- **Bentuk**: Biasanya menurun dari kiri atas ke kanan bawah
- **Area di bawah curve**: Semakin besar = semakin baik

---

## 🎯 Step 5: Menghitung AP (Average Precision)

### Formula (dari `ultralytics/utils/metrics.py`):

```python
def compute_ap(recall, precision):
    """
    Menghitung AP = Area di bawah Precision-Recall curve.
    
    Metode: Interpolasi 101-point (COCO standard)
    """
    # 1. Tambahkan sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    
    # 2. Compute precision envelope (monotonic decreasing)
    # Setiap titik precision = max precision untuk recall >= titik tersebut
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    
    # 3. Interpolasi ke 101 titik (0.0, 0.01, 0.02, ..., 1.0)
    x = np.linspace(0, 1, 101)
    
    # 4. Hitung area di bawah curve (trapezoidal integration)
    ap = np.trapz(np.interp(x, mrec, mpre), x)
    
    return ap
```

### Contoh Perhitungan AP:

**PR Curve Data:**
```
Recall:    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
Precision: [1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
```

**Precision Envelope (monotonic):**
```
Recall:    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
Precision: [1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
            (sudah monotonic, tidak perlu diubah)
```

**Interpolasi ke 101 titik:**
```
x = [0.00, 0.01, 0.02, ..., 0.99, 1.00]
y = [interpolated precision values]
```

**AP = Area di bawah curve:**
```
AP = ∫(precision d(recall)) dari 0.0 sampai 1.0
   = 0.85 (contoh)
```

**Interpretasi:**
- **AP = 0.85**: Rata-rata precision di semua level recall adalah 85%
- **AP lebih tinggi** = Model lebih baik (lebih banyak area di bawah curve)

---

## 🎯 Step 6: Menghitung mAP50

### Formula:

```python
# Untuk setiap kelas, hitung AP@0.5
ap50_per_class = [AP_class0, AP_class1, AP_class2, ...]

# mAP50 = Rata-rata AP@0.5 untuk semua kelas
mAP50 = mean(ap50_per_class)
```

### Contoh:

**Dataset dengan 3 kelas:**
- **Class 0 (person)**: AP@0.5 = 0.85
- **Class 1 (car)**: AP@0.5 = 0.78
- **Class 2 (bike)**: AP@0.5 = 0.92

**mAP50:**
```
mAP50 = (0.85 + 0.78 + 0.92) / 3 = 0.85
```

**Kode dari `ultralytics/utils/metrics.py`:**
```python
@property
def map50(self):
    """
    Returns the mean Average Precision (mAP) at an IoU threshold of 0.5.
    """
    return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0
    # all_ap[:, 0] = AP@0.5 untuk semua kelas
    # .mean() = rata-rata
```

---

## 🔍 Bagaimana FP & FN Mempengaruhi mAP50?

### **False Positives (FP) → Menurunkan Precision → Menurunkan mAP50**

**Mekanisme:**
```
FP naik → Precision turun → PR curve turun → AP turun → mAP50 turun
```

**Contoh:**
```
Skenario A (FP rendah):
- TP=8, FP=2 → Precision = 8/10 = 0.80
- AP = 0.82

Skenario B (FP tinggi):
- TP=8, FP=8 → Precision = 8/16 = 0.50
- AP = 0.65

mAP50 turun dari 0.82 ke 0.65 karena FP naik!
```

**Cara Mengurangi FP:**
1. ✅ Tingkatkan `cls` loss weight (1.0 → 1.5)
2. ✅ Gunakan architecture yang mengurangi FP (FBSB, Texture Punish)
3. ✅ Naikkan NMS threshold (iou: 0.6 → 0.7)
4. ✅ Perbaiki kualitas data (kurangi mislabeling)

---

### **False Negatives (FN) → Menurunkan Recall → Menurunkan mAP50**

**Mekanisme:**
```
FN naik → Recall turun → PR curve tidak mencapai recall tinggi → AP turun → mAP50 turun
```

**Contoh:**
```
Skenario A (FN rendah):
- TP=9, FN=1 → Recall = 9/10 = 0.90
- PR curve mencapai recall tinggi → AP = 0.85

Skenario B (FN tinggi):
- TP=6, FN=4 → Recall = 6/10 = 0.60
- PR curve hanya sampai recall 0.60 → AP = 0.72

mAP50 turun dari 0.85 ke 0.72 karena FN naik!
```

**Cara Mengurangi FN:**
1. ✅ Naikkan image size (640 → 832 atau 1024)
2. ✅ Gunakan P3-weighted loss untuk small objects
3. ✅ Turunkan confidence threshold saat validation
4. ✅ Gunakan architecture untuk small objects (SOE, SPDConv)
5. ✅ Tambah lebih banyak data

---

## 📊 Contoh Lengkap: Dari Deteksi ke mAP50

### **Input:**
```
Ground Truth (10 objek):
- GT1: [10, 10, 50, 50, class=0]  (person)
- GT2: [100, 100, 150, 150, class=0]  (person)
- GT3: [200, 200, 250, 250, class=1]  (car)
- ... (7 objek lainnya)

Predictions (12 deteksi):
- P1: [12, 12, 52, 52, class=0, conf=0.95]  → IoU=0.65 → TP ✅
- P2: [102, 102, 152, 152, class=0, conf=0.90]  → IoU=0.70 → TP ✅
- P3: [60, 60, 100, 100, class=0, conf=0.85]  → IoU=0.20 → FP ❌
- P4: [202, 202, 252, 252, class=1, conf=0.80]  → IoU=0.75 → TP ✅
- ... (8 deteksi lainnya)
```

### **Step 1: Matching (IoU threshold = 0.5)**
```
P1 ↔ GT1: IoU=0.65 → TP ✅
P2 ↔ GT2: IoU=0.70 → TP ✅
P3 ↔ (no match): IoU=0.20 < 0.5 → FP ❌
P4 ↔ GT3: IoU=0.75 → TP ✅
...
Result: TP=7, FP=5, FN=3
```

### **Step 2: Hitung Precision & Recall**
```
Precision = 7 / (7 + 5) = 0.583
Recall    = 7 / (7 + 3) = 0.700
```

### **Step 3: Buat PR Curve (variasi confidence threshold)**
```
Conf Threshold | TP | FP | Precision | Recall
---------------|----|----|-----------|-------
0.95          | 1  | 0  | 1.00      | 0.10
0.90          | 2  | 0  | 1.00      | 0.20
0.85          | 2  | 1  | 0.67      | 0.20
0.80          | 3  | 1  | 0.75      | 0.30
0.75          | 4  | 2  | 0.67      | 0.40
0.70          | 5  | 3  | 0.63      | 0.50
0.65          | 6  | 4  | 0.60      | 0.60
0.60          | 7  | 5  | 0.58      | 0.70
0.55          | 7  | 6  | 0.54      | 0.70
...
```

### **Step 4: Hitung AP@0.5**
```
AP@0.5 = Area di bawah PR curve = 0.72
```

### **Step 5: Hitung mAP50 (jika ada multiple classes)**
```
Class 0 (person): AP@0.5 = 0.72
Class 1 (car):    AP@0.5 = 0.78
Class 2 (bike):   AP@0.5 = 0.65

mAP50 = (0.72 + 0.78 + 0.65) / 3 = 0.717
```

---

## 💡 Kesimpulan

### **Rantai Pengaruh:**

```
FP naik → Precision turun → PR curve turun → AP turun → mAP50 turun
FN naik → Recall turun → PR curve tidak mencapai recall tinggi → AP turun → mAP50 turun
```

### **Faktor yang Mempengaruhi FP:**
- ❌ Class misclassification
- ❌ Background noise terdeteksi sebagai objek
- ❌ IoU terlalu rendah (< 0.5)
- ❌ Deteksi objek yang tidak ada

### **Faktor yang Mempengaruhi FN:**
- ❌ Objek terlalu kecil tidak terdeteksi
- ❌ Objek terhalang/occluded
- ❌ Confidence terlalu rendah
- ❌ Model tidak cukup baik untuk mendeteksi objek tertentu

### **Cara Meningkatkan mAP50:**
1. ✅ **Kurangi FP**: Tingkatkan precision dengan loss tuning, architecture, data quality
2. ✅ **Kurangi FN**: Tingkatkan recall dengan image size, P3-weighted loss, architecture untuk small objects
3. ✅ **Balance Precision & Recall**: Jangan fokus hanya pada satu metrik, keduanya penting!

---

**Referensi Kode:**
- `ultralytics/utils/metrics.py`: `ap_per_class()`, `compute_ap()`, `map50`
- `ultralytics/engine/validator.py`: `match_predictions()`






























