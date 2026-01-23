# 🔍 Apakah `nc` di YAML Mempengaruhi mAP50?

## ❓ Pertanyaan

**"Kelas saya cuma 1, tapi `nc=80` di YAML, ngaruh ke mAP?"**

## ✅ Jawaban Singkat

**TIDAK, `nc` di YAML TIDAK mempengaruhi mAP50!**

mAP50 hanya menghitung kelas yang **benar-benar ada di data**, bukan dari YAML.

---

## 🔍 Penjelasan Detail

### 1. **Bagaimana mAP50 Dihitung?**

Dari kode `ultralytics/utils/metrics.py`:

```python
def ap_per_class(tp, conf, pred_cls, target_cls, ...):
    # 1. Cari unique classes yang ada di DATA (bukan dari YAML!)
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # Jumlah kelas yang ada di data
    
    # 2. Hanya hitung AP untuk kelas yang ada di data
    for ci, c in enumerate(unique_classes):
        if n_p == 0 or n_l == 0:  # Jika tidak ada prediction atau label
            continue  # SKIP kelas ini!
        
        # Hitung AP untuk kelas ini
        ap[ci, j] = compute_ap(recall, precision)
    
    return ap  # Hanya berisi AP untuk kelas yang ada data
```

**Kesimpulan:**
- ✅ mAP50 hanya menghitung kelas yang **ada di ground truth data**
- ✅ Kelas yang tidak ada data akan **di-skip** (tidak dihitung)
- ✅ `nc` di YAML **TIDAK digunakan** untuk perhitungan mAP50

### 2. **Contoh Praktis**

**Skenario:**
- YAML: `nc: 80` (COCO dataset)
- Data Anda: Hanya 1 kelas (misalnya "person")

**Perhitungan mAP50:**
```
1. Sistem cek unique classes di data → [0] (hanya class 0)
2. Hanya hitung AP untuk class 0
3. mAP50 = AP_class0 (bukan mean dari 80 kelas!)
```

**Hasil:**
- ✅ mAP50 = AP untuk kelas yang ada saja
- ✅ Tidak ada pengaruh dari `nc=80` di YAML
- ✅ Kelas 1-79 yang tidak ada data **tidak dihitung**

---

## ⚠️ Tapi `nc` di YAML Tetap Penting!

Meskipun tidak mempengaruhi mAP50, `nc` di YAML tetap penting untuk:

### 1. **Ukuran Output Layer**

Dari `ultralytics/nn/modules/head.py`:

```python
class Detect(nn.Module):
    def __init__(self, nc=80, ch=()):
        self.nc = nc  # number of classes
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        
        # Output layer untuk classification
        self.cv3 = nn.ModuleList(
            nn.Conv2d(c3, self.nc, 1)  # Output: nc channels
            for x in ch
        )
```

**Impact:**
- Jika `nc=80` → Output layer punya **80 channels** (untuk 80 kelas)
- Jika `nc=1` → Output layer punya **1 channel** (untuk 1 kelas)
- **Memory usage**: `nc=80` lebih boros memory daripada `nc=1`

### 2. **Model Architecture**

```yaml
# YAML
nc: 80  # atau 1

head:
  - [[15, 18, 21], 1, Detect, [nc]]  # Detect head menggunakan nc
```

**Impact:**
- Ukuran model berbeda (nc=80 lebih besar)
- Parameter count berbeda
- Inference speed sedikit berbeda

### 3. **Training Loss**

Loss function menggunakan `nc` untuk:
- Classification loss: `nc` output channels
- Class weights: Jika menggunakan class weights

---

## 📊 Perbandingan: `nc=80` vs `nc=1`

### **Untuk Dataset 1 Kelas:**

| Aspek | `nc=80` | `nc=1` | Impact |
|-------|---------|--------|--------|
| **mAP50** | ✅ Sama | ✅ Sama | ❌ Tidak ada perbedaan |
| **Memory** | ⚠️ Lebih besar | ✅ Lebih kecil | ⚠️ Boros memory |
| **Model Size** | ⚠️ Lebih besar | ✅ Lebih kecil | ⚠️ ~79 channels tidak terpakai |
| **Training Speed** | ⚠️ Sedikit lebih lambat | ✅ Sedikit lebih cepat | ⚠️ Minimal |
| **Inference Speed** | ⚠️ Sedikit lebih lambat | ✅ Sedikit lebih cepat | ⚠️ Minimal |

---

## ✅ Rekomendasi

### **Untuk Dataset 1 Kelas:**

**Option 1: Set `nc=1` di YAML** (RECOMMENDED)
```yaml
nc: 1  # Sesuai dengan dataset Anda
```

**Keuntungan:**
- ✅ Model lebih kecil
- ✅ Memory lebih efisien
- ✅ Training/inference sedikit lebih cepat
- ✅ Tidak ada channel yang tidak terpakai

**Option 2: Tetap `nc=80`** (Jika mau fleksibel)
```yaml
nc: 80  # Tetap 80
```

**Keuntungan:**
- ✅ Bisa digunakan untuk dataset multi-class di masa depan
- ✅ Compatible dengan pretrained weights COCO

**Kekurangan:**
- ⚠️ Boros memory (79 channels tidak terpakai)
- ⚠️ Model sedikit lebih besar

---

## 🔍 Verifikasi: Apakah `nc` Mempengaruhi mAP50?

### **Test Sederhana:**

```python
from ultralytics import YOLO

# Test 1: nc=1
model1 = YOLO('yolov12.yaml')  # dengan nc=1
results1 = model1.val(data='your_data.yaml')
print(f"mAP50 (nc=1): {results1.box.map50}")

# Test 2: nc=80 (ubah di YAML)
model2 = YOLO('yolov12.yaml')  # dengan nc=80
results2 = model2.val(data='your_data.yaml')
print(f"mAP50 (nc=80): {results2.box.map50}")

# Hasil: mAP50 akan SAMA! ✅
```

---

## 📝 Kesimpulan

### **Untuk mAP50:**
- ❌ **TIDAK**, `nc` di YAML tidak mempengaruhi mAP50
- ✅ mAP50 hanya menghitung kelas yang ada di data
- ✅ Kelas yang tidak ada data akan di-skip

### **Untuk Model:**
- ⚠️ **YA**, `nc` mempengaruhi ukuran model dan memory
- ⚠️ `nc=80` lebih boros memory daripada `nc=1`
- ✅ Rekomendasi: Set `nc=1` jika dataset Anda hanya 1 kelas

### **Best Practice:**
```yaml
# Set nc sesuai dengan jumlah kelas di dataset Anda
nc: 1  # Jika dataset hanya 1 kelas
```

---

## 💡 Tips

1. ✅ **Selalu set `nc` sesuai dataset** untuk efisiensi
2. ✅ **mAP50 tidak terpengaruh** oleh `nc` di YAML
3. ✅ **Jika ragu**, cek jumlah kelas di dataset YAML:
   ```yaml
   # dataset.yaml
   names:
     0: person  # Hanya 1 kelas
   ```
   Maka set `nc: 1` di model YAML

---

**Referensi Kode:**
- `ultralytics/utils/metrics.py`: `ap_per_class()` - line 574-575
- `ultralytics/nn/modules/head.py`: `Detect.__init__()` - line 34-40

















