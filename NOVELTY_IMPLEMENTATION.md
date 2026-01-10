# 📚 Implementasi Novelty untuk YOLOv12 P3

Dokumentasi implementasi tiga novelty untuk deteksi small object (BTA/bacilli) di P3.

---

## 🎯 Ringkasan Novelty

### 1. **FS-Neck (Frequency-Separated Neck)** ⭐ **PALING MUDAH**
- **Status**: ✅ **Siap Pakai** - Self-contained, tidak perlu modifikasi lain
- **File**: `ultralytics/nn/modules/block.py` (class `FSNeck`)
- **Config**: `ultralytics/cfg/models/v12/yolov12-p3-fsneck.yaml`

**Mengapa paling mudah?**
- Self-contained block, tidak perlu akses layer lain
- Langsung bisa dipakai di YAML config
- Tidak perlu modifikasi Detect head atau komponen lain

**Cara Pakai:**
```yaml
- [[-1, 4], 1, Concat, [1]]   # cat backbone P3
- [-1, 1, FSNeck, [256, 7]]   # FS-Neck: kernel_size=7
- [-1, 2, A2C2f, [256, False, -1]]
```

**Parameter:**
- `c1`: Input channels (256 untuk P3)
- `c2`: Output channels (default: sama dengan input)
- `kernel_size`: Kernel size untuk low-pass filter (default: 7, bisa 5)
- `use_avgpool`: Pakai AvgPool atau DWConv untuk low-pass (default: False)

---

### 2. **DI-Fuse (Detail Injection Fuse)** ⭐⭐ **SEDANG**
- **Status**: ✅ **Siap Pakai** - Perlu akses P2 dari backbone
- **File**: `ultralytics/nn/modules/block.py` (class `DIFuse`)
- **Config**: `ultralytics/cfg/models/v12/yolov12-p3-difuse.yaml`

**Mengapa sedang?**
- Perlu akses P2 feature dari backbone (layer 2)
- Harus pastikan index layer P2 benar di YAML
- Tapi tetap self-contained sebagai block

**Cara Pakai:**
```yaml
- [[-1, 4], 1, Concat, [1]]   # cat backbone P3
- [-1, 2, A2C2f, [256, False, -1]]  # P3 neck output (layer 14)
- [[14, 2], 1, DIFuse, [256, 256]]  # DI-Fuse: [P3_neck, P2_backbone]
```

**Parameter:**
- `c_p2`: P2 channels (256)
- `c_p3`: P3 channels (256)
- `c_out`: Output channels (default: sama dengan P3)
- `alpha_init`: Initial value untuk learnable alpha (default: 0.1)
- `gate_from_p3`: Compute gate dari P3 atau P2toP3 (default: True)

**Catatan:**
- Pastikan index layer P2 backbone benar (biasanya layer 2)
- DI-Fuse menerima list `[P3, P2]` sebagai input

---

### 3. **SAD-Head (Scale-Aware Dynamic Head)** ⭐⭐⭐ **PALING SUSAH**
- **Status**: ⚠️ **Butuh Modifikasi Detect Head**
- **File**: `ultralytics/nn/modules/block.py` (class `SADHead`)
- **Config**: `ultralytics/cfg/models/v12/yolov12-p3-sadhead.yaml`

**Mengapa paling susah?**
- Perlu modifikasi `Detect` head untuk apply penalty ke cls logits
- SAD-Head return tuple `(features, penalty)` yang perlu di-handle
- Perlu custom integration di forward pass Detect head

**Cara Pakai (Current - Butuh Modifikasi):**
```yaml
- [-1, 2, A2C2f, [256, False, -1]]  # P3 neck
- [-1, 1, SADHead, [256]]  # SAD-Head (returns features, penalty)
# TODO: Modify Detect head to use penalty for P3 cls logits
```

**Yang Perlu Dilakukan:**
1. Modifikasi `Detect` head di `ultralytics/nn/modules/head.py`
2. Store penalty dari SAD-Head untuk P3
3. Apply `apply_penalty_to_cls()` method sebelum sigmoid di cls branch
4. Hanya apply untuk P3, bukan P4/P5

**Parameter:**
- `c1`: Input channels (256 untuk P3)
- `reg_max`: DFL reg_max (default: 16)
- `beta_init`: Initial value untuk learnable beta penalty (default: 0.5)

---

## 🚀 Quick Start

### FS-Neck (Paling Mudah - Recommended untuk Start)
```bash
# Train dengan FS-Neck
python train.py model=yolov12n-p3-fsneck.yaml data=your_dataset.yaml epochs=100
```

### DI-Fuse
```bash
# Train dengan DI-Fuse
python train.py model=yolov12n-p3-difuse.yaml data=your_dataset.yaml epochs=100
```

### Combo FS-Neck + DI-Fuse
```bash
# Train dengan kombinasi FS-Neck + DI-Fuse
python train.py model=yolov12n-p3-combo-novelties.yaml data=your_dataset.yaml epochs=100
```

---

## 📊 Perbandingan

| Novelty | Kesulitan | Self-Contained | Butuh Modifikasi | Siap Pakai |
|---------|-----------|----------------|------------------|------------|
| **FS-Neck** | ⭐ Mudah | ✅ Ya | ❌ Tidak | ✅ Ya |
| **DI-Fuse** | ⭐⭐ Sedang | ✅ Ya | ❌ Tidak | ✅ Ya |
| **SAD-Head** | ⭐⭐⭐ Susah | ❌ Tidak | ✅ Ya (Detect head) | ⚠️ Partial |

---

## 🔧 Implementasi Detail

### FS-Neck Architecture
```
Input (P3 features)
  ↓
Low-pass: AvgPool/DWConv(k=7) → low
  ↓
High-pass: x - low → high
  ↓
Process high: Conv(high) → high_processed
  ↓
Fusion: Conv([x, high_processed]) → output
```

### DI-Fuse Architecture
```
Input: [P3, P2]
  ↓
P2 → Downsample(stride=2) → P2toP3
  ↓
Gate: sigmoid(Conv(P3)) → gate
  ↓
Fusion: P3 + α * (gate * P2toP3) → output
```

### SAD-Head Architecture
```
Input (P3 features)
  ↓
Size Proxy: Conv(features) → size_proxy
  ↓
Penalty: sigmoid(Conv(size_proxy)) → penalty
  ↓
Output: (features, penalty)
  ↓
[Di Detect Head] Apply: cls_logits * (1 - β * penalty)
```

---

## 📝 Catatan Penting

1. **FS-Neck**: Paling mudah, langsung pakai, recommended untuk eksperimen pertama
2. **DI-Fuse**: Perlu pastikan index P2 backbone benar (cek di YAML)
3. **SAD-Head**: Butuh modifikasi Detect head - belum fully integrated, perlu custom code

---

## 🎯 Rekomendasi

**Untuk Start:**
1. ✅ Pakai **FS-Neck** dulu (paling mudah, langsung bisa)
2. ✅ Kalau perlu detail lebih, tambah **DI-Fuse**
3. ⚠️ **SAD-Head** tunggu dulu sampai ada waktu untuk modifikasi Detect head

**Klaim Novelty:**
- FS-Neck: "frequency separation to suppress stain/background dominance in microscopic small-object detection"
- DI-Fuse: "gated detail injection from higher-resolution stage to improve tiny bacilli localization while controlling noise"
- SAD-Head: "tiny-object prior integrated into detection head for bacilli-scale bounding boxes"

---

**Dokumentasi dibuat untuk YOLOv12 Novelty Implementation v1.0**










