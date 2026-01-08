# 🔧 SFAN Error Fix - NotImplementedError

## ❌ Error yang Terjadi

```
NotImplementedError: WARNING ⚠️ 'YOLO' model does not support '_new' mode for 'None' task yet.
```

## 🔍 Root Cause

Error ini terjadi karena **modul-modul baru di YAML belum diimplement**:
- `NoiseSuppression` - belum ada
- `AdaptiveFeatureFusion` - belum ada  
- `SmallObjectEnhancementHead` - belum ada

Parser YAML tidak bisa menemukan modul-modul ini, sehingga gagal saat parsing.

---

## ✅ Solusi

### **Option 1: Pakai Workaround Version (RECOMMENDED untuk test dulu)**

Gunakan `yolov12-sfan-workaround.yaml` yang menggunakan modul **yang sudah ada**:

```python
from ultralytics import YOLO

# Pakai workaround version yang bisa langsung dipakai
model = YOLO('ultralytics/cfg/models/v12/yolov12-sfan-workaround.yaml')
model.train(data='your_data.yaml', epochs=100, imgsz=640)
```

**Workaround mapping:**
- `NoiseSuppression` → `ChannelAttention + SpatialAttention` (CBAM-like)
- `AdaptiveFeatureFusion` → `Concat` biasa (temporary)
- `SmallObjectEnhancementHead` → `Detect` biasa + `SOE` (temporary)

**Keuntungan:**
- ✅ Bisa langsung di-parse dan di-train
- ✅ Mendapat sebagian benefit (noise suppression via attention)
- ✅ Bisa test konsep SFAN dulu

**Kekurangan:**
- ⚠️ Belum full SFAN functionality
- ⚠️ AFF masih pakai Concat biasa (belum learnable weights)
- ⚠️ SOEH masih pakai Detect biasa (belum auxiliary loss)

---

### **Option 2: Implement Modul Baru Dulu**

Ikuti guide di `SFAN_IMPLEMENTATION_GUIDE.md` untuk implement 3 modul baru:

1. **NoiseSuppressionBlock** - `ultralytics/nn/modules/block.py`
2. **AdaptiveFeatureFusion** - `ultralytics/nn/modules/block.py`
3. **SmallObjectEnhancementHead** - `ultralytics/nn/modules/head.py`

Setelah implement, register di:
- `ultralytics/nn/modules/__init__.py`
- `ultralytics/nn/tasks.py` (parsing logic)

Lalu pakai `yolov12-sfan.yaml` atau `yolov12-sfan-simple.yaml`.

---

## 📋 File yang Tersedia

1. **`yolov12-sfan.yaml`** - Full SFAN (perlu implement modul baru)
2. **`yolov12-sfan-simple.yaml`** - Simple SFAN (perlu implement modul baru)
3. **`yolov12-sfan-workaround.yaml`** - Workaround (bisa langsung dipakai) ⭐

---

## 🚀 Quick Start

### Test dengan Workaround:
```python
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/v12/yolov12-sfan-workaround.yaml')
model.train(
    data='your_data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

### Setelah Implement Modul Baru:
```python
from ultralytics import YOLO

# Full SFAN
model = YOLO('ultralytics/cfg/models/v12/yolov12-sfan.yaml')
# atau Simple SFAN
model = YOLO('ultralytics/cfg/models/v12/yolov12-sfan-simple.yaml')

model.train(data='your_data.yaml', epochs=100, imgsz=640)
```

---

## 📝 Catatan

- **Workaround version** sudah bisa dipakai untuk test konsep
- **Full SFAN** perlu implement modul baru dulu (lihat `SFAN_IMPLEMENTATION_GUIDE.md`)
- **Simple SFAN** lebih cepat implement (hanya 2 modul: NSB + SOEH)

**Good luck! 🚀**

