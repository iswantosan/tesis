# 🚀 C3k2AttnV2 - Enhanced Dual Attention Module

## 📖 Deskripsi

`C3k2AttnV2` adalah versi enhanced dari `C3k2Attn` dengan **dual attention mechanism** untuk feature extraction yang lebih baik.

### ✨ Fitur Utama

1. **Pre-Attention (Optional)**: ECA attention pada input untuk feature selection
2. **Dual Post-Attention**: 
   - ECA (Efficient Channel Attention) untuk channel-wise attention
   - CoordinateAttention untuk spatial-channel attention
3. **Better Feature Extraction**: Kombinasi attention mechanisms untuk enhanced representation learning

---

## 🔧 Cara Pakai

### 1. Di YAML Config File

#### Format Minimal (Default Settings)
```yaml
- [-1, 2, C3k2AttnV2, [256]]  # c2=256, semua default
```

#### Format Lengkap
```yaml
- [-1, 2, C3k2AttnV2, [256, False, 0.5, 1, True, True]]
#                    [c2, c3k, e, g, shortcut, use_pre_attn]
```

#### Parameter:
- `c2` (int, **required**): Output channels
- `c3k` (bool, default: `False`): Use C3k blocks instead of Bottleneck
- `e` (float, default: `0.5`): Expansion ratio
- `g` (int, default: `1`): Groups for convolution
- `shortcut` (bool, default: `True`): Use shortcut connection
- `use_pre_attn` (bool, default: `True`): Enable pre-attention on input

#### Contoh di Backbone:
```yaml
backbone:
  - [-1, 1, Conv,  [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]] # 1-P2/4
  - [-1, 2, C3k2AttnV2,  [256, False, 0.25]] # 2 - Enhanced attention
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]] # 3-P3/8
  - [-1, 2, C3k2AttnV2,  [512, False, 0.25]] # 4 - Enhanced attention
```

#### Contoh di Head:
```yaml
head:
  - [-1, 1, Conv, [512, 3, 2]] # 18
  - [[-1, 8], 1, Concat, [1]] # 19 cat head P5
  - [-1, 2, C3k2AttnV2, [1024, True]] # 20 - Enhanced attention di P5
```

---

### 2. Di Python Code

#### Import Module
```python
from ultralytics.nn.modules import C3k2AttnV2
import torch

# Atau dari YOLO model
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/v12/yolov12-c3k2attnv2-example.yaml')
```

#### Direct Usage
```python
import torch
from ultralytics.nn.modules import C3k2AttnV2

# Create module
module = C3k2AttnV2(
    c1=128,           # Input channels
    c2=256,           # Output channels
    n=2,              # Number of bottleneck blocks
    c3k=False,        # Use C3k blocks?
    e=0.5,            # Expansion ratio
    g=1,              # Groups
    shortcut=True,    # Shortcut connection
    use_pre_attn=True # Pre-attention
)

# Forward pass
x = torch.randn(1, 128, 64, 64)  # [B, C, H, W]
out = module(x)  # [1, 256, 64, 64]
```

---

### 3. Training dengan C3k2AttnV2

```python
from ultralytics import YOLO

# Load model config dengan C3k2AttnV2
model = YOLO('ultralytics/cfg/models/v12/yolov12-c3k2attnv2-example.yaml')

# Train
model.train(
    data='path/to/your/data.yaml',
    epochs=200,
    imgsz=640,
    batch=16,
    device=0,  # GPU
    project='runs/detect',
    name='yolov12-c3k2attnv2'
)
```

---

## 📊 Perbandingan: C3k2Attn vs C3k2AttnV2

| Feature | C3k2Attn (V1) | C3k2AttnV2 |
|---------|---------------|------------|
| **Pre-Attention** | ❌ Tidak ada | ✅ ECA (optional) |
| **Post-Attention** | ✅ ECA saja | ✅ ECA + CoordinateAttention |
| **Attention Type** | Single (Channel) | Dual (Channel + Spatial-Channel) |
| **Feature Selection** | ❌ | ✅ Pre-attention untuk input |
| **Parameters** | `[c2, c3k, e]` | `[c2, c3k, e, g, shortcut, use_pre_attn]` |
| **Performance** | Good | **Better** (enhanced) |

---

## 🎯 Kapan Pakai C3k2AttnV2?

### ✅ Gunakan C3k2AttnV2 jika:
- Butuh **better feature extraction** untuk small objects
- Ingin **enhanced attention mechanisms**
- Perlu **dual attention** (channel + spatial-channel)
- Dataset dengan **complex scenes** atau **cluttered backgrounds**

### ⚠️ Pertimbangan:
- Sedikit lebih **berat** dari C3k2Attn (tapi masih lightweight)
- Lebih banyak **parameters** karena dual attention
- **Training time** sedikit lebih lama

---

## 📝 Contoh Lengkap YAML

Lihat file: `ultralytics/cfg/models/v12/yolov12-c3k2attnv2-example.yaml`

Atau gunakan template ini:

```yaml
# YOLOv12 dengan C3k2AttnV2
nc: 1
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2AttnV2,  [256, False, 0.25]]  # Enhanced attention
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]
  - [-1, 2, C3k2AttnV2,  [512, False, 0.25]]  # Enhanced attention
  - [-1, 1, Conv,  [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv,  [1024, 3, 2]]
  - [-1, 4, A2C2f, [1024, True, 1]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2AttnV2, [1024, True]]  # Enhanced attention di P5
  - [[14, 17, 20], 1, Detect, [nc]]
```

---

## 🔍 Architecture Detail

```
Input (x)
    ↓
[Pre-Attention: ECA] (optional, if use_pre_attn=True)
    ↓
C3k2 Block
    ├─ cv1: Split channels
    ├─ m: Bottleneck/C3k blocks (n repeats)
    └─ cv2: Concat & Conv
    ↓
[Post-Attention: ECA] (channel attention)
    ↓
[Post-Attention: CoordinateAttention] (spatial-channel attention)
    ↓
Output
```

---

## 💡 Tips & Best Practices

1. **Untuk Small Objects**: Gunakan di P3/P4 layers (higher resolution)
2. **Untuk Large Objects**: Gunakan di P5 layer (semantic features)
3. **Pre-Attention**: Aktifkan (`use_pre_attn=True`) untuk better feature selection
4. **Backbone vs Head**: Bisa dipakai di backbone atau head, tergantung kebutuhan
5. **Mixing**: Bisa dikombinasi dengan C3k2Attn biasa untuk balance performance

---

## 🐛 Troubleshooting

### Error: "C3k2AttnV2 not found"
```python
# Pastikan sudah di-import dengan benar
from ultralytics.nn.modules import C3k2AttnV2
```

### Error: "Module not recognized in YAML"
```yaml
# Pastikan format benar
- [-1, 2, C3k2AttnV2, [256]]  # ✅ Benar
- [-1, 2, c3k2attnv2, [256]]  # ❌ Salah (case sensitive)
```

### Parameter Error
```yaml
# Minimal: hanya c2
- [-1, 2, C3k2AttnV2, [256]]  # ✅ OK

# Lengkap: semua parameter
- [-1, 2, C3k2AttnV2, [256, False, 0.5, 1, True, True]]  # ✅ OK
```

---

## 📚 Referensi

- **C3k2Attn**: Original version dengan ECA attention
- **ECA**: Efficient Channel Attention (CVPR 2020)
- **CoordinateAttention**: Coordinate Attention for Efficient Mobile Network Design (CVPR 2021)

---

## ✅ Checklist Setup

- [x] Class `C3k2AttnV2` dibuat di `block.py`
- [x] Ditambahkan ke `__all__` exports
- [x] Ditambahkan ke `tasks.py` untuk YAML parsing
- [x] Ditambahkan ke `__init__.py` untuk imports
- [x] Contoh YAML file dibuat
- [x] Dokumentasi lengkap dibuat

---

**Selamat menggunakan C3k2AttnV2! 🎉**






