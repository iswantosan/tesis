# 🔬 SFAN Implementation Guide - Selective Feature Amplification Network

## 📊 Analisis Masalah dari Heatmap

### Masalah yang Teridentifikasi:
1. **P3(s8) & P4(s16)**: Terlalu noisy, activation tersebar kemana-mana
2. **P5(s32)**: Justru paling fokus (ironis untuk objek kecil)
3. **Head layers**: Belum cukup diskriminatif
4. **Bounding Box**: Beberapa objek kecil kelewat (false negatives)

### Root Cause:
- **Low-level features (P3, P4)** kebanyakan noise dari background
- **High-level features (P5)** terlalu coarse untuk objek kecil
- Perlu feature refinement di mid-level

---

## 🎯 Konsep SFAN

### Filosofi:
1. **Suppress noise** di P4/P5 (low-level setelah NSB)
2. **Amplify signal** untuk objek kecil
3. **Cross-scale fusion** yang lebih pintar (AFF)

### 3 Modul Baru:

#### 1. **NoiseSuppressionBlock (NSB)**
**Fungsi**: Suppress activation yang tidak penting di P4 & P5
- Dual attention: Channel + Spatial
- Channel attention: Deteksi feature channel mana yang noise
- Spatial attention: Deteksi region mana yang tidak penting
- Output = input × channel_mask × spatial_mask

**Posisi**: Setelah backbone P4 & P5 output

**Expected**: Heatmap P4/P5 lebih fokus seperti P5(s32) sekarang

#### 2. **AdaptiveFeatureFusion (AFF)**
**Fungsi**: Ganti concat/add biasa dengan fusion yang bisa "pilih" feature penting
- 2 learnable parameters: w_p4 dan w_p5
- Fused = (w_p4 × P4 + w_p5 × P5_upsampled) / (w_p4 + w_p5)
- Model belajar sendiri feature mana yang lebih penting
- Tambah local context extraction (depthwise conv)

**Posisi**: Di neck, replace bagian concat P4+P5

**Expected**: Model bisa "pilih" fokus ke feature yang paling informatif

#### 3. **SmallObjectEnhancementHead (SOEH)**
**Fungsi**: Detection head dengan 2 branch: standard + small-object-specific
- **Branch 1 (standard)**: Detection normal
- **Branch 2 (enhanced)**: 
  - Extra conv layers untuk refine
  - Objectness weight (sigmoid) untuk fokus ke region berpotensi ada objek
  - Output = feature × objectness_weight
- **Training**: 2 loss (standard + auxiliary untuk objek <32px)
- **Inference**: Weighted ensemble (0.6×standard + 0.4×enhanced)

**Posisi**: Replace Detect head di output P4 & P5

**Expected**: Sensitivity lebih tinggi untuk objek kecil, false negative turun

---

## 📁 File Structure

### YAML Configs:
1. **`yolov12-sfan.yaml`**: Full SFAN (NSB + AFF + SOEH)
2. **`yolov12-sfan-simple.yaml`**: Simple version (NSB + SOEH, skip AFF)

---

## 🔧 Arsitektur Flow (P4-P5 Base)

```
Backbone:
  ... (sama seperti biasa)
  ↓
[P4 output] → NSB → P4_clean  (layer 7)
  ↓
[P5 output] → NSB → P5_clean  (layer 10)

Neck (Full SFAN):
  P5_clean → Upsample(2x) ─┐
                            ├→ AFF → C2f → P4_neck (layer 13)
  P4_clean ─────────────────┘
  
  P4_neck → Conv(downsample) ─┐
                               ├→ Concat → C3k2 → P5_neck (layer 16)
  P5_clean ────────────────────┘

Head:
  P4_neck → SOEH ─→ predictions_p4
  P5_neck → SOEH ─→ predictions_p5
```

---

## 💻 Implementation Steps

### Step 1: Implement NoiseSuppressionBlock
**File**: `ultralytics/nn/modules/block.py`

```python
class NoiseSuppression(nn.Module):
    """
    Noise Suppression Block (NSB)
    Mengurangi activation noise di low-level features dengan dual attention
    """
    def __init__(self, c1, ratio=4):
        super().__init__()
        # Channel attention untuk deteksi noise
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # MLP untuk generate suppression mask
        self.mlp = nn.Sequential(
            nn.Conv2d(c1*2, c1//ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1//ratio, c1, 1, bias=False)
        )
        
        # Spatial attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise suppression
        avg_feat = self.avg_pool(x)
        max_feat = self.max_pool(x)
        channel_mask = self.sigmoid(
            self.mlp(torch.cat([avg_feat, max_feat], 1))
        )
        
        # Spatial suppression
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_mask = self.spatial_conv(
            torch.cat([avg_spatial, max_spatial], 1)
        )
        
        # Combined suppression
        x = x * channel_mask * spatial_mask
        return x
```

### Step 2: Implement AdaptiveFeatureFusion
**File**: `ultralytics/nn/modules/block.py`

```python
class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive Feature Fusion (AFF)
    Fusion P4 & P5 dengan learnable weight
    """
    def __init__(self, c1, c2):
        super().__init__()
        from .conv import Conv
        
        # Learnable fusion weight
        self.weight_p4 = nn.Parameter(torch.ones(1))
        self.weight_p5 = nn.Parameter(torch.ones(1))
        
        # Upsampling untuk P5
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Fusion conv
        self.fusion_conv = nn.Sequential(
            Conv(c1, c2, 1),
            Conv(c2, c2, 3)
        )
        
        # Local context extractor
        self.context = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, c2, 1),
        )
    
    def forward(self, p4, p5):
        # Upsample P5 ke size P4
        p5_up = self.upsample(p5)
        
        # Adaptive weighted fusion
        w_sum = self.weight_p4 + self.weight_p5 + 1e-4
        fused = (self.weight_p4 * p4 + self.weight_p5 * p5_up) / w_sum
        
        # Fusion conv
        out = self.fusion_conv(fused)
        
        # Add local context
        out = out + self.context(out)
        
        return out
```

### Step 3: Implement SmallObjectEnhancementHead
**File**: `ultralytics/nn/modules/head.py`

```python
class SmallObjectEnhancementHead(Detect):
    """
    Detection Head dengan enhancement untuk small object
    2 branch: standard + small-object-specific dengan auxiliary loss
    """
    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        from .conv import Conv
        
        # Small object branch (auxiliary)
        self.small_obj_branch = nn.ModuleList([
            nn.Sequential(
                Conv(c, c, 3),
                Conv(c, c, 3),
                nn.Conv2d(c, self.no, 1)
            ) for c in ch
        ])
        
        # Objectness enhancement
        self.obj_enhance = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c//2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c//2, 1, 1),
                nn.Sigmoid()
            ) for c in ch
        ])
    
    def forward(self, x):
        # Standard detection
        standard_out = []
        enhanced_out = []
        
        for i, (feat, m, small_m, obj_m) in enumerate(
            zip(x, self.cv3, self.small_obj_branch, self.obj_enhance)
        ):
            # Standard path
            standard = m(feat)
            
            # Small object enhanced path
            obj_weight = obj_m(feat)
            small_enhanced = small_m(feat * obj_weight)
            
            standard_out.append(standard)
            enhanced_out.append(small_enhanced)
        
        if self.training:
            # Return both untuk auxiliary loss
            return standard_out, enhanced_out
        else:
            # Inference: weighted combination
            final_out = [
                0.6 * s + 0.4 * e 
                for s, e in zip(standard_out, enhanced_out)
            ]
            return final_out
```

### Step 4: Register Modules
**File**: `ultralytics/nn/modules/__init__.py`

```python
from .block import (
    ...,
    NoiseSuppression,
    AdaptiveFeatureFusion
)

from .head import (
    ...,
    SmallObjectEnhancementHead
)
```

### Step 5: Update tasks.py Parser
**File**: `ultralytics/nn/tasks.py`

Tambah parsing logic untuk:
- `NoiseSuppression`: Args `[c1, ratio]`
- `AdaptiveFeatureFusion`: Args `[c1, c2]`, takes 2 inputs `[p5_upsampled, p4]`
- `SmallObjectEnhancementHead`: Args `[nc]`, takes list of inputs

---

## 🚀 Usage

### Full SFAN:
```python
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/v12/yolov12-sfan.yaml')
model.train(
    data='your_data.yaml',
    epochs=300,
    imgsz=640,
    batch=16,
    patience=50
)
```

### Simple SFAN (Recommended untuk start):
```python
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/v12/yolov12-sfan-simple.yaml')
model.train(
    data='your_data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

---

## 📈 Expected Results

### Full SFAN:
- **Heatmap P4/P5**: Lebih fokus, less noise (seperti P5 sekarang)
- **mAP50**: +1-1.5% (target 88%+ dari baseline 87.4%)
- **Small object recall**: +2-3%
- **Precision/Recall**: Lebih balanced

### Simple SFAN:
- **Heatmap P4/P5**: Lebih fokus (NSB effect)
- **mAP50**: +0.5-1% (lebih cepat implement)
- **Small object recall**: +1-2%

---

## ⏱️ Timeline

### Full SFAN:
- NSB implementation: 2-3 jam
- AFF implementation: 1-2 jam
- SOEH implementation: 3-4 jam
- Integration & testing: 2-3 jam
- **Total: 1 hari kerja intensif**

### Simple SFAN:
- NSB implementation: 2-3 jam
- SOEH implementation: 3-4 jam
- Integration & testing: 1-2 jam
- **Total: 6-9 jam (lebih cepat)**

---

## 🎯 Kenapa Ini Bisa Kerja

1. **NSB** benerin masalah utama: noise di P4/P5
2. **AFF** bikin fusion lebih pintar, tidak sekedar concat
3. **SOEH** specifically boost detection objek kecil dengan auxiliary supervision

---

## 🔄 Alternative: Simple Version First

**Rekomendasi**: Start dengan **`yolov12-sfan-simple.yaml`** (NSB + SOEH)
- Implementasi lebih cepat (6-9 jam vs 1 hari)
- Expected improvement +0.5-1% (cukup baik untuk validate konsep)
- Jika hasil bagus, baru implement full SFAN dengan AFF

**Good luck! 🚀**

