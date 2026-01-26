# 🔍 Bukti: `nc` Mempengaruhi Ukuran Output Layer

## 📋 Klaim

**`nc=80` membuat output layer lebih besar (80 channels) daripada `nc=1` (1 channel)**

---

## ✅ Bukti dari Kode

### **1. Detect Head Initialization**

**File:** `ultralytics/nn/modules/head.py`  
**Line:** 34-57

```python
class Detect(nn.Module):
    """YOLO Detect head for detection models."""
    
    def __init__(self, nc=80, ch=()):
        """Initializes the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes ← BUKTI 1: nc disimpan
        
        # BUKTI 2: Jumlah output per anchor bergantung pada nc
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        # Jika nc=80: self.no = 80 + 16*4 = 144
        # Jika nc=1:  self.no = 1 + 16*4 = 65
        
        # BUKTI 3: Channel size untuk classification bergantung pada nc
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        # c3 = max(ch[0], min(nc, 100))
        # Jika nc=80: c3 = max(ch[0], 80)  (atau 100 jika ch[0] > 100)
        # Jika nc=1:  c3 = max(ch[0], 1)   (atau 100 jika ch[0] > 100)
        
        # BUKTI 4: Output layer menggunakan self.nc sebagai output channels
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), 
                          nn.Conv2d(c3, self.nc, 1)) for x in ch)  # ← BUKTI!
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),  # ← BUKTI! Output = nc channels
                )
                for x in ch
            )
        )
```

**Penjelasan:**
- **Line 47 & 53**: `nn.Conv2d(c3, self.nc, 1)` 
  - Input: `c3` channels
  - **Output: `self.nc` channels** ← Inilah buktinya!
  - Jika `nc=80` → Output = **80 channels**
  - Jika `nc=1` → Output = **1 channel**

---

## 📊 Perhitungan Detail

### **Skenario 1: `nc=80`**

```python
nc = 80
reg_max = 16

# 1. Jumlah output per anchor
self.no = nc + reg_max * 4
self.no = 80 + 16 * 4 = 80 + 64 = 144 outputs

# 2. Channel size untuk classification
# Asumsi: ch[0] = 256 (input channel dari neck)
c3 = max(ch[0], min(nc, 100))
c3 = max(256, min(80, 100))
c3 = max(256, 80) = 256

# 3. Output layer
nn.Conv2d(c3, self.nc, 1)
nn.Conv2d(256, 80, 1)  # ← Output: 80 channels
```

**Parameter Count untuk Output Layer:**
```
Parameter = c3 * nc + bias
Parameter = 256 * 80 + 80 = 20,480 + 80 = 20,560 parameters
```

### **Skenario 2: `nc=1`**

```python
nc = 1
reg_max = 16

# 1. Jumlah output per anchor
self.no = nc + reg_max * 4
self.no = 1 + 16 * 4 = 1 + 64 = 65 outputs

# 2. Channel size untuk classification
# Asumsi: ch[0] = 256 (input channel dari neck)
c3 = max(ch[0], min(nc, 100))
c3 = max(256, min(1, 100))
c3 = max(256, 1) = 256

# 3. Output layer
nn.Conv2d(c3, self.nc, 1)
nn.Conv2d(256, 1, 1)  # ← Output: 1 channel
```

**Parameter Count untuk Output Layer:**
```
Parameter = c3 * nc + bias
Parameter = 256 * 1 + 1 = 256 + 1 = 257 parameters
```

---

## 📈 Perbandingan Ukuran Model

### **Per Layer (P3, P4, P5):**

| Komponen | `nc=80` | `nc=1` | Selisih |
|----------|---------|--------|---------|
| **Output Channels** | 80 | 1 | **79 channels lebih banyak** |
| **Parameters (per layer)** | 20,560 | 257 | **20,303 parameters lebih banyak** |
| **Output per anchor** | 144 | 65 | 79 outputs lebih banyak |

### **Total untuk 3 Layers (P3, P4, P5):**

| Komponen | `nc=80` | `nc=1` | Selisih |
|----------|---------|--------|---------|
| **Total Parameters** | 61,680 | 771 | **60,909 parameters lebih banyak** |
| **Memory (FP32)** | ~246 KB | ~3 KB | **~243 KB lebih besar** |
| **Memory (FP16)** | ~123 KB | ~1.5 KB | **~121.5 KB lebih besar** |

---

## 🔍 Bukti dari Forward Pass

**File:** `ultralytics/nn/modules/head.py`  
**Line:** 64-74

```python
def forward(self, x):
    """Concatenates and returns predicted bounding boxes and class probabilities."""
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        # cv2: regression branch (4 * reg_max = 64 channels)
        # cv3: classification branch (nc channels) ← BUKTI!
        # 
        # Jika nc=80:
        #   x[i] shape: [batch, 64+80=144, H, W]
        # Jika nc=1:
        #   x[i] shape: [batch, 64+1=65, H, W]
```

**Bukti dari Shape:**
- `nc=80`: Output shape = `[batch, 144, H, W]` (64 reg + 80 cls)
- `nc=1`: Output shape = `[batch, 65, H, W]` (64 reg + 1 cls)

---

## 🧪 Test Praktis

### **Script untuk Verifikasi:**

```python
import torch
from ultralytics.nn.modules.head import Detect

# Test 1: nc=80
detect_80 = Detect(nc=80, ch=(256, 512, 1024))
print("=== nc=80 ===")
for i, layer in enumerate(detect_80.cv3):
    print(f"Layer {i} (cv3): {layer}")
    # Cek output layer (layer terakhir)
    last_layer = list(layer.children())[-1]
    print(f"  Output channels: {last_layer.out_channels}")  # Harus 80
    print(f"  Parameters: {sum(p.numel() for p in last_layer.parameters())}")

# Test 2: nc=1
detect_1 = Detect(nc=1, ch=(256, 512, 1024))
print("\n=== nc=1 ===")
for i, layer in enumerate(detect_1.cv3):
    print(f"Layer {i} (cv3): {layer}")
    # Cek output layer (layer terakhir)
    last_layer = list(layer.children())[-1]
    print(f"  Output channels: {last_layer.out_channels}")  # Harus 1
    print(f"  Parameters: {sum(p.numel() for p in last_layer.parameters())}")

# Test 3: Forward pass
x_80 = [torch.randn(1, 256, 80, 80),   # P3
        torch.randn(1, 512, 40, 40),   # P4
        torch.randn(1, 1024, 20, 20)]   # P5

x_1 = [torch.randn(1, 256, 80, 80),
       torch.randn(1, 512, 40, 40),
       torch.randn(1, 1024, 20, 20)]

out_80 = detect_80(x_80)
out_1 = detect_1(x_1)

print("\n=== Output Shapes ===")
print(f"nc=80: {[o.shape for o in out_80]}")
print(f"nc=1:  {[o.shape for o in out_1]}")
# nc=80: [(1, 144, 80, 80), (1, 144, 40, 40), (1, 144, 20, 20)]
# nc=1:  [(1, 65, 80, 80), (1, 65, 40, 40), (1, 65, 20, 20)]
```

**Expected Output:**
```
=== nc=80 ===
Layer 0 (cv3): Sequential(...)
  Output channels: 80
  Parameters: 20560
Layer 1 (cv3): Sequential(...)
  Output channels: 80
  Parameters: 41040
Layer 2 (cv3): Sequential(...)
  Output channels: 80
  Parameters: 82000

=== nc=1 ===
Layer 0 (cv3): Sequential(...)
  Output channels: 1
  Parameters: 257
Layer 1 (cv3): Sequential(...)
  Output channels: 1
  Parameters: 513
Layer 2 (cv3): Sequential(...)
  Output channels: 1
  Parameters: 1025

=== Output Shapes ===
nc=80: [(1, 144, 80, 80), (1, 144, 40, 40), (1, 144, 20, 20)]
nc=1:  [(1, 65, 80, 80), (1, 65, 40, 40), (1, 65, 20, 20)]
```

---

## 📊 Visualisasi Architecture

### **Detect Head dengan `nc=80`:**

```
Input (P3): [B, 256, H, W]
    ↓
cv3 branch:
  - Conv(256 → 256)
  - Conv(256 → 256)
  - Conv2d(256 → 80)  ← 80 output channels
    ↓
Output: [B, 80, H, W]  ← Classification logits untuk 80 kelas
```

### **Detect Head dengan `nc=1`:**

```
Input (P3): [B, 256, H, W]
    ↓
cv3 branch:
  - Conv(256 → 256)
  - Conv(256 → 256)
  - Conv2d(256 → 1)   ← 1 output channel
    ↓
Output: [B, 1, H, W]  ← Classification logit untuk 1 kelas
```

---

## ✅ Kesimpulan

### **Bukti yang Jelas:**

1. ✅ **Line 47 & 53**: `nn.Conv2d(c3, self.nc, 1)` 
   - Output channels = `self.nc`
   - `nc=80` → 80 channels
   - `nc=1` → 1 channel

2. ✅ **Line 40**: `self.no = nc + self.reg_max * 4`
   - Total outputs bergantung pada `nc`
   - `nc=80` → 144 outputs
   - `nc=1` → 65 outputs

3. ✅ **Parameter Count**:
   - `nc=80`: ~61,680 parameters (3 layers)
   - `nc=1`: ~771 parameters (3 layers)
   - **Selisih: ~60,909 parameters**

4. ✅ **Memory Usage**:
   - `nc=80`: ~246 KB (FP32)
   - `nc=1`: ~3 KB (FP32)
   - **Selisih: ~243 KB**

### **Kesimpulan Final:**

**`nc` di YAML MEMPENGARUHI ukuran output layer secara langsung!**

- Output layer menggunakan `nc` sebagai jumlah output channels
- Semakin besar `nc`, semakin besar ukuran model
- Untuk dataset 1 kelas, gunakan `nc=1` untuk efisiensi

---

**Referensi Kode:**
- `ultralytics/nn/modules/head.py`: `Detect.__init__()` - line 34-57
- `ultralytics/nn/modules/head.py`: `Detect.forward()` - line 64-74























