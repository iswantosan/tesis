# 🎯 Ide Perbaikan P3/P4 untuk BTA Detection

Berdasarkan analisis overlay heatmap, aktivasi di P3/P4 masih terlalu luas pada sel besar. Berikut ide dan implementasi yang lebih agresif:

## 📊 Analisis Masalah

1. **P3 Backbone**: Aktivasi terlalu luas, menutupi sel besar
2. **P4 Backbone**: Aktivasi besar dan kurang presisi
3. **HeadP3**: Sudah bagus tapi bisa lebih presisi untuk rod kecil
4. **HeadP4**: Aktivasi kurang fokus pada rod kecil

## 🚀 Blok Baru yang Diimplementasikan

### 1. AggressiveBackgroundSuppression
**Tujuan**: Suppress aktivasi besar lebih agresif dari BSG biasa

**Perbedaan dengan BSG:**
- Multi-scale low-pass (k=9 dan k=7) → lebih akurat detect background
- Langsung subtract background, bukan hanya gate
- Adaptive threshold berdasarkan foreground magnitude
- Suppression strength configurable (default: 0.7)

**Arsitektur:**
```
bg1 = DWConv(k=9)  # Large kernel
bg2 = DWConv(k=7)  # Medium kernel
bg_combined = (bg1 + bg2) / 2
fg = x - bg_combined
threshold = sigmoid(Conv1x1(|fg|))
out = fg * threshold + x * (1 - threshold) * 0.3
```

**Penempatan:**
- P3 backbone output (setelah C3k2, sebelum turun ke P4)
- P4 backbone output (setelah A2C2f)

### 2. CrossScaleSuppression
**Tujuan**: P3 dan P4 saling reference untuk suppress background

**Konsep:**
- P4 (coarser) membantu identify large structures di P3 (finer)
- P4 downsample → generate background mask → suppress P3
- Returns P3 yang sudah di-suppress

**Arsitektur:**
```
P4_down = Downsample(P4) to match P3
bg_mask = sigmoid(Conv3x3(P4_down))
P3_suppressed = P3 * (1 - bg_mask * strength)
```

**Penempatan:**
- Di head, setelah concat P3 dan P4
- Input: [HeadP3, HeadP4]
- Output: P3 yang sudah di-suppress

### 3. MultiScaleEdgeEnhancement
**Tujuan**: Enhance edges di multiple scales untuk rod kecil

**Perbedaan dengan ELEB:**
- Multi-scale high-pass (3x3 dan 5x5)
- Spatial attention + Channel attention
- Enhancement strength configurable (default: 2.0)

**Arsitektur:**
```
hp3 = DWConv3x3(x) - AvgPool3x3(x)
hp5 = DWConv5x5(x) - AvgPool5x5(x)
hp_combined = Conv1x1(concat([hp3, hp5]))
spatial_attn = sigmoid(Conv1x1(hp_combined))
channel_attn = sigmoid(MLP(GAP(hp_combined)))
out = x + hp_combined * spatial_attn * channel_attn * 2.0
```

**Penempatan:**
- HeadP3 (setelah CrossScaleSuppression)
- HeadP4 (optional)

## 📁 File YAML yang Tersedia

### 1. `yolov12-p3p4-aggressive.yaml`
**Strategi:**
- ABS di P3 dan P4 backbone
- CrossScaleSuppression di head (P3↔P4)
- MultiScaleEdgeEnhancement di HeadP3

**Flow:**
```
Backbone P3 → ABS → Backbone P4 → ABS
Head: Concat P3/P4 → CrossScaleSuppression → MSE → Detect
```

### 2. `yolov12-p3p4-strong.yaml`
**Strategi:**
- ABS di P3 dan P4 backbone (strength=0.8)
- MultiScaleEdgeEnhancement di HeadP3 dan HeadP4
- Final ABS di HeadP3 sebelum Detect

**Flow:**
```
Backbone P3 → ABS(0.8) → Backbone P4 → ABS(0.8)
Head: MSE di P3/P4 → Final ABS di P3 → Detect
```

## 🎛️ Parameter Tuning

### AggressiveBackgroundSuppression
- `suppression_strength`: 0.6-0.9
  - 0.6: Mild suppression
  - 0.7: Default (balanced)
  - 0.8: Strong suppression
  - 0.9: Very aggressive

### CrossScaleSuppression
- `suppression_strength`: 0.7-0.9
  - 0.7: Mild cross-scale suppression
  - 0.8: Default
  - 0.9: Very aggressive

### MultiScaleEdgeEnhancement
- `enhancement_strength`: 1.5-3.0
  - 1.5: Subtle enhancement
  - 2.0: Default
  - 2.5: Strong enhancement
  - 3.0: Very aggressive

## 💡 Tips Penggunaan

1. **Mulai dengan `yolov12-p3p4-aggressive.yaml`** - kombinasi lengkap
2. **Jika masih terlalu banyak false positive**: naikkan `suppression_strength` di ABS
3. **Jika rod kecil masih kurang terdeteksi**: naikkan `enhancement_strength` di MSE
4. **Jika P3 dan P4 tidak sinkron**: gunakan CrossScaleSuppression dengan strength lebih tinggi

## 🔍 Monitoring

Setelah training, cek overlay heatmap:
- **P3/P4 backbone**: Harusnya aktivasi lebih fokus, tidak terlalu luas
- **HeadP3**: Harusnya aktivasi lebih presisi pada rod kecil
- **HeadP4**: Harusnya aktivasi lebih fokus, tidak terlalu blur

## 📝 Catatan

- Blok ini lebih agresif dari versi sebelumnya
- Mungkin perlu adjust learning rate atau training schedule
- Monitor loss untuk memastikan tidak over-suppress atau over-enhance

