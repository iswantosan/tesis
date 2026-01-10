# YOLOv12-IDE3: FP Confidence Dampener

## ⚠️ INI BUKAN YAML CONFIG - INI LOSS-SIDE MODIFICATION

IDE 3 adalah **modifikasi loss function**, bukan perubahan arsitektur. Tidak perlu file YAML baru.

---

## 🎯 Tujuan

Kalau model confident tapi salah → dihukum lebih keras.

## 🧠 Intuisi

Texture FP itu:
- **Confident** (prediksi tinggi)
- **Konsisten** (sering muncul)
- **Tapi salah** (false positive)

## 🧩 Implementasi

### Option 1: Custom Loss Function (Recommended)

Buat file `ultralytics/utils/loss.py` atau modify existing loss:

```python
# Di loss computation (biasanya di compute_loss atau similar function)

# Original classification loss
loss_cls = BCE_loss(pred_cls, target_cls)

# FP Confidence Dampener
fp_weight = torch.sigmoid(pred_obj)  # Semakin yakin, semakin besar
beta = 1.5  # Dampening strength (1.0-2.0 recommended)

# Weighted loss: semakin confident, semakin dihukum
loss_cls_weighted = loss_cls * (1 + beta * fp_weight)

# Final loss
loss = loss_box + loss_cls_weighted + loss_dfl
```

### Option 2: Via Training Hyperparameters

Tambahkan ke training config (jika sudah ada support):

```yaml
# training_config.yaml
fp_dampener: True
fp_dampener_beta: 1.5  # Strength (1.0-2.0)
```

### Option 3: Modify Loss Module

Jika ada custom loss module, tambahkan di `__init__`:

```python
class FPWeightedLoss(nn.Module):
    def __init__(self, beta=1.5):
        super().__init__()
        self.beta = beta
    
    def forward(self, pred_cls, target_cls, pred_obj):
        # Original loss
        loss_cls = F.binary_cross_entropy(pred_cls, target_cls, reduction='none')
        
        # FP weight
        fp_weight = torch.sigmoid(pred_obj)  # [B, N, 1]
        
        # Weighted
        loss_cls = loss_cls * (1 + self.beta * fp_weight)
        
        return loss_cls.mean()
```

---

## 📌 Parameter Recommendations

- **beta = 1.0**: Light dampening (mulai dari sini)
- **beta = 1.5**: **Recommended** untuk kebanyakan kasus
- **beta = 2.0**: Aggressive (jika masih banyak FP)

---

## ⚠️ Catatan

1. **Tidak perlu ubah arsitektur** - ini pure loss weighting
2. **Compatible dengan semua YOLOv12 variants**
3. **Bisa dikombinasikan dengan IDE 1, 2, 4**
4. **Monitor loss curves** - pastikan tidak explode

---

## ✅ Expected Results

- **False Positive turun** (confident FP dihukum keras)
- **mAP50 naik** (lebih sedikit FP, precision naik)
- **Tidak impact recall** (hanya hukumi confident predictions)

---

**Usage:**
```python
# Pakai IDE 1, 2, atau 4 untuk arsitektur
# Tambahkan IDE 3 via custom loss function
model = YOLO('ultralytics/cfg/models/v12/yolov12-ide1-local-contrast.yaml')
# ... modify loss function dengan FP dampener ...
model.train(data='your_data.yaml', epochs=200)
```

