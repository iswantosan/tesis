"""
Contoh Training dengan Freeze dan Unfreeze Strategy
===================================================

Strategy:
1. Stage 1: Freeze backbone (freeze=10 atau 20), train hanya head
2. Stage 2: Unfreeze semua (freeze=None atau freeze=0), continue training dengan LR lebih kecil
"""

from ultralytics import YOLO

# ==================== STAGE 1: FREEZE BACKBONE ====================
print("=" * 60)
print("STAGE 1: Freeze Backbone - Train Only Head")
print("=" * 60)

# Load model
model = YOLO("yolov12.yaml")  # atau yolov12n.pt jika pakai pretrained

# Training dengan freeze backbone
# freeze=10 berarti freeze layer 0-9 (backbone)
# freeze=20 berarti freeze layer 0-19 (backbone + early head)
results_stage1 = model.train(
    data="your_dataset.yaml",
    epochs=50,  # Train head dulu
    imgsz=640,
    batch=16,
    freeze=10,  # FREEZE: Freeze first 10 layers (backbone)
    lr0=0.01,   # LR bisa lebih besar karena hanya train head
    optimizer="SGD",
    momentum=0.937,
    weight_decay=0.0005,
    patience=0,  # No early stopping di stage 1
    project="runs/detect",
    name="stage1_freeze",
    save=True,
    verbose=True,
)

print("\nStage 1 completed! Best model saved at:", model.trainer.best)

# ==================== STAGE 2: UNFREEZE ALL ====================
print("\n" + "=" * 60)
print("STAGE 2: Unfreeze All - Fine-tune Full Model")
print("=" * 60)

# Load best checkpoint dari stage 1
best_ckpt = model.trainer.best  # Path ke best.pt dari stage 1
model_unfreeze = YOLO(best_ckpt)  # Load checkpoint

# Continue training dengan UNFREEZE semua layers
results_stage2 = model_unfreeze.train(
    data="your_dataset.yaml",
    epochs=150,  # Total epochs (stage1 + stage2)
    imgsz=640,
    batch=16,
    freeze=None,  # UNFREEZE: Train semua layers (atau freeze=0)
    lr0=0.001,    # LR lebih kecil karena train semua layers
    lrf=0.1,      # Final LR = lr0 * lrf = 0.0001
    optimizer="AdamW",  # Bisa ganti ke AdamW untuk fine-tuning
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,  # Warmup lagi karena LR lebih kecil
    patience=50,      # Early stopping di stage 2
    cos_lr=True,      # Cosine LR scheduler untuk smooth decay
    project="runs/detect",
    name="stage2_unfreeze",
    save=True,
    verbose=True,
)

print("\nStage 2 completed! Final best model saved at:", model_unfreeze.trainer.best)


# ==================== ALTERNATIVE: PROGRESSIVE UNFREEZING ====================
"""
Progressive Unfreezing Strategy:
- Stage 1: freeze=20 (freeze backbone + early layers)
- Stage 2: freeze=10 (freeze hanya backbone)
- Stage 3: freeze=None (unfreeze semua)
"""

def progressive_unfreeze_training():
    """Progressive unfreezing example"""
    
    # Stage 1: Freeze banyak layers
    model = YOLO("yolov12.yaml")
    model.train(
        data="your_dataset.yaml",
        epochs=30,
        freeze=20,  # Freeze first 20 layers
        lr0=0.01,
        project="runs/detect",
        name="progressive_stage1",
    )
    
    # Stage 2: Freeze lebih sedikit
    model = YOLO(model.trainer.best)  # Load best dari stage 1
    model.train(
        data="your_dataset.yaml",
        epochs=30,
        freeze=10,  # Freeze first 10 layers only
        lr0=0.005,  # LR lebih kecil
        project="runs/detect",
        name="progressive_stage2",
    )
    
    # Stage 3: Unfreeze semua
    model = YOLO(model.trainer.best)  # Load best dari stage 2
    model.train(
        data="your_dataset.yaml",
        epochs=100,
        freeze=None,  # Unfreeze semua
        lr0=0.001,    # LR lebih kecil lagi
        lrf=0.1,
        project="runs/detect",
        name="progressive_stage3",
    )


# ==================== NOTES ====================
"""
PENTING:
1. Freeze Parameter:
   - freeze=10: Freeze layer 0-9 (backbone)
   - freeze=[0,1,2]: Freeze specific layers
   - freeze=None atau freeze=0: Tidak freeze (train semua)

2. Learning Rate Strategy:
   - Stage freeze: LR bisa lebih besar (0.01) karena hanya train head
   - Stage unfreeze: LR lebih kecil (0.001) karena train semua layers

3. Resume Training:
   - Jika pakai resume=True, model akan load last.pt dan continue
   - Jika ingin pakai best.pt, load manual: model = YOLO("best.pt")

4. Check Layer Names:
   - Untuk lihat layer names: print([name for name, _ in model.model.named_parameters()])
   - Adjust freeze number berdasarkan layer yang ingin di-freeze

5. Best Practice:
   - Dataset kecil: freeze lebih banyak (20-30)
   - Dataset besar: freeze sedikit (10) atau tidak freeze
   - Transfer learning: freeze backbone dulu, lalu unfreeze
"""


