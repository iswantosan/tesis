"""
Simple Freeze/Unfreeze Training Example
========================================
Praktis dan mudah digunakan
"""

from ultralytics import YOLO

# ==================== STEP 1: FREEZE TRAINING ====================
model = YOLO("yolov12.yaml")

# Freeze backbone (layer 0-9), train hanya head
model.train(
    data="your_dataset.yaml",
    epochs=50,
    freeze=10,  # Freeze first 10 layers
    lr0=0.01,
    batch=16,
    imgsz=640,
    project="runs/detect",
    name="freeze_stage",
)

# Save path untuk stage 2
freeze_best = model.trainer.best  # Path ke best.pt

# ==================== STEP 2: UNFREEZE TRAINING ====================
# Load best model dari freeze stage
model_unfreeze = YOLO(freeze_best)

# Unfreeze semua, continue training dengan LR lebih kecil
model_unfreeze.train(
    data="your_dataset.yaml",
    epochs=150,  # Total epochs
    freeze=None,  # UNFREEZE semua layers
    lr0=0.001,    # LR lebih kecil
    batch=16,
    imgsz=640,
    project="runs/detect",
    name="unfreeze_stage",
)

print(f"Training complete! Best model: {model_unfreeze.trainer.best}")


