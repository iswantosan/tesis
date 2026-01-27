"""
Training Script untuk mAP50 Boost
Kombinasi: Easy Boost Architecture + P3-weighted Loss + Optimal Training Settings
Expected: +3-5% mAP50 dari baseline
"""

from ultralytics import YOLO

# Load model dengan easy boost architecture
model = YOLO('ultralytics/cfg/models/v12/yolov12-map50-easy-boost.yaml')

# Training dengan optimal settings
results = model.train(
    # Data & Model
    data='your_dataset.yaml',  # Ganti dengan path dataset Anda
    epochs=300,
    patience=50,
    
    # Image & Batch
    imgsz=832,  # Naikkan dari 640 (CRITICAL untuk small objects!)
    batch=16,  # Naikkan ke 32 jika GPU memory cukup
    
    # Optimizer & LR
    optimizer='AdamW',  # Biasanya lebih baik dari SGD
    lr0=0.001,  # Turunkan untuk dataset kecil (default: 0.01)
    lrf=0.1,  # Final LR = lr0 * lrf (0.001 * 0.1 = 0.0001)
    warmup_epochs=5.0,  # Warmup untuk stabilisasi
    weight_decay=0.0005,
    cos_lr=True,  # Cosine learning rate schedule
    
    # Loss Weights
    box=7.5,  # Box regression loss
    cls=1.0,  # Naikkan dari 0.5 jika banyak false positives
    dfl=1.5,  # Distribution Focal Loss
    
    # P3-Weighted Loss (PENTING untuk small objects!)
    p3_weight=1.5,  # P3 dapat bobot 1.5x (RECOMMENDED)
    p4_weight=1.0,
    p5_weight=1.0,
    
    # Data Augmentation
    hsv_h=0.015,  # Hue augmentation (kecil untuk medical)
    hsv_s=0.5,  # Saturation
    hsv_v=0.4,  # Brightness
    degrees=0.0,  # Rotation (hati-hati untuk medical!)
    translate=0.1,  # Translation
    scale=0.5,  # Scale augmentation
    shear=0.0,  # Shear
    perspective=0.0,  # Perspective
    flipud=0.0,  # Flip up-down
    fliplr=0.5,  # Flip left-right
    mosaic=1.0,  # Mosaic augmentation (sangat efektif!)
    mixup=0.1,  # Mixup (hati-hati untuk medical)
    copy_paste=0.1,  # Copy-paste augmentation (baik untuk dataset kecil)
    
    # Training Strategy
    close_mosaic=10,  # Disable mosaic di 10 epochs terakhir
    multi_scale=False,  # Set True untuk multi-scale training (lebih lambat)
    amp=True,  # Mixed precision training
    cache=False,  # Set True untuk faster training (butuh RAM lebih)
    
    # Device
    device=0,  # GPU device
    workers=8,
    
    # Project
    project='runs/detect',
    name='yolov12_map50_boost',
    exist_ok=True,
    pretrained=True,
    verbose=True,
)

# Validation dengan TTA (optional, untuk final boost)
print("\n=== Validation dengan TTA ===")
metrics_tta = model.val(augment=True)  # Test Time Augmentation
print(f"mAP50 (TTA): {metrics_tta.box.map50:.4f}")
print(f"mAP50-95 (TTA): {metrics_tta.box.map:.4f}")

# Print final results
print("\n=== Final Results ===")
print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
print(f"Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
print(f"Recall: {results.results_dict.get('metrics/recall(B)', 'N/A')}")

print("\n✅ Training selesai! Check results di runs/detect/yolov12_map50_boost/")









