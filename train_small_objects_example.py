# Recommended Training Settings for YOLOv12 - Very Small Objects
# Optimized for medical microscopy (Ziehl–Neelsen stain, bacilli detection)

r1 = model.train(
    data=data_yaml,
    epochs=200,                 # More epochs for small objects
    imgsz=1280,                 # ⚠️ CRITICAL: Larger image size (640 is too small!)
    batch=8,                    # Adjust based on GPU memory
    
    # Optimizer
    optimizer="SGD",            # Or "AdamW" with lr0=0.001
    lr0=0.01,                   # Initial LR for SGD (use 0.001 if AdamW)
    lrf=0.1,                    # Final LR factor
    momentum=0.937,             # SGD momentum
    weight_decay=0.0005,        # Weight decay (use 0.01 if AdamW)
    
    # Learning rate schedule
    warmup_epochs=3,            # Warmup to stabilize
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    cos_lr=True,                # Cosine LR scheduler
    
    # Loss weights (important for small objects)
    box=7.5,                    # Higher box loss = better localization
    cls=0.5,                    # Classification loss
    dfl=1.5,                    # Distribution focal loss (helps small objects)
    
    # Data augmentation - MINIMAL untuk small objects
    # Color/HSV (conservative)
    hsv_h=0.010,                # Minimal hue shift (preserve stain colors)
    hsv_s=0.15,                 # Moderate saturation
    hsv_v=0.15,                 # Conservative brightness
    
    # Geometric (MINIMAL - small objects can disappear)
    scale=0.15,                 # ±7.5% scale only
    translate=0.05,             # 2.5% translation (very conservative)
    degrees=5.0,                # Max 5° rotation
    shear=0.0,                  # NO shear (distorts small objects)
    perspective=0.0,            # NO perspective (distorts small objects)
    
    # Flip
    fliplr=0.10,                # 10% horizontal flip
    flipud=0.0,                 # NO vertical flip
    
    # Mosaic (LOW - can make small objects disappear)
    mosaic=0.10,                # Only 10% probability
    close_mosaic=30,            # Turn off after 30 epochs
    
    # Mixing (OFF - can destroy small objects)
    copy_paste=0.0,             # OFF
    mixup=0.0,                  # OFF
    erasing=0.0,                # OFF
    
    # Other settings
    auto_augment=None,          # Disable auto augment (manual control)
    label_smoothing=0.0,        # No smoothing (or 0.05 if needed)
    freeze=0,                   # Train all layers
    patience=50,                # More patience for convergence
    single_cls=True,
    
    # Training options
    cache=True,                 # Cache images (faster)
    multi_scale=False,          # Fixed size instead
    rect=False,                 # Keep aspect ratio
    workers=8,
    
    # Validation & logging
    val=True,
    plots=True,
    save_period=10,
    
    project=project,
    name=f"YOLOv12s_{variant}_run{run_idx}_stage1",
    exist_ok=True,
    verbose=True,
    save=True,
    seed=run_idx-1
)


