import json

NB_PATH = 'Untitled8_(2).ipynb'

FPS_CELL_SOURCE = r'''# =========================================================
# 7. FPS BENCHMARK - measure inference speed per fold + log ke W&B
#    Runs after all folds trained. Uses best.pt from each fold.
#    Speed depends on architecture + hardware + imgsz (not weights significantly).
# =========================================================
import time
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

PROJECT_NAME    = "tb_detections_kfold5_yolov12s_project"
KFOLD_BASE_FPS  = "/content/kfold"
FOLDS_FPS       = [1, 2, 3, 4, 5]
N_WARMUP        = 10        # warmup iters (load model, JIT, cache)
N_TEST          = 100       # measurement iters (averaging)
IMGSZ_FPS       = 640
DEVICE_FPS      = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME        = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

def get_best_pt(fold):
    candidates = [
        f"tb_detections_kfold5_yolov12s_project/fold{fold}_train/weights/best.pt",
        f"tb_detections_kfold5_yolov12s/fold{fold}_train/weights/best.pt",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    matches = list(Path(".").rglob(f"fold{fold}_train/weights/best.pt"))
    return str(matches[0]) if matches else None

def get_test_images(fold, max_imgs=200):
    test_dir = Path(f"{KFOLD_BASE_FPS}/fold{fold}/test/images")
    if not test_dir.exists():
        test_dir = Path(f"{KFOLD_BASE_FPS}/test/images")
    if not test_dir.exists():
        return []
    imgs = sorted([str(p) for p in test_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    while len(imgs) < N_WARMUP + N_TEST and imgs:
        imgs = imgs * 2
    return imgs[:max_imgs]


def measure_fps(model_path, images, imgsz=640, n_warmup=10, n_test=100, device="cuda"):
    model = YOLO(model_path)
    for img in images[:n_warmup]:
        model.predict(img, imgsz=imgsz, device=device, conf=0.25, verbose=False)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    pre, inf, post = [], [], []
    t0 = time.time()
    for img in images[n_warmup:n_warmup + n_test]:
        r = model.predict(img, imgsz=imgsz, device=device, conf=0.25, verbose=False)[0]
        pre.append(r.speed["preprocess"])
        inf.append(r.speed["inference"])
        post.append(r.speed["postprocess"])
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t_wall = (time.time() - t0) * 1000 / n_test

    pre_ms, inf_ms, post_ms = np.mean(pre), np.mean(inf), np.mean(post)
    total_ms = pre_ms + inf_ms + post_ms
    return {
        "preprocess_ms":  float(pre_ms),
        "inference_ms":   float(inf_ms),
        "postprocess_ms": float(post_ms),
        "total_ms":       float(total_ms),
        "wall_total_ms":  float(t_wall),
        "fps_inference":  float(1000 / inf_ms),
        "fps_total":      float(1000 / total_ms),
        "n_images":       int(n_test),
    }


print(f"Hardware: {GPU_NAME}")
print(f"Settings: imgsz={IMGSZ_FPS}, batch=1, FP32, n_warmup={N_WARMUP}, n_test={N_TEST}\n")

fps_results = []
for fold in FOLDS_FPS:
    weights = get_best_pt(fold)
    if not weights:
        print(f"SKIP fold{fold}: best.pt not found")
        continue
    images = get_test_images(fold)
    if not images:
        print(f"SKIP fold{fold}: no test images found")
        continue
    print(f"Measuring fold{fold}...")
    r = measure_fps(weights, images, imgsz=IMGSZ_FPS,
                    n_warmup=N_WARMUP, n_test=N_TEST, device=DEVICE_FPS)
    r["fold"] = fold
    fps_results.append(r)
    print(f"  pre={r['preprocess_ms']:.2f}ms  inf={r['inference_ms']:.2f}ms  "
          f"post={r['postprocess_ms']:.2f}ms  | FPS_inf={r['fps_inference']:.1f}  "
          f"FPS_e2e={r['fps_total']:.1f}")

if not fps_results:
    print("\nNo fold weights found. Run cell 6 (training) first.")
else:
    print("\n" + "=" * 95)
    print(f"FPS BENCHMARK - {len(fps_results)} folds @ {GPU_NAME}, imgsz={IMGSZ_FPS}, batch=1, FP32")
    print("=" * 95)
    print(f"{'Fold':>5} {'Pre(ms)':>9} {'Inf(ms)':>9} {'Post(ms)':>10} "
          f"{'Total(ms)':>11} {'FPS(inf)':>10} {'FPS(e2e)':>10}")
    print("-" * 95)
    for r in fps_results:
        print(f"{r['fold']:>5} {r['preprocess_ms']:>9.2f} {r['inference_ms']:>9.2f} "
              f"{r['postprocess_ms']:>10.2f} {r['total_ms']:>11.2f} "
              f"{r['fps_inference']:>10.1f} {r['fps_total']:>10.1f}")
    print("-" * 95)

    def stat(key):
        vals = [r[key] for r in fps_results]
        return float(np.mean(vals)), float(np.std(vals))

    mean_inf, std_inf   = stat("inference_ms")
    mean_tot, std_tot   = stat("total_ms")
    mean_fpsi, std_fpsi = stat("fps_inference")
    mean_fpse, std_fpse = stat("fps_total")
    print(f"{'MEAN':>5} {'-':>9} {mean_inf:>9.2f} {'-':>10} {mean_tot:>11.2f} "
          f"{mean_fpsi:>10.1f} {mean_fpse:>10.1f}")
    print(f"{'STD':>5} {'-':>9} {std_inf:>9.2f} {'-':>10} {std_tot:>11.2f} "
          f"{std_fpsi:>10.1f} {std_fpse:>10.1f}")
    print("=" * 95)

    import pandas as pd
    df_fps = pd.DataFrame(fps_results)

    fps_run = wandb.init(
        project=PROJECT_NAME,
        name="fps_benchmark",
        reinit=True,
        tags=["fps", "benchmark", "inference"],
        config=dict(
            gpu=GPU_NAME, imgsz=IMGSZ_FPS, batch=1, precision="FP32",
            n_warmup=N_WARMUP, n_test=N_TEST,
        ),
    )

    fps_run.log({"fps/per_fold_table": wandb.Table(dataframe=df_fps)})

    tbl = wandb.Table(dataframe=df_fps.assign(fold_label=[f"fold{i}" for i in df_fps["fold"]]))
    fps_run.log({"fps/fps_inference_bar":
                 wandb.plot.bar(tbl, "fold_label", "fps_inference", title="FPS (inference) per Fold")})
    fps_run.log({"fps/fps_total_bar":
                 wandb.plot.bar(tbl, "fold_label", "fps_total", title="FPS (end-to-end) per Fold")})

    fps_run.summary["fps/inference_mean"]   = mean_fpsi
    fps_run.summary["fps/inference_std"]    = std_fpsi
    fps_run.summary["fps/total_mean"]       = mean_fpse
    fps_run.summary["fps/total_std"]        = std_fpse
    fps_run.summary["fps/inference_ms_mean"] = mean_inf
    fps_run.summary["fps/total_ms_mean"]     = mean_tot
    fps_run.summary["fps/gpu"]              = GPU_NAME
    fps_run.summary["fps/imgsz"]            = IMGSZ_FPS

    fig, ax = plt.subplots(figsize=(10, 5))
    folds   = [r["fold"] for r in fps_results]
    fps_inf = [r["fps_inference"] for r in fps_results]
    fps_tot = [r["fps_total"] for r in fps_results]
    x = np.arange(len(folds))
    ax.bar(x - 0.2, fps_inf, 0.4, label="FPS (inference only)", color="#2ca02c")
    ax.bar(x + 0.2, fps_tot, 0.4, label="FPS (end-to-end)", color="#1f77b4")
    for i, (vi, vt) in enumerate(zip(fps_inf, fps_tot)):
        ax.text(i - 0.2, vi + 1, f"{vi:.0f}", ha="center", fontsize=9)
        ax.text(i + 0.2, vt + 1, f"{vt:.0f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([f"fold{f}" for f in folds])
    ax.set_ylabel("FPS")
    ax.set_title(f"FPS Benchmark - {GPU_NAME}, imgsz={IMGSZ_FPS}")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("/content/fps_benchmark.png", dpi=160, bbox_inches="tight")
    fps_run.log({"fps/bar_chart": wandb.Image("/content/fps_benchmark.png")})
    plt.show()

    fps_run.finish()

    print(f"\nSingle-number for tesis: {mean_fpsi:.1f} +- {std_fpsi:.1f} FPS (inference only)")
    print(f"                          {mean_fpse:.1f} +- {std_fpse:.1f} FPS (end-to-end)")
'''

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

new_cell = {
    "cell_type": "code",
    "metadata": {},
    "source": FPS_CELL_SOURCE.splitlines(keepends=True),
    "execution_count": None,
    "outputs": []
}

# Insert at position 7 (after training cell 6)
nb['cells'].insert(7, new_cell)

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"OK. Notebook now has {len(nb['cells'])} cells.")
print("New FPS cell inserted at index 7.")
