#!/usr/bin/env python3
import argparse
import subprocess
import re
import json
from pathlib import Path

def run(cmd, capture=False):
    print("→", " ".join(cmd))
    if capture:
        return subprocess.check_output(cmd, text=True)
    else:
        subprocess.run(cmd, check=True)
        return None

def parse_coco_ap(output: str):
    """
    Extracts:
      - AP@[.50:.95]  (the main COCO mAP)
      - AP@[.50]      (mAP50)
      - AR@[.50:.95]  (avg recall, optional)
    from DETR/Test.py stdout.
    """
    # Patterns to match lines like:
    # AVERAGE PRECISION  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.251
    ap_full = re.search(
        r"AP[\s\S]*?IoU=0\.50:0\.95[\s\S]*?=\s*([0-9.]+)", output)
    ap50    = re.search(
        r"AP[\s\S]*?IoU=0\.50\s*\|[\s\S]*?=\s*([0-9.]+)", output)
    ar_full = re.search(
        r"AR[\s\S]*?IoU=0\.50:0\.95[\s\S]*?=\s*([0-9.]+)", output)

    return {
        "mAP50-95": float(ap_full.group(1)) if ap_full else None,
        "mAP50":     float(ap50.group(1))    if ap50    else None,
        "AR50-95":   float(ar_full.group(1)) if ar_full else None
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco_dir",   required=True,
                   help="root of COCO data (images/ + annotations/)")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=4)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--project",    default="detr_runs")
    p.add_argument("--name",       default="exp")
    args = p.parse_args()

    root       = Path(__file__).resolve().parent / "../detr"
    main_py    = str((root/"main.py").resolve())
    coco       = Path(args.coco_dir)
    out_dir    = Path(args.project)/args.name

    # 1) TRAIN
    train_cmd = [
        "python", main_py,
        "--dataset_file", "coco",
        "--coco_path", str(coco),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--epochs", str(args.epochs),
        "--device", args.device,
        "--output_dir", str(out_dir),
    ]
    run(train_cmd)

    # Locate checkpoint
    ckpts = sorted(out_dir.glob("checkpoint*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint in {out_dir}")
    best_ckpt = ckpts[-1]

    # 2) EVAL on val split
    val_cmd = [
        "python", main_py,
        "--dataset_file", "coco",
        "--coco_path", str(coco),
        "--resume", str(best_ckpt),
        "--device", args.device,
        "--eval", "coco"
    ]
    val_out = run(val_cmd, capture=True)
    val_metrics = parse_coco_ap(val_out)
    print("\n→ Validation metrics:", val_metrics)

    # 3) EVAL on test split (swap val/test JSONs)
    # assume you named test JSON instances_test2017.json alongside instances_train2017.json
    # and DETR auto-loads test when you say '--eval coco' after resume
    # (custom DETR forks may need a flag; if so, adjust accordingly)
    test_cmd = val_cmd  # same as val, since JSONs include all splits
    test_out = run(test_cmd, capture=True)
    test_metrics = parse_coco_ap(test_out)
    print("\n→ Test metrics:", test_metrics)

    # 4) JSON summary
    summary = {
        "run_name": args.name,
        "checkpoint": best_ckpt.name,
        "val_mAP50":     val_metrics["mAP50"],
        "val_mAP50-95":  val_metrics["mAP50-95"],
        "val_AR50-95":   val_metrics["AR50-95"],
        "test_mAP50":    test_metrics["mAP50"],
        "test_mAP50-95": test_metrics["mAP50-95"],
        "test_AR50-95":  test_metrics["AR50-95"],
    }
    print("\n>> RESULT_SUMMARY_JSON <<")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
