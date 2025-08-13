#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path
import yaml


def generate_data_yaml(dataset_dir: Path) -> str:
    dataset_dir = Path(dataset_dir)
    class_ids = set()
    train_labels_dir = dataset_dir / "labels" / "train"
    if not train_labels_dir.exists():
        raise RuntimeError(f"Missing labels directory: {train_labels_dir}")
    for lbl in train_labels_dir.glob("*.txt"):
        for line in lbl.read_text().splitlines():
            if not line.strip():
                continue
            class_ids.add(int(float(line.split()[0])))
    if not class_ids:
        raise RuntimeError(f"No labels found in {train_labels_dir}")
    nc = max(class_ids) + 1
    names = [str(i) for i in range(nc)]
    cfg = {
        "train": str((dataset_dir / "images" / "train").resolve()),
        "val": str((dataset_dir / "images" / "val").resolve()),
        "test": str((dataset_dir / "images" / "test").resolve()),
        "nc": nc,
        "names": names,
    }
    out = dataset_dir / "data.yaml"
    out.write_text(yaml.dump(cfg))
    print(f"→ Wrote data config to {out}")
    return str(out)


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv7 using the official repo scripts (train.py/test.py)")
    p.add_argument("--dataset_dir", required=True, help="Root with images/{train,val,test} and labels/{train,val,test}")
    p.add_argument("--yolov7_repo", required=True, help="Path to cloned YOLOv7 repo (must contain train.py/test.py)")
    p.add_argument("--cfg", default="cfg/training/yolov7.yaml", help="Model cfg relative to repo root")
    p.add_argument("--weights", default="yolov7.pt", help="Pretrained weights path or name relative to repo")
    p.add_argument("--epochs", type=int, default=50, help="Epochs")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size")
    p.add_argument("--imgsz", type=int, default=640, help="Image size")
    p.add_argument("--device", default="0", help="CUDA device(s), e.g. '0' or '0,1' or 'cpu'")
    p.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    p.add_argument("--project", default="runs/train", help="Project directory (created under repo root)")
    p.add_argument("--name", default="exp_yolov7", help="Experiment name")
    p.add_argument("--resume", action="store_true", help="Resume training")
    p.add_argument("--resume_from", type=Path, default=None, help="Path to checkpoint to resume from (overrides --resume)")
    p.add_argument("--hyp", type=str, default=None, help="Path to hyperparameters yaml (relative or absolute)")
    p.add_argument("--cache", action="store_true", help="Cache images for faster training")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()
    repo = Path(args.yolov7_repo).resolve()
    train_py = repo / "train.py"
    test_py = repo / "test.py"
    if not train_py.exists():
        raise FileNotFoundError(f"train.py not found under {repo}. Did you set --yolov7_repo correctly?")
    if not test_py.exists():
        raise FileNotFoundError(f"test.py not found under {repo}. Did you set --yolov7_repo correctly?")

    data_yaml = generate_data_yaml(Path(args.dataset_dir))

    cmd = [
        sys.executable, str(train_py),
        "--workers", str(args.workers),
        "--device", str(args.device),
        "--batch-size", str(args.batch_size),
        "--data", data_yaml,
        "--cfg", str(args.cfg),
        "--weights", str(args.weights),
        "--name", str(args.name),
        "--project", str(args.project),
        "--epochs", str(args.epochs),
        "--img-size", str(args.imgsz), str(args.imgsz),
        "--seed", str(args.seed),
    ]
    if args.cache:
        cmd += ["--cache-images"]
    if args.resume_from:
        cmd += ["--resume", str(args.resume_from)]
    elif args.resume:
        cmd += ["--resume"]
    if args.hyp:
        cmd += ["--hyp", str(args.hyp)]

    print("→ Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(repo))

    best = (repo / args.project / args.name / "weights" / "best.pt").resolve()
    if not best.exists():
        print("best.pt not found, trying last.pt")
        best = (repo / args.project / args.name / "weights" / "last.pt").resolve()
    print(f"→ Evaluating {best}")
    test_cmd = [
        sys.executable, str(test_py),
        "--data", data_yaml,
        "--img", str(args.imgsz),
        "--batch", str(args.batch_size),
        "--weights", str(best),
        "--device", str(args.device),
        "--task", "test",
    ]
    subprocess.run(test_cmd, check=True, cwd=str(repo))


if __name__ == "__main__":
    main()
