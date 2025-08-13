#!/usr/bin/env python3
import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO


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
    p = argparse.ArgumentParser(description="Train a YOLOv11 detector on a split dataset")
    p.add_argument("--dataset_dir", required=True, help="Root with images/{train,val,test} and labels/{train,val,test}")
    p.add_argument("--weights", default="yolo11s.pt", help="YOLOv11 weights to start from (e.g. yolo11s.pt or yolov11s.pt)")
    p.add_argument("--epochs", type=int, default=50, help="Epochs")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size")
    p.add_argument("--imgsz", type=int, default=640, help="Image size")
    p.add_argument("--device", default="auto", help="Device, e.g. 0 or cpu or auto")
    p.add_argument("--workers", type=int, default=8, help="Data loader workers")
    p.add_argument("--project", default="runs/train", help="Ultralytics project directory")
    p.add_argument("--name", default="exp", help="Experiment name")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--lr0", type=float, default=None, help="Initial learning rate")
    p.add_argument("--lrf", type=float, default=None, help="Final LR fraction")
    p.add_argument("--optimizer", type=str, default=None, help="Optimizer: SGD, Adam, AdamW, or auto")
    p.add_argument("--momentum", type=float, default=None, help="Momentum (or beta1 for Adam/AdamW)")
    p.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    p.add_argument("--warmup_epochs", type=float, default=None, help="Warmup epochs")
    p.add_argument("--warmup_bias_lr", type=float, default=None, help="Warmup bias LR")
    return p.parse_args()


def main():
    args = parse_args()
    data_yaml = generate_data_yaml(Path(args.dataset_dir))
    model = YOLO(args.weights)
    print(f"→ Starting training with weights={args.weights}")
    train_kwargs = dict(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
    )
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    if args.lrf is not None:
        train_kwargs["lrf"] = args.lrf
    if args.optimizer is not None:
        train_kwargs["optimizer"] = args.optimizer
    if args.momentum is not None:
        train_kwargs["momentum"] = args.momentum
    if args.weight_decay is not None:
        train_kwargs["weight_decay"] = args.weight_decay
    if args.warmup_epochs is not None:
        train_kwargs["warmup_epochs"] = args.warmup_epochs
    if args.warmup_bias_lr is not None:
        train_kwargs["warmup_bias_lr"] = args.warmup_bias_lr
    model.train(**train_kwargs)
    print("→ Running validation on best.pt")
    best = Path(args.project) / args.name / "weights" / "best.pt"
    if not best.exists():
        print("   best.pt not found, using last.pt")
        best = Path(args.project) / args.name / "weights" / "last.pt"
    results = model.val(
        data=data_yaml,
        weights=str(best),
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
    )
    mAP50 = results.box.map50
    mAP = results.box.map
    p = results.box.p
    r = results.box.r
    f = results.box.fitness
    print(f"→ Validation → mAP50: {mAP50:.4f}, mAP: {mAP:.4f}, P: {p:.4f}, R: {r:.4f}, Fitness: {f:.4f}")


if __name__ == "__main__":
    main()
