#!/usr/bin/env python3
"""
Split-aware processor:
- Handles dataset structured as:
    images/{train,val,test}/[images]
    labels/{train,val,test}/[labels]
  (the inner "images"/"labels" subfolders are optional; both layouts work.)
- TRAIN: crop every image into 4 quadrants, then augment the cropped tiles only.
- VAL & TEST: crop every image into 4 quadrants and write COCO JSON.
- Writes YOLO labels for outputs and a COCO JSON per split.

Examples
--------
python process_splits_crop_and_augment.py \
  --images_root /data/ds/images \
  --labels_root /data/ds/labels \
  --output_root /data/out \
  --names_yaml /data/ds/data.yaml \
  --copies 2 --intensity 0.8 --seed 42
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import albumentations as A
import cv2
import yaml

BBoxXYXY = Tuple[float, float, float, float]
Tile = Tuple[int, int, int, int]

def load_names_from_yaml(names_yaml: Optional[Path]) -> Optional[List[str]]:
    if not names_yaml:
        return None
    if not names_yaml.exists():
        print(f"[WARN] names_yaml not found at: {names_yaml}. Falling back to numeric names.")
        return None
    try:
        with open(names_yaml, "r") as f:
            data = yaml.safe_load(f)
        names = data.get("names")
        if isinstance(names, dict):
            ordered = [name for _, name in sorted(names.items(), key=lambda kv: int(kv[0]))]
            return ordered
        if isinstance(names, list):
            return names
        print("[WARN] Could not find 'names' list/dict in YAML; using numeric names.")
    except Exception as e:
        print(f"[WARN] Failed to parse names_yaml: {e}. Using numeric names.")
    return None

def yolo_to_xyxy(box: Tuple[float, float, float, float], w: int, h: int) -> BBoxXYXY:
    cx, cy, bw, bh = box
    cx *= w; cy *= h; bw *= w; bh *= h
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    x1 = max(0.0, min(float(w), x1))
    y1 = max(0.0, min(float(h), y1))
    x2 = max(0.0, min(float(w), x2))
    y2 = max(0.0, min(float(h), y2))
    return x1, y1, x2, y2

def xyxy_to_yolo(box: BBoxXYXY, w: int, h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / w, cy / h, bw / w, bh / h

def box_intersection(a: BBoxXYXY, b: Tile) -> Optional[BBoxXYXY]:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2

def box_area_xyxy(box: BBoxXYXY) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def tiles_for_image(w: int, h: int) -> List[Tile]:
    midx = w // 2
    midy = h // 2
    return [
        (0, 0, midx, midy),
        (midx, 0, w, midy),
        (0, midy, midx, h),
        (midx, midy, w, h),
    ]

def read_yolo_labels(lbl_path: Path) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    out = []
    if not lbl_path.exists():
        return out
    with open(lbl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                c = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
                out.append((c, (x, y, w, h)))
            except Exception:
                continue
    return out

def ensure_split_dirs(root: Path, split: str) -> Tuple[Path, Path]:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, lbl_dir

def resolve_split_dir(root: Path, split: str, prefer_images_subdir: bool) -> Path:
    base = root / split
    with_inner = base / ("images" if prefer_images_subdir else "labels")
    if with_inner.exists() and any(with_inner.iterdir()):
        return with_inner
    return base

def crop_quadrants(
    img, img_path: Path, labels_px: List[Tuple[int, BBoxXYXY]],
    tiles: List[Tile],
    out_img_dir: Path, out_lbl_dir: Path,
    coco_images: List[Dict[str, Any]], coco_annotations: List[Dict[str, Any]],
    img_id_start: int, ann_id_start: int,
    overlap_threshold: float, min_box_area_pixels: float
) -> Tuple[int, int]:
    img_id = img_id_start
    ann_id = ann_id_start
    for tile_idx, tile in enumerate(tiles):
        tx1, ty1, tx2, ty2 = tile
        tile_w = tx2 - tx1
        tile_h = ty2 - ty1
        if tile_w <= 0 or tile_h <= 0:
            continue
        crop = img[ty1:ty2, tx1:tx2]
        new_name = f"{img_path.stem}_q{tile_idx}{img_path.suffix}"
        out_img_path = out_img_dir / new_name
        cv2.imwrite(str(out_img_path), crop)
        yolo_lines: List[str] = []
        for c, full_box in labels_px:
            inter = box_intersection(full_box, tile)
            if inter is None:
                continue
            inter_area = box_area_xyxy(inter)
            orig_area = box_area_xyxy(full_box)
            if orig_area <= 0:
                continue
            if inter_area / orig_area < overlap_threshold:
                continue
            ix1, iy1, ix2, iy2 = inter
            local_x1 = ix1 - tx1
            local_y1 = iy1 - ty1
            local_x2 = ix2 - tx1
            local_y2 = iy2 - ty1
            local_w = max(0.0, local_x2 - local_x1)
            local_h = max(0.0, local_y2 - local_y1)
            if local_w * local_h < min_box_area_pixels:
                continue
            ycx, ycy, ybw, ybh = xyxy_to_yolo((local_x1, local_y1, local_x2, local_y2), tile_w, tile_h)
            ycx = min(1.0, max(0.0, ycx))
            ycy = min(1.0, max(0.0, ycy))
            ybw = min(1.0, max(0.0, ybw))
            ybh = min(1.0, max(0.0, ybh))
            if ybw <= 0 or ybh <= 0:
                continue
            yolo_lines.append(f"{c} {ycx:.6f} {ycy:.6f} {ybw:.6f} {ybh:.6f}")
            coco_bbox = [float(local_x1), float(local_y1), float(local_w), float(local_h)]
            coco_annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(c),
                "bbox": [round(v, 2) for v in coco_bbox],
                "area": round(local_w * local_h, 2),
                "iscrowd": 0,
                "segmentation": [],
            })
            ann_id += 1
        out_lbl_path = out_lbl_dir / f"{img_path.stem}_q{tile_idx}.txt"
        with open(out_lbl_path, "w") as f:
            if yolo_lines:
                f.write("\n".join(yolo_lines) + "\n")
        coco_images.append({
            "id": img_id,
            "file_name": new_name,
            "width": tile_w,
            "height": tile_h,
        })
        img_id += 1
    return img_id, ann_id

def build_rgb_transform(scale: float) -> A.Compose:
    blur_limit = int(3 * scale) + 1
    noise_var = 10.0 * scale
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1*scale, scale_limit=0.15*scale, rotate_limit=int(30*scale), p=0.7),
        A.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2*scale, contrast_limit=0.2*scale, p=0.7),
        A.HueSaturationValue(hue_shift_limit=int(20*scale), sat_shift_limit=int(30*scale), val_shift_limit=int(20*scale), p=0.5),
        A.CLAHE(p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.GaussianBlur(blur_limit=(blur_limit, blur_limit+2), p=0.4),
        A.GaussNoise(var_limit=(noise_var, noise_var*2), p=0.3),
        A.ImageCompression(quality_lower=70, quality_upper=90, p=0.2),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.5, p=0.1),
        A.RandomSunFlare(flare_roi=(0,0,1,0.5), p=0.1),
        A.CoarseDropout(max_holes=8, max_height=int(32*scale), max_width=int(32*scale), min_holes=1, min_height=8, min_width=8, fill_value=random.randint(0,255), p=0.7),
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['category_ids']))

def augment_split(
    images_dir: Path, labels_dir: Path,
    out_img_dir: Path, out_lbl_dir: Path,
    copies: int, intensity: float, seed: int,
    coco_images: List[Dict[str, Any]], coco_annotations: List[Dict[str, Any]],
    img_id_start: int, ann_id_start: int
) -> Tuple[int, int]:
    random.seed(seed)
    scale = max(0.1, min(1.0, intensity))
    rgb_transform = build_rgb_transform(scale)
    img_id = img_id_start
    ann_id = ann_id_start
    image_paths = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}])
    for img_path in image_paths:
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue
        bboxes = []
        category_ids = []
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
                bboxes.append([x, y, w, h])
                category_ids.append(cls)
        if not bboxes:
            continue
        for i in range(copies):
            try:
                augmented = rgb_transform(image=image, bboxes=bboxes, category_ids=category_ids)
                aug_img = augmented['image']
                aug_boxes = augmented['bboxes']
                aug_cls = augmented['category_ids']
                if not aug_boxes:
                    continue
                new_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
                cv2.imwrite(str(out_img_dir / new_name), aug_img)
                with open(out_lbl_dir / f"{img_path.stem}_aug{i}.txt", 'w') as f:
                    for c, box in zip(aug_cls, aug_boxes):
                        f.write(f"{int(c)} " + " ".join(f"{coord:.6f}" for coord in box) + "\n")
                h, w = aug_img.shape[:2]
                coco_images.append({
                    "id": img_id,
                    "file_name": new_name,
                    "width": w,
                    "height": h,
                })
                for c, (x, y, bw, bh) in zip(aug_cls, aug_boxes):
                    cx = x * w; cy = y * h; pw = bw * w; ph = bh * h
                    x1 = cx - pw/2.0; y1 = cy - ph/2.0
                    coco_annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(c),
                        "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(pw), 2), round(float(ph), 2)],
                        "area": round(float(pw*ph), 2),
                        "iscrowd": 0,
                        "segmentation": [],
                    })
                    ann_id += 1
                img_id += 1
            except Exception as e:
                print(f"[ERROR] Augmenting {img_path.name}: {e}")
                continue
    return img_id, ann_id

def process_split(
    split: str,
    images_root: Path, labels_root: Path, output_root: Path,
    crop: bool, augment: bool,
    copies: int, intensity: float, seed: int,
    names: Optional[List[str]],
    overlap_threshold: float, min_box_area_pixels: float
):
    print(f"=== Processing split: {split} | crop={crop} augment={augment} ===")
    in_img_dir = resolve_split_dir(images_root, split, prefer_images_subdir=True)
    in_lbl_dir = resolve_split_dir(labels_root, split, prefer_images_subdir=False)
    out_img_dir, out_lbl_dir = ensure_split_dirs(output_root, split)
    coco_images: List[Dict[str, Any]] = []
    coco_annotations: List[Dict[str, Any]] = []
    img_id = 1
    ann_id = 1
    image_paths = sorted([p for p in in_img_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}])
    if crop:
        for img_path in image_paths:
            lbl_path = in_lbl_dir / f"{img_path.stem}.txt"
            labels = read_yolo_labels(lbl_path)
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] Failed to read image: {img_path}")
                continue
            h, w = img.shape[:2]
            tiles = tiles_for_image(w, h)
            labels_px: List[Tuple[int, BBoxXYXY]] = []
            for c, box in labels:
                xyxy = yolo_to_xyxy(box, w, h)
                labels_px.append((c, xyxy))
            img_id, ann_id = crop_quadrants(
                img, img_path, labels_px, tiles,
                out_img_dir, out_lbl_dir,
                coco_images, coco_annotations,
                img_id, ann_id,
                overlap_threshold, min_box_area_pixels,
            )
    if augment:
        augment_src_img_dir = out_img_dir if crop else in_img_dir
        augment_src_lbl_dir = out_lbl_dir if crop else in_lbl_dir
        img_id, ann_id = augment_split(
            augment_src_img_dir, augment_src_lbl_dir,
            out_img_dir, out_lbl_dir,            copies, intensity, seed,
            coco_images, coco_annotations,
            img_id, ann_id,
        )
    if names is not None:
        categories = [
            {"id": int(i), "name": str(names[i]) if i < len(names) else str(i)}
            for i in sorted({ann["category_id"] for ann in coco_annotations})
        ]
    else:
        categories = [
            {"id": int(i), "name": str(i)}
            for i in sorted({ann["category_id"] for ann in coco_annotations})
        ]
    coco = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    out_json = output_root / f"coco_{split}.json"
    with open(out_json, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"[COCO] {split}: images={len(coco_images)} anns={len(coco_annotations)} -> {out_json}")

def main():
    ap = argparse.ArgumentParser(description="Crop val/test into quadrants, augment train; write YOLO + COCO per split.")
    ap.add_argument("--images_root", required=True, type=Path, help="Root containing train/val/test (optionally with inner 'images' folders)")
    ap.add_argument("--labels_root", required=True, type=Path, help="Root containing train/val/test (optionally with inner 'labels' folders)")
    ap.add_argument("--output_root", required=True, type=Path, help="Output root; creates images/<split>, labels/<split>, coco_<split>.json")
    ap.add_argument("--names_yaml", type=Path, default=None, help="Optional YOLO data.yaml for category names")
    ap.add_argument("--no_augment_train", action="store_true", help="Disable augmentation on train (default: augment enabled)")
    ap.add_argument("--overlap_threshold", type=float, default=0.2, help="Min fraction of original box area overlapping a tile to keep it")
    ap.add_argument("--min_box_area_pixels", type=float, default=4.0, help="Drop boxes smaller than this area (pixels) after clipping")
    ap.add_argument("--copies", type=int, default=1, help="Augmented copies per train image")
    ap.add_argument("--intensity", type=float, default=0.8, help="Augmentation intensity [0..1]")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()
    random.seed(args.seed)
    names = load_names_from_yaml(args.names_yaml)
    args.output_root.mkdir(parents=True, exist_ok=True)
    process_split(
        split="train",
        images_root=args.images_root,
        labels_root=args.labels_root,
        output_root=args.output_root,
        crop=True,
        augment=not args.no_augment_train,
        copies=args.copies,
        intensity=args.intensity,
        seed=args.seed,
        names=names,
        overlap_threshold=args.overlap_threshold,
        min_box_area_pixels=args.min_box_area_pixels,
    )
    for split in ("val", "test"):
        process_split(
            split=split,
            images_root=args.images_root,
            labels_root=args.labels_root,
            output_root=args.output_root,
            crop=True,
            augment=False,
            copies=args.copies,
            intensity=args.intensity,
            seed=args.seed,
            names=names,
            overlap_threshold=args.overlap_threshold,
            min_box_area_pixels=args.min_box_area_pixels,
        )
    print("==== All splits done ====")

if __name__ == "__main__":
    main()
