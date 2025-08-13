#!/usr/bin/env python3
"""
convert_yolo_to_coco.py

Usage:
  python convert_yolo_to_coco.py /path/to/augmented
"""

import os
import glob
import json
import sys
from PIL import Image

# 1) Your two class names here:
CLASS_NAMES = ['class0', 'class1']
CATEGORIES = [
    {"id": idx, "name": name, "supercategory": "none"}
    for idx, name in enumerate(CLASS_NAMES)
]

def convert_split(dataset_dir, split):
    img_dir = os.path.join(dataset_dir, 'images', split)
    lbl_dir = os.path.join(dataset_dir, 'labels', split)
    out_json = os.path.join(dataset_dir, f'{split}_annotations.coco.json')

    if not os.path.isdir(img_dir):
        print(f"[WARN] No images/{split}, skipping.")
        return
    if not os.path.isdir(lbl_dir):
        print(f"[WARN] No labels/{split}, skipping.")
        return
    if os.path.exists(out_json):
        print(f"[SKIP] {out_json} already exists.")
        return

    images = []
    annotations = []
    ann_id = 1

    # Gather all image files
    img_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        img_files.extend(glob.glob(os.path.join(img_dir, ext)))

    for img_id, img_path in enumerate(sorted(img_files), start=1):
        fn = os.path.basename(img_path)
        w, h = Image.open(img_path).size
        images.append({
            "id": img_id,
            "file_name": fn,
            "width": w,
            "height": h
        })

        txt_path = os.path.join(lbl_dir, os.path.splitext(fn)[0] + '.txt')
        if not os.path.isfile(txt_path):
            continue

        with open(txt_path) as f:
            for line in f:
                cls, xc, yc, ww, hh = line.split()
                cls = int(cls)
                x_c = float(xc) * w
                y_c = float(yc) * h
                bw = float(ww) * w
                bh = float(hh) * h
                x0 = x_c - bw/2
                y0 = y_c - bh/2

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls,
                    "bbox": [x0, y0, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0
                })
                ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES
    }

    with open(out_json, 'w') as f:
        json.dump(coco, f, indent=2)
    print(f"[CREATED] {out_json}")

def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    dataset_dir = sys.argv[1]
    for split in ('train', 'val', 'test'):
        convert_split(dataset_dir, split)

if __name__ == '__main__':
    main()
