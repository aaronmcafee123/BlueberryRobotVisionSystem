#!/usr/bin/env python3
"""
train_rf_detr.py

Usage:
    python trainers/train_rf_detr.py \
      --dataset_dir /path/to/augmented \
      --output_dir output \
      --epochs 20 \
      --batch_size 4 \
      --grad_accum_steps 4 \
      --lr 1e-4
"""

import os
import glob
import shutil
import argparse
from rfdetr import RFDETRBase

def prepare_flat_dataset(root, split_map):
    """
    Build <root>/rfdetr_data/{train,valid,test}/
    copying images + the COCO JSON for each split.
    """
    flat_root = os.path.join(root, 'rfdetr_data')
    shutil.rmtree(flat_root, ignore_errors=True)

    for orig_split, target_split in split_map.items():
        src_imgs = os.path.join(root, 'images', orig_split)
        src_json = os.path.join(root, f'{orig_split}_annotations.coco.json')
        dst = os.path.join(flat_root, target_split)
        os.makedirs(dst, exist_ok=True)

        if not os.path.isdir(src_imgs):
            raise FileNotFoundError(f"Missing images directory: {src_imgs}")
        if not os.path.isfile(src_json):
            raise FileNotFoundError(f"Missing COCO JSON: {src_json}")

        # copy images
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            for img in glob.glob(os.path.join(src_imgs, ext)):
                shutil.copy(img, dst)

        # copy annotation JSON into place
        shutil.copy(src_json, os.path.join(dst, '_annotations.coco.json'))

    return flat_root

def main():
    parser = argparse.ArgumentParser(
        description="Flatten YOLO→COCO data and train RF-DETR"
    )
    parser.add_argument('--dataset_dir', required=True,
                        help="root containing images/ and *_annotations.coco.json")
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accum_steps', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    # Map your splits: 'val' JSON → 'valid' folder RF-DETR expects
    split_map = {
        'train': 'train',
        'val':   'valid',
        'test':  'test'
    }

    # Prepare the flattened dataset
    flat_data = prepare_flat_dataset(args.dataset_dir, split_map)

    # Initialize and train RF-DETR
    model = RFDETRBase()
    model.train(
        dataset_dir    = flat_data,
        output_dir     = args.output_dir,
        epochs         = args.epochs,
        batch_size     = args.batch_size,
        grad_accum_steps = args.grad_accum_steps,
        lr             = args.lr
    )

if __name__ == '__main__':
    main()
