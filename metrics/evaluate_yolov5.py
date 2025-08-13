#!/usr/bin/env python3
"""
YOLOv5 Evaluation Script (COCO JSON support)
- Loads YOLOv5 `.pt` models via torch.hub (ultralytics/yolov5)
- Reads test images + YOLO-format `.txt` labels for two classes (0 and 1)
- Optional: uses a COCO-style JSON with ground truth annotations for mAP evaluation
  (if provided via `--coco`). Otherwise auto-generates minimal COCO JSON from labels.
- Runs inference using `res.xyxy` for absolute pixel coords
- Computes:
    • mAP@0.5:0.95 (overall and per class)
    • mAP@0.5 (overall and per class)
    • detection-level Precision, Recall, F1, Accuracy at IoU=0.5 (per class)
    • Average inference time per image (ms)
- Appends these metrics, with class labels, to a CSV
- Optional visualization of predictions, colored per original class
"""
import os
import json
import tempfile
import argparse
import time
from pathlib import Path

import numpy as np
if not hasattr(np, 'float'):
    np.float = float
import pandas as pd
from PIL import Image, ImageDraw
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Define your classes and mapping to COCO IDs
ORIG_CLASS_IDS = [0, 1]
CLASS_COLORS = {0: 'red', 1: 'blue'}
COCO_ID_MAP = {0: 1, 1: 2}


def load_or_create_coco_json(images_dir, labels_dir, coco_path=None):
    if coco_path:
        return coco_path
    images, annotations = [], []
    categories = [{'id': COCO_ID_MAP[c], 'name': str(c)} for c in ORIG_CLASS_IDS]
    ann_id, img_id = 1, 1
    for img_path in sorted(Path(images_dir).glob('*')):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        w, h = Image.open(img_path).size
        images.append({'id': img_id, 'file_name': img_path.name, 'width': w, 'height': h})
        label_file = Path(labels_dir) / f"{img_path.stem}.txt"
        if label_file.exists():
            for line in open(label_file):
                cls, xc, yc, bw, bh = map(float, line.split())
                x1 = (xc - bw/2) * w
                y1 = (yc - bh/2) * h
                bw_px, bh_px = bw * w, bh * h
                annotations.append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': COCO_ID_MAP[int(cls)],
                    'bbox': [x1, y1, bw_px, bh_px],
                    'area': bw_px * bh_px,
                    'iscrowd': 0
                })
                ann_id += 1
        img_id += 1
    coco = {'images': images, 'annotations': annotations, 'categories': categories}
    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
    json.dump(coco, tmp)
    tmp.close()
    return tmp.name


def compute_coco_map(coco_gt_json, det_json, orig_class=None):
    """
    Compute COCO-style mAP metrics. If orig_class is specified, filters to that category.
    """
    coco_gt = COCO(coco_gt_json)
    coco_dt = coco_gt.loadRes(det_json)
    evaler = COCOeval(coco_gt, coco_dt, 'bbox')
    # for per-class evaluation, restrict to a single category
    if orig_class is not None:
        coco_id = COCO_ID_MAP[orig_class]
        evaler.params.catIds = [coco_id]
    # ensure all images are evaluated
    evaler.params.imgIds = coco_gt.getImgIds()
    evaler.evaluate()
    evaler.accumulate()
    evaler.summarize()
    stats = evaler.stats  # [AP, AP50, AP75, ...]
    ap = float(stats[0]) if len(stats) > 0 else 0.0
    ap50 = float(stats[1]) if len(stats) > 1 else 0.0
    return ap, ap50


def compute_det_metrics_per_class(gt_ann, detections, iou_thr=0.5):
    results = {}
    for orig in ORIG_CLASS_IDS:
        coco_cls = COCO_ID_MAP[orig]
        gt_objs = [a for a in gt_ann if a['category_id'] == coco_cls]
        pred_objs = [d for d in detections if d['category_id'] == coco_cls]
        # map GT by image
        gt_map = {}
        for a in gt_objs:
            gt_map.setdefault(a['image_id'], []).append({'bbox': a['bbox'], 'matched': False})
        TP = FP = 0
        for d in sorted(pred_objs, key=lambda x: -x['score']):
            matched = False
            for gt in gt_map.get(d['image_id'], []):
                # IoU\                
                xA = max(d['bbox'][0], gt['bbox'][0])
                yA = max(d['bbox'][1], gt['bbox'][1])
                xB = min(d['bbox'][0] + d['bbox'][2], gt['bbox'][0] + gt['bbox'][2])
                yB = min(d['bbox'][1] + d['bbox'][3], gt['bbox'][1] + gt['bbox'][3])
                interW = max(0, xB - xA)
                interH = max(0, yB - yA)
                inter = interW * interH
                union = d['bbox'][2]*d['bbox'][3] + gt['bbox'][2]*gt['bbox'][3] - inter
                iou_val = inter/union if union > 0 else 0
                if not gt['matched'] and iou_val >= iou_thr:
                    TP += 1
                    gt['matched'] = True
                    matched = True
                    break
            if not matched:
                FP += 1
        FN = sum(1 for lst in gt_map.values() for gt in lst if not gt['matched'])
        precision = TP/(TP+FP) if TP+FP>0 else 0.0
        recall = TP/(TP+FN) if TP+FN>0 else 0.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall>0 else 0.0
        accuracy = TP/(TP+FP+FN) if TP+FP+FN>0 else 0.0
        results[orig] = {'precision': precision, 'recall': recall,
                         'f1': f1, 'accuracy': accuracy}
    return results


def evaluate(models, images_dir, labels_dir, coco_json, csv_file, viz):
    # prepare ground truth JSON
    coco_gt = load_or_create_coco_json(images_dir, labels_dir, coco_json)
    coco_data = COCO(coco_gt).dataset['images']
    gt_ann = COCO(coco_gt).dataset['annotations']

    for mpath in models:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=mpath,
                               source='github', trust_repo=True)
        detections = []
        total_time = 0.0
        count = 0
        for img in coco_data:
            img_id, fname = img['id'], img['file_name']
            img_path = str(Path(images_dir) / fname)
            t0 = time.time()
            res = model(img_path)
            total_time += time.time() - t0
            count += 1
            for x1, y1, x2, y2, conf, cls in res.xyxy[0].cpu().numpy():
                detections.append({
                    'image_id': img_id,
                    'category_id': COCO_ID_MAP[int(cls)],
                    'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    'score': float(conf)
                })
        # write detections JSON
        det_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(detections, det_file)
        det_file.close()

        rows = []
        avg_ms = (total_time/count)*1000 if count>0 else 0.0
        rows.append({'model': Path(mpath).stem, 'class': 'all',
                     'metric': 'avg_inference_ms', 'value': avg_ms})
        # overall mAP
        ap_all, ap50_all = compute_coco_map(coco_gt, det_file.name)
        rows.append({'model': Path(mpath).stem, 'class': 'all',
                     'metric': 'mAP@0.5:0.95', 'value': ap_all})
        rows.append({'model': Path(mpath).stem, 'class': 'all',
                     'metric': 'mAP@0.5', 'value': ap50_all})
        # class-wise metrics
        det_metrics = compute_det_metrics_per_class(gt_ann, detections)
        for orig in ORIG_CLASS_IDS:
            ap_c, ap50_c = compute_coco_map(coco_gt, det_file.name, orig)
            rows.append({'model': Path(mpath).stem, 'class': orig,
                         'metric': 'mAP@0.5:0.95', 'value': ap_c})
            rows.append({'model': Path(mpath).stem, 'class': orig,
                         'metric': 'mAP@0.5', 'value': ap50_c})
            for metric, val in det_metrics[orig].items():
                rows.append({'model': Path(mpath).stem, 'class': orig,
                             'metric': metric, 'value': val})
        pd.DataFrame(rows).to_csv(csv_file,
                                  mode='a' if os.path.isfile(csv_file) else 'w',
                                  header=not os.path.isfile(csv_file), index=False)
        if viz:
            out_dir = Path(csv_file).parent / 'predictions' / Path(mpath).stem
            os.makedirs(out_dir, exist_ok=True)
            for img in coco_data:
                pil = Image.open(Path(images_dir) / img['file_name']).convert('RGB')
                draw = ImageDraw.Draw(pil)
                for d in [d for d in detections if d['image_id'] == img['id']]:
                    orig = [k for k,v in COCO_ID_MAP.items() if v == d['category_id']][0]
                    color = CLASS_COLORS[orig]
                    x,y,w,h = d['bbox']
                    draw.rectangle([x,y,x+w,y+h], outline=color, width=2)
                    draw.text((x,y), f"{orig}:{d['score']:.2f}", fill=color)
                pil.save(out_dir/f"{img['id']}_{Path(mpath).stem}.jpg")

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', required=True, help='Paths to YOLOv5 .pt model files')
    p.add_argument('--images', required=True, help='Directory of test images')
    p.add_argument('--labels', required=True, help='Directory of YOLO-format .txt labels')
    p.add_argument('--coco', default=None, help='Optional COCO JSON for GT; if omitted, JSON is auto-generated from labels')
    p.add_argument('--csv', default='results_v5.csv', help='Output CSV file')
    p.add_argument('--viz', action='store_true', help='Save visualized predictions')
    args = p.parse_args()
    evaluate(
        models=args.models,
        images_dir=args.images,
        labels_dir=args.labels,
        coco_json=args.coco,
        csv_file=args.csv,
        viz=args.viz
    )
