#!/usr/bin/env python3
"""
YOLO/DETR Evaluation Script (Flexible Folders + Optional COCO JSON + Debug + GT Viz)
- Loads YOLO/RT-DETR/RFDETR `.pt` models via Ultralytics API
- Supports either:
    A) explicit --images/--labels
    B) a dataset --root with --split {train,val,test} where it expects
       <root>/images/<split>/ and <root>/labels/<split>/
- Can auto-generate COCO JSON from YOLO labels OR use a provided --gt_json
- Computes:
    • mAP@0.5:0.95 and mAP@0.5 (overall and per original class)
    • detection-level Precision, Recall, F1, and Accuracy at IoU=0.5 (per class)
    • average inference time (ms)
- Appends metrics to a CSV
- Visualization of predictions (class-colors) + GT boxes in green

DETR-friendly features added:
    • --detr_like turns on top-K filtering per image (no NMS) using --max_det
    • Proper FP accounting on images with no GT for a class
    • Optional confidence-threshold sweep to report best-F1 threshold
    • Configurable IoU for detection metrics via --iou_det
"""
import os
import json
import tempfile
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
# patch for deprecated np.float (older deps)
if not hasattr(np, 'float'):
    np.float = float
import pandas as pd
import time
from PIL import Image, ImageDraw
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ===== Defaults (can be overridden via CLI) =====
ORIG_CLASS_IDS_DEFAULT = [0, 1]               # classes present in YOLO txts
CLASS_COLORS_DEFAULT = {0: 'red', 1: 'blue'}  # prediction box colors per original class


def make_coco_id_map(orig_class_ids):
    """Map original class ids (e.g., [0,1]) to COCO category ids (1..N)."""
    return {c: i + 1 for i, c in enumerate(orig_class_ids)}


def create_coco_from_yolo(images_dir, labels_dir, orig_class_ids):
    images, annotations = [], []
    COCO_ID_MAP = make_coco_id_map(orig_class_ids)
    categories = [{'id': COCO_ID_MAP[c], 'name': str(c)} for c in orig_class_ids]
    ann_id, img_id = 1, 1

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    for img_path in sorted(images_dir.rglob('*')):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        w, h = Image.open(img_path).size
        images.append({'id': img_id, 'file_name': img_path.name, 'width': w, 'height': h})

        label_file = labels_dir / f"{img_path.stem}.txt"
        if label_file.exists():
            with open(label_file, 'r') as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        # skip malformed lines
                        continue
                    orig_cls, x_c, y_c, bw, bh = map(float, parts)
                    x1 = (x_c - bw/2) * w
                    y1 = (y_c - bh/2) * h
                    bw_px, bh_px = bw * w, bh * h
                    coco_cls = COCO_ID_MAP.get(int(orig_cls))
                    if coco_cls is None:
                        # unknown class id in labels; skip
                        continue
                    annotations.append({
                        'id': ann_id,
                        'image_id': img_id,
                        'category_id': coco_cls,
                        'bbox': [x1, y1, bw_px, bh_px],
                        'area': bw_px * bh_px,
                        'iscrowd': 0
                    })
                    ann_id += 1
        img_id += 1

    coco = {
        'info': {'description': 'Auto-generated from YOLO labels', 'version': '1.0'},
        'licenses': [],
        'images': images,
        'annotations': annotations,
        'categories': categories
    }
    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
    json.dump(coco, tmp)
    tmp.close()
    return tmp.name, make_coco_id_map(orig_class_ids)


def compute_coco_map(coco_gt_json, det_json, eval_cat_ids):
    with open(det_json) as f:
        dets = json.load(f)
    if len(dets) == 0:
        return 0.0, 0.0

    coco_gt = COCO(coco_gt_json)
    coco_dt = coco_gt.loadRes(det_json)
    evaler = COCOeval(coco_gt, coco_dt, 'bbox')
    evaler.params.catIds = list(eval_cat_ids)

    # debug
    print(f"[mAP debug] Using categories {evaler.params.catIds}; num images: {len(evaler.params.imgIds)}")
    print(f"[mAP debug] GT anns: {len(coco_gt.dataset.get('annotations', []))}, DT entries: {len(coco_dt.dataset.get('annotations', []))}")

    evaler.evaluate(); evaler.accumulate(); evaler.summarize()
    stats = getattr(evaler, 'stats', [])
    ap   = stats[0] if len(stats) > 0 else 0.0
    ap50 = stats[1] if len(stats) > 1 else 0.0
    return ap, ap50


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]); yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    inter = interW * interH
    union = boxA[2]*boxA[3] + boxB[2]*boxB[3] - inter
    return inter/union if union>0 else 0


def compute_detection_metrics_per_class(gt_ann, detections, coco_id_map, iou_thresh=0.5):
    """
    Compute P/R/F1 per original class at a fixed IoU threshold.
    Counts false positives on images that have no GT for the class as well.
    """
    results = {}
    for orig_cls, coco_cls in coco_id_map.items():
        gt_for_class  = [a for a in gt_ann if a['category_id'] == coco_cls]
        det_for_class = [d for d in detections if d['category_id'] == coco_cls]

        gt_map, pred_map = {}, {}
        for a in gt_for_class:
            gt_map.setdefault(a['image_id'], []).append({'bbox': a['bbox'], 'matched': False})
        for d in det_for_class:
            pred_map.setdefault(d['image_id'], []).append(d)

        # include images that have predictions but no GT for this class
        all_img_ids = set(gt_map.keys()) | set(pred_map.keys())

        TP = FP = FN = 0
        for img_id in all_img_ids:
            gts   = gt_map.get(img_id, [])
            preds = sorted(pred_map.get(img_id, []), key=lambda x: -x['score'])

            for p in preds:
                matched = False
                for gt in gts:
                    if not gt['matched'] and iou(p['bbox'], gt['bbox']) >= iou_thresh:
                        TP += 1; gt['matched'] = True; matched = True; break
                if not matched:
                    FP += 1
            FN += sum(1 for gt in gts if not gt['matched'])

        precision = TP/(TP+FP) if TP+FP>0 else 0.0
        recall    = TP/(TP+FN) if TP+FN>0 else 0.0
        f1        = 2*precision*recall/(precision+recall) if precision+recall>0 else 0.0
        accuracy  = TP/(TP+FP+FN) if TP+FP+FN>0 else 0.0
        results[orig_cls] = {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}
    return results


def sweep_thresholds(gt_ann, dets_all, coco_id_map, thresholds, iou_thresh):
    out = []
    for t in thresholds:
        dets_t = [d for d in dets_all if d['score'] >= t]
        metrics = compute_detection_metrics_per_class(gt_ann, dets_t, coco_id_map, iou_thresh)
        P = float(np.mean([m['precision'] for m in metrics.values()])) if metrics else 0.0
        R = float(np.mean([m['recall']    for m in metrics.values()])) if metrics else 0.0
        F1 = 2*P*R/(P+R) if P+R>0 else 0.0
        out.append((t, P, R, F1))
    return out


def run_evaluation(models, images_dir, labels_dir, csv_path, viz, conf, imgsz,
                   orig_class_ids, class_colors, gt_json_path=None, save_jsons=False,
                   detr_like=False, max_det=300, iou_det=0.5, conf_sweep=None):
    # Default imgsz to 640 if None (some Ultralytics versions require explicit value)
    if imgsz is None:
        imgsz = 640
    # Build or load GT COCO
    if gt_json_path:
        coco_gt = gt_json_path
        print(f"[info] Using provided GT COCO: {coco_gt}")
        # infer coco_id_map from categories (map names back to int original ids if possible)
        coco = json.load(open(coco_gt))
        try:
            cat_map = {int(cat['name']): cat['id'] for cat in coco.get('categories', [])}
        except Exception:
            # if names are not ints, assume order 0..N-1
            cat_map = {i: cat['id'] for i, cat in enumerate(coco.get('categories', []))}
        coco_id_map = cat_map
        gt_ann = coco.get('annotations', [])
    else:
        coco_gt, coco_id_map = create_coco_from_yolo(images_dir, labels_dir, orig_class_ids)
        coco = json.load(open(coco_gt))
        gt_ann = coco['annotations']
        print(f"[info] Auto-generated GT COCO at: {coco_gt}")

    print("[debug] images:", len(coco.get('images', [])))
    print("[debug] GT anns total:", len(gt_ann))
    print("[debug] GT per class:", Counter(a['category_id'] for a in gt_ann))
    print("[debug] eval cat IDs:", list(coco_id_map.values()))

    for model_path in models:
        model = YOLO(model_path)
        detections = []
        total_time = 0.0
        image_count = 0

        for img in coco.get('images', []):
            img_id, fname = img['id'], img['file_name']
            img_path = str(Path(images_dir) / fname)

            t0 = time.time()
            res = model(img_path, conf=conf, imgsz=imgsz, max_det=max_det, verbose=False)[0]
            total_time += time.time() - t0
            image_count += 1

            preds_this_image = []
            if res.boxes is None or len(res.boxes) == 0:
                # still account for images with no preds later
                detections.extend(preds_this_image)
                continue

            # Ultralytics gives xywh in pixels, matching our GT pixel bboxes
            for box, score, cls in zip(res.boxes.xywh.cpu().numpy(),
                                       res.boxes.conf.cpu().numpy(),
                                       res.boxes.cls.cpu().numpy()):
                cx, cy, bw, bh = [float(x) for x in box]
                x1, y1 = cx - bw/2, cy - bh/2
                cat = coco_id_map.get(int(cls), None)
                if cat is None:
                    continue
                preds_this_image.append({
                    'image_id': img_id,
                    'category_id': cat,
                    'bbox': [x1, y1, bw, bh],
                    'score': float(score)
                })

            # DETR-style: keep only top-K by score (no NMS)
            if detr_like and preds_this_image:
                preds_this_image.sort(key=lambda d: d['score'], reverse=True)
                preds_this_image = preds_this_image[:max_det]

            detections.extend(preds_this_image)

        print("[debug] detections total:", len(detections))
        print("[debug] DT per class:", Counter(d['category_id'] for d in detections))

        # Persist dets to a temp file (also optionally save alongside CSV)
        tmpd = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(detections, tmpd)
        tmpd.close()

        if save_jsons:
            out_dir = Path(csv_path).parent
            gt_out = out_dir / 'gt_coco.json' if not gt_json_path else Path(gt_json_path)
            if not gt_json_path:
                with open(gt_out, 'w') as f:
                    json.dump(coco, f, indent=2)
            det_out = out_dir / f"dets_{Path(model_path).stem}.json"
            with open(det_out, 'w') as f:
                json.dump(detections, f, indent=2)
            print(f"[info] Saved GT to {gt_out}")
            print(f"[info] Saved DETS to {det_out}")

        rows = []
        avg_ms = (total_time / image_count) * 1000 if image_count > 0 else 0.0
        rows.append({'model': Path(model_path).stem, 'class': 'all', 'metric': 'avg_inference_ms', 'value': avg_ms})

        # Overall mAP
        ap_all, ap50_all = compute_coco_map(coco_gt, tmpd.name, eval_cat_ids=list(coco_id_map.values()))
        rows.append({'model': Path(model_path).stem, 'class': 'all', 'metric': 'mAP@0.5:0.95', 'value': ap_all})
        rows.append({'model': Path(model_path).stem, 'class': 'all', 'metric': 'mAP@0.5', 'value': ap50_all})

        # Per-class mAP (using each single category)
        for orig_cls, coco_cls in coco_id_map.items():
            ap_c, ap50_c = compute_coco_map(coco_gt, tmpd.name, eval_cat_ids=[coco_cls])
            rows.append({'model': Path(model_path).stem, 'class': orig_cls, 'metric': 'mAP@0.5:0.95', 'value': ap_c})
            rows.append({'model': Path(model_path).stem, 'class': orig_cls, 'metric': 'mAP@0.5', 'value': ap50_c})

        # Detection metrics @ IoU
        det_metrics = compute_detection_metrics_per_class(gt_ann, detections, coco_id_map, iou_thresh=iou_det)
        for orig_cls, m in det_metrics.items():
            for metric_name, val in m.items():
                rows.append({'model': Path(model_path).stem, 'class': orig_cls, 'metric': metric_name, 'value': val})

        # Optional threshold sweep for best F1 (micro-averaged)
        if conf_sweep:
            try:
                ths = [float(x) for x in conf_sweep]
            except Exception:
                ths = []
            if ths:
                sweep = sweep_thresholds(gt_ann, detections, coco_id_map, ths, iou_thresh=iou_det)
                if sweep:
                    best_t, best_p, best_r, best_f1 = max(sweep, key=lambda x: x[3])
                    rows.append({'model': Path(model_path).stem, 'class': 'all', 'metric': 'best_F1_threshold', 'value': best_t})
                    rows.append({'model': Path(model_path).stem, 'class': 'all', 'metric': 'best_F1', 'value': best_f1})
                    rows.append({'model': Path(model_path).stem, 'class': 'all', 'metric': 'best_F1_precision', 'value': best_p})
                    rows.append({'model': Path(model_path).stem, 'class': 'all', 'metric': 'best_F1_recall', 'value': best_r})

        # Write/append CSV
        csv_path = Path(csv_path)
        csv_exists = csv_path.is_file()
        pd.DataFrame(rows).to_csv(csv_path, mode='a' if csv_exists else 'w', header=not csv_exists, index=False)

        # Visualization
        if viz:
            save_dir = csv_path.parent / 'predictions' / Path(model_path).stem
            os.makedirs(save_dir, exist_ok=True)
            # index detections by image
            det_by_img = {}
            for d in detections:
                det_by_img.setdefault(d['image_id'], []).append(d)

            class_colors = class_colors or CLASS_COLORS_DEFAULT

            for img in coco.get('images', []):
                pil = Image.open(Path(images_dir) / img['file_name']).convert('RGB')
                draw = ImageDraw.Draw(pil)

                # draw GT (green)
                for a in [a for a in gt_ann if a['image_id'] == img['id']]:
                    x, y, w, h = a['bbox']
                    draw.rectangle([x, y, x + w, y + h], outline='green', width=2)
                    draw.text((x, y), f"GT:{a['category_id']}", fill='green')

                # draw preds (per original class color)
                inv_map = {v: k for k, v in coco_id_map.items()}
                for d in det_by_img.get(img['id'], []):
                    orig_cls = inv_map.get(d['category_id'], 'unk')
                    color = class_colors.get(orig_cls, 'yellow')
                    x, y, w, h = d['bbox']
                    draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
                    draw.text((x, y), f"{orig_cls}:{d['score']:.2f}", fill=color)

                pil.save(save_dir / f"{img['id']}_{Path(model_path).stem}.jpg")


def parse_args():
    p = argparse.ArgumentParser()

    # Input models
    p.add_argument('--models', nargs='+', required=True)

    # Option A: explicit paths
    p.add_argument('--images', help='Path to images dir for a single split')
    p.add_argument('--labels', help='Path to labels dir for the same split')

    # Option B: dataset root + split
    p.add_argument('--root', help='Dataset root containing images/<split>/ and labels/<split>/')
    p.add_argument('--split', default='test', choices=['train', 'val', 'test'])

    # Optional ground-truth COCO json (if you already have one)
    p.add_argument('--gt_json', help='Path to an existing COCO GT json. If given, skips auto-generation.')

    # Output CSV and viz
    p.add_argument('--csv', default='results.csv')
    p.add_argument('--viz', action='store_true')

    # Inference params
    p.add_argument('--conf', type=float, default=0.001, help='Confidence threshold passed to Ultralytics')
    p.add_argument('--imgsz', type=int, default=640, help='Inference image size for Ultralytics (e.g., 640)')

    # Classes + colors
    p.add_argument('--classes', default='0,1', help='Comma-separated original class ids as they appear in YOLO txts')

    # Save jsons for inspection
    p.add_argument('--save_jsons', action='store_true', help='Persist GT and DET jsons next to the CSV')

    # DETR-related controls
    p.add_argument('--detr_like', action='store_true', help='Enable DETR-style filtering (no NMS; keep top-K by score).')
    p.add_argument('--max_det', type=int, default=300, help='Max detections per image; also used as top-K for DETR-like models.')
    p.add_argument('--iou_det', type=float, default=0.5, help='IoU threshold for detection metrics.')
    p.add_argument('--conf_sweep', default='', help='Optional comma-separated thresholds to sweep for best F1 (e.g. "0.2,0.3,0.4").')

    return p.parse_args()


def main():
    args = parse_args()

    # Resolve images/labels based on provided flags
    if args.images and args.labels:
        images_dir = args.images
        labels_dir = args.labels
    elif args.root:
        images_dir = str(Path(args.root) / 'images' / args.split)
        labels_dir = str(Path(args.root) / 'labels' / args.split)
    else:
        raise SystemExit("Provide either --images and --labels, or --root (with --split)")

    # Parse classes
    try:
        orig_class_ids = [int(x) for x in args.classes.split(',') if x.strip() != '']
    except Exception:
        orig_class_ids = ORIG_CLASS_IDS_DEFAULT

    # Parse sweep list (keep raw strings; run_evaluation will cast)
    conf_sweep = [s.strip() for s in args.conf_sweep.split(',') if s.strip() != ''] if args.conf_sweep else None

    run_evaluation(
        models=args.models,
        images_dir=images_dir,
        labels_dir=labels_dir,
        csv_path=args.csv,
        viz=args.viz,
        conf=args.conf,
        imgsz=args.imgsz,
        orig_class_ids=orig_class_ids,
        class_colors=CLASS_COLORS_DEFAULT,
        gt_json_path=args.gt_json,
        save_jsons=args.save_jsons,
        detr_like=args.detr_like,
        max_det=args.max_det,
        iou_det=args.iou_det,
        conf_sweep=conf_sweep,
    )


if __name__ == '__main__':
    main()
