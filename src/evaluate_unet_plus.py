#!/usr/bin/env python3
# evaluate_unet_plus.py
# Evaluate a trained UNetPlus keypoint heatmap model on the test set.
# Reports heatmap MSE, localization error metrics, **and mAP** (distance-based),
# plus average inference time per image.
#
# mAP here is defined over distance thresholds τ (normalized by image size):
#   A prediction is a TP if dist(px,py,gt_x,gt_y) ≤ τ * scale(image).
# We compute AP(τ) from the precision-recall curve (sorted by confidence),
# and mAP = mean_τ AP(τ). This is appropriate for single-keypoint per image.
# If you need COCO OKS-AP, see the notes at the bottom and flags in the CLI.
#
# Visuals: predicted (red) and ground-truth (green) keypoints are drawn as circles.
# You can control their size using --marker_radius and --marker_width.
# Collages: add --make_grids to assemble saved visuals into 4x4 (or custom) grids.

import os
import argparse
import time
import xml.etree.ElementTree as ET
from typing import Tuple, List

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ------------------------------
# Dataset (matches training expectations)
# ------------------------------
class BlueberryKeypointDataset(Dataset):
    """
    Reads <split>/<split>_annotations.xml where XML has <image name="..."> children
    with a 'points' attribute like "x,y;...". Uses FIRST point.

    Returns (img_tensor, heatmap_tensor, (w,h), (x,y), img_path).
    """
    def __init__(self, images_dir: str, ann_file: str, input_size: int = 256, heatmap_size: int = 64, sigma: float = 2.0):
        self.images_dir = images_dir
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        tree = ET.parse(ann_file)
        root = tree.getroot()
        self.samples = []  # list[(img_path, (x,y))] in ORIGINAL image coords
        for img_node in root.findall('image'):
            name = img_node.get('name')
            img_path = os.path.join(images_dir, os.path.basename(name))
            kp = None
            for child in img_node:
                if 'points' in child.tag and child.get('points'):
                    pts_str = child.get('points')
                    first = pts_str.split(';')[0]
                    x_str, y_str = first.split(',')
                    kp = (float(x_str), float(y_str))
                    break
            if kp and os.path.exists(img_path):
                self.samples.append((img_path, kp))

        self.img_transform = T.Compose([
            T.Resize((self.input_size, self.input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, (x, y) = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        img_tensor = self.img_transform(img)

        xs = x * self.heatmap_size / w
        ys = y * self.heatmap_size / h
        xx, yy = np.meshgrid(np.arange(self.heatmap_size), np.arange(self.heatmap_size))
        heatmap = np.exp(-((xx - xs) ** 2 + (yy - ys) ** 2) / (2 * (self.sigma ** 2))).astype(np.float32)
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0)

        return img_tensor, heatmap_tensor, (w, h), (x, y), img_path

# ------------------------------
# Model (must match training)
# ------------------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, gn_groups=None):
        super().__init__()
        padding = ks // 2
        if gn_groups:
            Norm = lambda c: nn.GroupNorm(gn_groups, c)
        else:
            Norm = nn.BatchNorm2d
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ks, padding=padding, bias=False),
            Norm(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, ks, padding=padding, bias=False),
            Norm(out_ch),
            nn.SiLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class SE(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, max(1, ch // r), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(1, ch // r), ch, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, gn_groups=None, use_se=True):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.attn = SE(skip_ch) if use_se else nn.Identity()
        self.conv = ConvBNAct(out_ch + skip_ch, out_ch, gn_groups=gn_groups)
    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = self.reduce(x)
        skip = self.attn(skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNetPlus(nn.Module):
    def __init__(self, in_ch=3, base=64, gn_groups=16, out_ch=1, use_se=True):
        super().__init__()
        self.enc1 = ConvBNAct(in_ch, base, gn_groups=gn_groups)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBNAct(base, base * 2, gn_groups=gn_groups)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBNAct(base * 2, base * 4, gn_groups=gn_groups)
        self.pool3 = nn.MaxPool2d(2)
        self.bot = nn.Sequential(
            nn.Conv2d(base * 4, base * 8, 3, padding=2, dilation=2, bias=False),
            (nn.GroupNorm(gn_groups, base * 8) if gn_groups else nn.BatchNorm2d(base * 8)),
            nn.SiLU(inplace=True),
            nn.Conv2d(base * 8, base * 8, 3, padding=4, dilation=4, bias=False),
            (nn.GroupNorm(gn_groups, base * 8) if gn_groups else nn.BatchNorm2d(base * 8)),
            nn.SiLU(inplace=True),
        )
        self.up3 = UpBlock(base * 8, base * 4, base * 4, gn_groups=gn_groups, use_se=use_se)
        self.up2 = UpBlock(base * 4, base * 2, base * 2, gn_groups=gn_groups, use_se=use_se)
        self.up1 = UpBlock(base * 2, base, base, gn_groups=gn_groups, use_se=use_se)
        self.head1 = nn.Conv2d(base, out_ch, 1)
        self.head2 = nn.Conv2d(base * 2, out_ch, 1)
        self.head3 = nn.Conv2d(base * 4, out_ch, 1)

    def forward(self, x, return_multi=False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bot(self.pool3(e3))
        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        h3 = F.interpolate(self.head3(d3), size=d1.shape[-2:], mode='bilinear', align_corners=False)
        h2 = F.interpolate(self.head2(d2), size=d1.shape[-2:], mode='bilinear', align_corners=False)
        h1 = self.head1(d1)
        if return_multi:
            return [torch.sigmoid(h) for h in (h1, h2, h3)]
        return torch.sigmoid(h1)

# ------------------------------
# Utilities
# ------------------------------

def compute_loss(preds, target, criterion, heatmap_size):
    if isinstance(preds, list):
        losses = []
        for p in preds:
            if p.shape[-1] != heatmap_size or p.shape[-2] != heatmap_size:
                p = F.interpolate(p, size=(heatmap_size, heatmap_size), mode='bilinear', align_corners=False)
            losses.append(criterion(p, target))
        return sum(losses) / len(losses)
    else:
        p = preds
        if p.shape[-1] != heatmap_size or p.shape[-2] != heatmap_size:
            p = F.interpolate(p, size=(heatmap_size, heatmap_size), mode='bilinear', align_corners=False)
        return criterion(p, target)


def argmax_2d(heatmap: np.ndarray) -> Tuple[int, int]:
    idx = np.argmax(heatmap)
    y, x = np.unravel_index(idx, heatmap.shape)
    return int(y), int(x)


def pr_ap_from_scores(targets: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    t = targets[order]
    tp = np.cumsum(t)
    fp = np.cumsum(1 - t)
    rec = tp / max(1, t.sum())
    prec = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


def compute_map_distance(records: List[dict], thresholds: List[float], norm: str = 'maxwh'):
    ap_per_t = {}
    for tau in thresholds:
        tgts = []
        scrs = []
        for r in records:
            w, h = r['w'], r['h']
            if norm == 'diag':
                scale = float((w ** 2 + h ** 2) ** 0.5)
            elif norm == 'shorter':
                scale = float(min(w, h))
            else:
                scale = float(max(w, h))
            thr = tau * scale
            tgts.append(1.0 if r['dist'] <= thr else 0.0)
            scrs.append(r['score'])
        tgts = np.array(tgts, dtype=np.float32)
        scrs = np.array(scrs, dtype=np.float32)
        ap_per_t[tau] = pr_ap_from_scores(tgts, scrs)
    mAP = float(np.mean(list(ap_per_t.values()))) if ap_per_t else float('nan')
    return mAP, ap_per_t

# ---- Collage helpers --------------------------------------------------------

def _natural_key(s: str):
    out = []
    num = ''
    for ch in s:
        if ch.isdigit():
            num += ch
        else:
            if num:
                out.append(int(num))
                num = ''
            out.append(ch.lower())
    if num:
        out.append(int(num))
    return out


def build_grids(images_dir: str, out_dir: str, cols: int = 4, rows: int = 4, tile: int = 256, padding: int = 8,
                bg=(0, 0, 0)) -> int:
    """Create contact-sheet style collages from images_dir into out_dir.
    Returns number of collage files written.
    """
    files = [f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        return 0
    files.sort(key=_natural_key)

    per_grid = cols * rows
    total = 0

    # PIL resample fallback
    try:
        resample = Image.Resampling.LANCZOS
    except Exception:
        resample = Image.LANCZOS

    for gi in range(0, len(files), per_grid):
        batch = files[gi:gi + per_grid]
        W = cols * tile + padding * (cols + 1)
        H = rows * tile + padding * (rows + 1)
        canvas = Image.new('RGB', (W, H), bg)
        for idx, fname in enumerate(batch):
            r = idx // cols
            c = idx % cols
            x0 = padding + c * (tile + padding)
            y0 = padding + r * (tile + padding)
            with Image.open(os.path.join(images_dir, fname)) as im:
                im = im.convert('RGB')
                im.thumbnail((tile, tile), resample)
                offx = x0 + (tile - im.width) // 2
                offy = y0 + (tile - im.height) // 2
                canvas.paste(im, (offx, offy))
        out_path = os.path.join(out_dir, f'collage_{cols}x{rows}_{gi // per_grid + 1:03d}.png')
        canvas.save(out_path)
        total += 1
    return total

# ------------------------------
# Evaluation
# ------------------------------

def evaluate(args):
    split_dir = os.path.join(args.dataset_dir, 'test')
    ann_path = os.path.join(split_dir, 'test_annotations.xml')
    test_ds = BlueberryKeypointDataset(split_dir, ann_path, input_size=args.input_size, heatmap_size=args.heatmap_size, sigma=args.sigma)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    device = torch.device(args.device)
    model = UNetPlus(in_ch=3, base=args.base_ch, gn_groups=(None if args.gn_groups <= 0 else args.gn_groups), out_ch=1, use_se=not args.no_se).to(device)

    state = torch.load(args.weights, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    new_state = {}
    for k, v in state.items():
        nk = k.replace('module.', '')
        new_state[nk] = v
    model.load_state_dict(new_state, strict=True)
    model.eval()

    criterion = nn.MSELoss()
    os.makedirs(args.out_dir, exist_ok=True)
    examples_dir = os.path.join(args.out_dir, 'examples_eval')
    if args.save_visuals:
        os.makedirs(examples_dir, exist_ok=True)

    if device.type == 'cuda':
        dummy = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=device)
        with torch.no_grad():
            for _ in range(min(5, len(test_loader))):
                _ = model(dummy, return_multi=args.deep_supervision)
        torch.cuda.synchronize()

    total_loss = 0.0
    total_images = 0
    pixel_errors = []
    pr_records = []

    total_forward_s = 0.0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            imgs, heatmaps, *_ = batch
            bsz = imgs.size(0)
            imgs = imgs.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)

            if device.type == 'cuda':
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                preds = model(imgs, return_multi=args.deep_supervision)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
            else:
                t0 = time.perf_counter()
                preds = model(imgs, return_multi=args.deep_supervision)
                t1 = time.perf_counter()
            total_forward_s += (t1 - t0)

            total_loss += compute_loss(preds, heatmaps, criterion, args.heatmap_size).item() * bsz
            total_images += bsz

            if isinstance(preds, list):
                pred_vis = preds[0]
            else:
                pred_vis = preds
            if pred_vis.shape[-1] != args.heatmap_size or pred_vis.shape[-2] != args.heatmap_size:
                pred_vis = F.interpolate(pred_vis, size=(args.heatmap_size, args.heatmap_size), mode='bilinear', align_corners=False)

            pred_vis_np = pred_vis.detach().cpu().numpy()

            for j in range(bsz):
                global_idx = i * args.batch_size + j
                img_path, (gt_x, gt_y) = test_ds.samples[global_idx]

                with Image.open(img_path) as im:
                    w, h = im.size
                    heat = pred_vis_np[j, 0]
                    py_hm, px_hm = argmax_2d(heat)
                    px = px_hm * w / args.heatmap_size
                    py = py_hm * h / args.heatmap_size

                    dist = float(((px - gt_x) ** 2 + (py - gt_y) ** 2) ** 0.5)
                    score = float(heat.max())
                    pr_records.append({'dist': dist, 'score': score, 'w': w, 'h': h})

                    pixel_errors.append(dist)

                    if args.save_visuals:
                        # Only save up to num_visuals images in total
                        saved_count = len([f for f in os.listdir(examples_dir) if f.lower().endswith((".png",".jpg",".jpeg"))])
                        if saved_count < args.num_visuals:
                            vis = im.convert('RGB')
                            draw = ImageDraw.Draw(vis)
                            r = args.marker_radius
                            wd = args.marker_width
                            # Predicted keypoint (red outline)
                            draw.ellipse((px - r, py - r, px + r, py + r), outline='red', width=wd)
                            # Ground-truth keypoint (green outline)
                            draw.ellipse((gt_x - r, gt_y - r, gt_x + r, gt_y + r), outline='green', width=wd)
                            vis.save(os.path.join(examples_dir, f"eval_{global_idx}.png"))

    mse = total_loss / max(1, total_images)
    pixel_errors = np.array(pixel_errors, dtype=np.float32) if len(pixel_errors) else np.array([np.nan], dtype=np.float32)
    mean_err = float(np.nanmean(pixel_errors))
    median_err = float(np.nanmedian(pixel_errors))

    avg_forward_ms = (total_forward_s / max(1, total_images)) * 1000.0
    throughput = max(1e-9, total_images / total_forward_s) if total_forward_s > 0 else float('nan')

    thresholds = [float(t) for t in args.map_thresholds.split(',') if t]
    mAP, ap_per_t = compute_map_distance(pr_records, thresholds, norm=args.map_norm)

    print("=== Evaluation Results ===")
    print(f"Test images: {total_images}")
    print(f"Heatmap MSE: {mse:.6f}")
    print(f"Pixel error (orig px): mean={mean_err:.3f}, median={median_err:.3f}")
    print(f"Avg inference time: {avg_forward_ms:.3f} ms / image | Throughput: {throughput:.2f} images/s")
    print("-- Distance-based AP per threshold (τ × {} scale) --".format(args.map_norm))
    for tau in thresholds:
        print(f"AP@{tau:.2f}: {ap_per_t[tau]:.4f}")
    print(f"mAP (mean over τ): {mAP:.4f}")

    # Build 4x4 (or custom) collages from saved visuals
    if args.save_visuals and args.make_grids:
        num = build_grids(examples_dir, args.out_dir, cols=args.grid_cols, rows=args.grid_rows,
                          tile=args.grid_tile, padding=args.grid_padding)
        print(f"Saved {num} collage(s) to {args.out_dir}")

    if args.coco_json:
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            import json
        except Exception as e:
            print("pycocotools not available; skipping COCO OKS-AP. Install with `pip install pycocotools`. Error:", e)
            return

        coco = COCO(args.coco_json)
        imgid_by_filename = {img['file_name']: img['id'] for img in coco.loadImgs(coco.getImgIds())}

        preds_coco = []
        for idx, (img_path, _) in enumerate(test_ds.samples):
            fname = os.path.basename(img_path)
            if fname not in imgid_by_filename:
                continue
            img_id = imgid_by_filename[fname]
            r = pr_records[idx]
            pass
        print("NOTE: COCO OKS-AP stub present. To enable, we need to store predicted (px,py) per image and construct 'keypoints' arrays. Modify pr_records to keep px/py and fill preds_coco accordingly.")

# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate UNetPlus on test set with timing and mAP")
    parser.add_argument('--dataset_dir', required=True, help='Path with train/valid/test splits')
    parser.add_argument('--weights',     required=True, help='Path to model weights (.pth)')
    parser.add_argument('--out_dir',     required=True, help='Where to save visuals and logs')

    parser.add_argument('--input_size',   type=int,   default=256)
    parser.add_argument('--heatmap_size', type=int,   default=64)
    parser.add_argument('--sigma',        type=float, default=2.0)

    parser.add_argument('--base_ch',      type=int,   default=32, help='MATCH training base_ch of checkpoint')
    parser.add_argument('--gn_groups',    type=int,   default=16, help='Set <=0 to use BatchNorm')
    parser.add_argument('--no_se',        action='store_true', help='Disable SE attention on skips')
    parser.add_argument('--deep_supervision', action='store_true', help='Use multi-scale head (match training)')

    parser.add_argument('--batch_size',   type=int,   default=16)
    parser.add_argument('--workers',      type=int,   default=4)
    parser.add_argument('--device',       default='cuda', help='cuda | cpu | cuda:0')

    parser.add_argument('--save_visuals', action='store_true', help='Save example predictions')
    parser.add_argument('--num_visuals',  type=int,   default=20, help='Max visuals to save (across dataset)')

    parser.add_argument('--map_thresholds', default='0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50',
                        help='Comma-separated τ values; TP if dist ≤ τ * scale')
    parser.add_argument('--map_norm', choices=['maxwh', 'diag', 'shorter'], default='maxwh',
                        help='Scale for distance normalization: max(w,h), image diagonal, or shorter side')

    parser.add_argument('--coco_json', default='', help='Path to COCO keypoints JSON (test annotations)')
    parser.add_argument('--category_id', type=int, default=1, help='COCO category_id for keypoints')

    # === Visual marker size ===
    parser.add_argument('--marker_radius', type=int, default=25,
                        help='Radius in pixels for drawn keypoints (pred/gt)')
    parser.add_argument('--marker_width', type=int, default=3,
                        help='Stroke width of keypoint circles')

    # === Collage options ===
    parser.add_argument('--make_grids', action='store_true',
                        help='Assemble saved visuals into collages')
    parser.add_argument('--grid_rows', type=int, default=4, help='Rows per collage')
    parser.add_argument('--grid_cols', type=int, default=4, help='Cols per collage')
    parser.add_argument('--grid_tile', type=int, default=256, help='Each tile max side (px)')
    parser.add_argument('--grid_padding', type=int, default=8, help='Padding between tiles (px)')

    args = parser.parse_args()

    if args.gn_groups <= 0:
        args.gn_groups = 0

    evaluate(args)
