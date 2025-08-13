#!/usr/bin/env python3
# keypoint_heatmap_unet_plus.py
# Upgraded PyTorch training script for keypoint-heatmap regression with
# - GroupNorm/BatchNorm + SiLU activations
# - Bilinear upsample + conv (no transposed conv)
# - SE attention on skip connections
# - Optional deep supervision (multi-scale heads)
# - Decoupled input resolution vs heatmap resolution

import os
import argparse
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# -----------------------------------
# Dataset for XML keypoints
# -----------------------------------
class BlueberryKeypointDataset(Dataset):
    """
    Expects an XML with <image name="..."> nodes, each containing a child tag with
    a 'points' attribute like "x,y;...". Uses the FIRST point in that list.

    Returns an input image tensor resized to `input_size` and a target heatmap of
    size `heatmap_size x heatmap_size` centered at the GT keypoint.
    """
    def __init__(self, images_dir, ann_file, input_size=256, heatmap_size=64, sigma=2.0):
        self.images_dir = images_dir
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        tree = ET.parse(ann_file)
        root = tree.getroot()
        self.samples = []  # list of (image_path, (x,y)) in ORIGINAL image coords
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

    def __getitem__(self, idx):
        img_path, (x, y) = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        # Prepare input image
        img_tensor = self.img_transform(img)

        # Prepare target heatmap at heatmap_size
        xs = x * self.heatmap_size / w
        ys = y * self.heatmap_size / h
        xx, yy = np.meshgrid(np.arange(self.heatmap_size), np.arange(self.heatmap_size))
        heatmap = np.exp(-((xx - xs) ** 2 + (yy - ys) ** 2) / (2 * (self.sigma ** 2))).astype(np.float32)
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0)  # (1, Hm, Wm)

        return img_tensor, heatmap_tensor, (w, h), (x, y), img_path

# -----------------------------------
# UNet++ style blocks (upgraded UNet)
# -----------------------------------
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
        # Encoder
        self.enc1 = ConvBNAct(in_ch, base, gn_groups=gn_groups)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBNAct(base, base * 2, gn_groups=gn_groups)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBNAct(base * 2, base * 4, gn_groups=gn_groups)
        self.pool3 = nn.MaxPool2d(2)
        # Bottleneck with dilation (larger RF)
        self.bot = nn.Sequential(
            nn.Conv2d(base * 4, base * 8, 3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(gn_groups, base * 8) if gn_groups else nn.BatchNorm2d(base * 8),
            nn.SiLU(inplace=True),
            nn.Conv2d(base * 8, base * 8, 3, padding=4, dilation=4, bias=False),
            nn.GroupNorm(gn_groups, base * 8) if gn_groups else nn.BatchNorm2d(base * 8),
            nn.SiLU(inplace=True),
        )
        # Decoder
        self.up3 = UpBlock(base * 8, base * 4, base * 4, gn_groups=gn_groups, use_se=use_se)
        self.up2 = UpBlock(base * 4, base * 2, base * 2, gn_groups=gn_groups, use_se=use_se)
        self.up1 = UpBlock(base * 2, base, base, gn_groups=gn_groups, use_se=use_se)
        # Heads (deep supervision)
        self.head1 = nn.Conv2d(base, out_ch, 1)
        self.head2 = nn.Conv2d(base * 2, out_ch, 1)
        self.head3 = nn.Conv2d(base * 4, out_ch, 1)

    def forward(self, x, return_multi=False):
        e1 = self.enc1(x)               # 1/1
        e2 = self.enc2(self.pool1(e1))  # 1/2
        e3 = self.enc3(self.pool2(e2))  # 1/4
        b  = self.bot(self.pool3(e3))   # 1/8

        d3 = self.up3(b, e3)            # 1/4
        d2 = self.up2(d3, e2)           # 1/2
        d1 = self.up1(d2, e1)           # 1/1

        # Heads at multiple scales
        h3 = F.interpolate(self.head3(d3), size=d1.shape[-2:], mode='bilinear', align_corners=False)
        h2 = F.interpolate(self.head2(d2), size=d1.shape[-2:], mode='bilinear', align_corners=False)
        h1 = self.head1(d1)

        if return_multi:
            return [torch.sigmoid(h) for h in (h1, h2, h3)]
        return torch.sigmoid(h1)

# -----------------------------------
# Training + validation + test
# -----------------------------------

def compute_loss(preds, target, criterion, heatmap_size):
    """Preds can be a tensor (N,1,H,W) or a list of such tensors (deep supervision).
    We resize predictions to (heatmap_size, heatmap_size) before computing the loss.
    """
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


def train(args):
    def load_split(split):
        split_dir = os.path.join(args.dataset_dir, split)
        ann_path = os.path.join(split_dir, f"{split}_annotations.xml")
        return BlueberryKeypointDataset(
            split_dir,
            ann_path,
            input_size=args.input_size,
            heatmap_size=args.heatmap_size,
            sigma=args.sigma,
        )

    train_ds = load_split("train")
    val_ds   = load_split("valid")
    test_ds  = load_split("test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True if device.type == 'cuda' else False

    model = UNetPlus(in_ch=3, base=args.base_ch, gn_groups=args.gn_groups, out_ch=1, use_se=not args.no_se).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(args.out_dir, exist_ok=True)

    best_val = float('inf')

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            imgs, heatmaps, *_ = batch
            imgs = imgs.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)

            preds = model(imgs, return_multi=args.deep_supervision)
            loss = compute_loss(preds, heatmaps, criterion, args.heatmap_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        train_loss = total_loss / max(1, len(train_loader))

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs, heatmaps, *_ = batch
                imgs = imgs.to(device, non_blocking=True)
                heatmaps = heatmaps.to(device, non_blocking=True)

                preds = model(imgs, return_multi=args.deep_supervision)
                val_loss += compute_loss(preds, heatmaps, criterion, args.heatmap_size).item()
        val_loss /= max(1, len(val_loader))

        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save latest and best
        torch.save(model.state_dict(), os.path.join(args.out_dir, f"unet_plus_epoch{epoch}.pth"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "unet_plus_best.pth"))

    # ---- Test + visualize ----
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    examples_dir = os.path.join(args.out_dir, 'examples')
    os.makedirs(examples_dir, exist_ok=True)

    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            imgs, heatmaps, sizes, gts, paths = batch
            imgs = imgs.to(device)
            heatmaps = heatmaps.to(device)

            preds = model(imgs, return_multi=args.deep_supervision)
            # For loss, use same path as training
            test_loss += compute_loss(preds, heatmaps, criterion, args.heatmap_size).item()

            # Prepare a single-tensor prediction at heatmap_size for visualization
            if isinstance(preds, list):
                pred_vis = preds[0]
            else:
                pred_vis = preds
            if pred_vis.shape[-1] != args.heatmap_size or pred_vis.shape[-2] != args.heatmap_size:
                pred_vis = F.interpolate(pred_vis, size=(args.heatmap_size, args.heatmap_size), mode='bilinear', align_corners=False)

            # Visualize up to num_visuals per batch
            for j in range(min(args.num_visuals, imgs.size(0))):
                pred = pred_vis[j, 0].detach().cpu().numpy()
                pred_y, pred_x = np.unravel_index(pred.argmax(), pred.shape)

                img_path = paths[j]
                (w, h) = sizes[0][j].item(), sizes[1][j].item() if isinstance(sizes, (list, tuple)) else sizes[j]
                # sizes is returned as a tuple of lists by default collate; to be robust, re-open the image
                img = Image.open(img_path).convert('RGB')
                w, h = img.size

                # Map heatmap coords back to original image coords
                px = pred_x * w / args.heatmap_size
                py = pred_y * h / args.heatmap_size

                # Extract GT from original XML coords
                gt_x, gt_y = gts[0][j].item() if isinstance(gts, (list, tuple)) else gts[j]
                # Re-opened above; draw markers
                draw = ImageDraw.Draw(img)
                draw.ellipse((px - 5, py - 5, px + 5, py + 5), outline='red', width=2)   # prediction
                draw.ellipse((gt_x - 5, gt_y - 5, gt_x + 5, gt_y + 5), outline='green', width=2)  # ground truth

                img.save(os.path.join(examples_dir, f"test_{i * args.batch_size + j}.png"))

    test_loss /= max(1, len(test_loader))
    print(f"Test Loss: {test_loss:.4f}")

# -----------------------------------
# CLI
# -----------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train & evaluate UNetPlus keypoint regressor')
    parser.add_argument('--dataset_dir', required=True, help='Path with train/valid/test splits')
    parser.add_argument('--out_dir',     required=True)

    # Model & training
    parser.add_argument('--input_size',    type=int, default=256, help='Input image size (square)')
    parser.add_argument('--heatmap_size',  type=int, default=64,  help='Heatmap resolution')
    parser.add_argument('--sigma',         type=float, default=2.0)
    parser.add_argument('--base_ch',       type=int, default=64)
    parser.add_argument('--gn_groups',     type=int, default=16, help='Set 0 to use BatchNorm')
    parser.add_argument('--no_se',         action='store_true', help='Disable SE attention on skips')
    parser.add_argument('--deep_supervision', action='store_true', help='Use multi-scale supervision')
    parser.add_argument('--batch_size',    type=int, default=16)
    parser.add_argument('--lr',            type=float, default=1e-3)
    parser.add_argument('--epochs',        type=int, default=20)
    parser.add_argument('--workers',       type=int, default=4)

    # Device
    parser.add_argument('--device', default='cuda', help='cuda | cpu | cuda:0 etc.')

    # Visualization
    parser.add_argument('--num_visuals',   type=int, default=10)

    args = parser.parse_args()

    # If gn_groups <= 0, switch to BatchNorm inside the model by passing None
    if args.gn_groups <= 0:
        args.gn_groups = None

    train(args)
