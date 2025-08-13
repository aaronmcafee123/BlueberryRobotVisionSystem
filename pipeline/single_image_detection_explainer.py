import os
import cv2
import json
import math
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from ultralytics import YOLO
try:
    import open3d as o3d
except ImportError:
    o3d = None
from read_write_model import read_model

@dataclass
class Det:
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    conf: float
    cls: int

@dataclass
class Pred3D:
    kp3d: Optional[np.ndarray]
    cc3d: Optional[np.ndarray]
    class_id: int

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
            nn.SiLU(inplace=True)
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
            nn.Sigmoid()
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
            nn.GroupNorm(gn_groups, base * 8) if gn_groups else nn.BatchNorm2d(base * 8),
            nn.SiLU(inplace=True),
            nn.Conv2d(base * 8, base * 8, 3, padding=4, dilation=4, bias=False),
            nn.GroupNorm(gn_groups, base * 8) if gn_groups else nn.BatchNorm2d(base * 8),
            nn.SiLU(inplace=True)
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
        b = self.bot(self.pool3(e3))
        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        h3 = F.interpolate(self.head3(d3), size=d1.shape[-2:], mode='bilinear', align_corners=False)
        h2 = F.interpolate(self.head2(d2), size=d1.shape[-2:], mode='bilinear', align_corners=False)
        h1 = self.head1(d1)
        if return_multi:
            return [torch.sigmoid(h) for h in (h1, h2, h3)]
        return torch.sigmoid(h1)

class KeypointDetector:
    def __init__(self, weights: str, device: str, thresh: float, input_size: int = 256, heatmap_size: int = 64, base_ch: Optional[int] = None, gn_groups: Optional[int] = 16, use_se: bool = True, deep_supervision: bool = False):
        self.device = torch.device(device)
        self.thresh = float(thresh)
        self.input_size = int(input_size)
        self.heatmap_size = int(heatmap_size)
        self.deep_supervision = bool(deep_supervision)
        sd = torch.load(weights, map_location=self.device)
        inferred_base = None
        k_w = 'enc1.block.0.weight'
        if k_w in sd and hasattr(sd[k_w], 'shape'):
            inferred_base = int(sd[k_w].shape[0])
        base = int(base_ch) if base_ch is not None else (inferred_base or 64)
        uses_bn = any(k.endswith('running_mean') or k.endswith('running_var') for k in sd.keys())
        if uses_bn:
            gn = None
        else:
            candidates = [32, 16, 8, 4, 2, 1]
            gn = next((g for g in candidates if base % g == 0), 1)
        if gn_groups is not None:
            if gn_groups <= 0:
                gn = None
            else:
                gn = int(gn_groups)
        self.model = UNetPlus(in_ch=3, base=base, gn_groups=gn, out_ch=1, use_se=use_se).to(self.device)
        try:
            self.model.load_state_dict(sd, strict=True)
        except Exception:
            self.model.load_state_dict(sd, strict=False)
        self.model.eval()
        self.tf = T.Compose([
            T.ToPILImage(),
            T.Resize((self.input_size, self.input_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    @staticmethod
    def _to_numpy_heatmap(t: torch.Tensor) -> np.ndarray:
        if t.dim() == 4:
            t = t[0, 0]
        elif t.dim() == 3:
            t = t[0]
        return t.detach().cpu().numpy()
    def _forward_single_heatmap(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.model(x, return_multi=self.deep_supervision)
            if isinstance(out, list):
                out = torch.stack(out, dim=0).mean(dim=0)
            if self.heatmap_size > 0 and (out.shape[-2] != self.heatmap_size or out.shape[-1] != self.heatmap_size):
                out = F.interpolate(out, size=(self.heatmap_size, self.heatmap_size), mode='bilinear', align_corners=False)
            return out
    def detect(self, roi: np.ndarray) -> Tuple[Optional[Tuple[int, int]], float, np.ndarray]:
        if roi is None or roi.size == 0:
            return None, 0.0, None
        roi_h, roi_w = roi.shape[:2]
        inp = self.tf(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
        heat = self._forward_single_heatmap(inp)
        heat_np = self._to_numpy_heatmap(heat)
        conf = float(heat_np.max())
        if conf < self.thresh:
            print(f"[KP] No keypoint above threshold: conf={conf:.3f} < {self.thresh:.3f}")
            return None, conf, heat_np
        py, px = np.unravel_index(int(heat_np.argmax()), heat_np.shape)
        kx = int(px * roi_w / heat_np.shape[1])
        ky = int(py * roi_h / heat_np.shape[0])
        print(f"[KP] Keypoint at ROI px=({kx},{ky}) with conf={conf:.3f}")
        return (kx, ky), conf, heat_np

class YOLODetector:
    def __init__(self, weights: str, conf_thresh: float):
        self.model = YOLO(weights, verbose=False)
        self.conf_thresh = float(conf_thresh)
    def _run(self, img_path: str):
        try:
            return self.model(img_path)
        except TypeError:
            return self.model.predict(img_path)
    def detect(self, img_path: str) -> List[Det]:
        res = self._run(img_path)
        r0 = res[0]
        boxes = getattr(r0, 'boxes', None)
        out: List[Det] = []
        if boxes is None:
            return out
        xyxy = boxes.xyxy
        confs = getattr(boxes, 'conf', None)
        clss = getattr(boxes, 'cls', None)
        if hasattr(xyxy, 'cpu'):
            xyxy = xyxy.cpu().numpy()
        else:
            xyxy = np.asarray(xyxy)
        if confs is not None and hasattr(confs, 'cpu'):
            confs = confs.cpu().numpy()
        if clss is not None and hasattr(clss, 'cpu'):
            clss = clss.cpu().numpy()
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(int, xyxy[i].tolist())
            conf = float(confs[i]) if confs is not None else 1.0
            cls = int(clss[i]) if clss is not None else 0
            if conf < self.conf_thresh:
                continue
            out.append(Det((x1, y1, x2, y2), ((x1 + x2) // 2, (y1 + y2) // 2), conf, cls))
        print(f"[YOLO] {len(out)} detections above conf>={self.conf_thresh}")
        return out

class CircleDetector:
    def __init__(self, p: Dict):
        self.p = p
    def detect(self, roi: np.ndarray) -> Tuple[List[Tuple[int, int, int]], np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if self.p.get('median_blur', 5) > 0:
            gray = cv2.medianBlur(gray, self.p['median_blur'])
        gray_eq = cv2.equalizeHist(gray) if self.p.get('equalize', False) else gray
        edges = cv2.Canny(gray_eq, self.p['canny1'], self.p['canny2'])
        if self.p.get('dilate_iter', 0) > 0:
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=self.p['dilate_iter'])
        src = gray_eq if self.p.get('hough_on_gray', False) else edges
        circles = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, dp=self.p['dp'], minDist=self.p['minDist'], param1=self.p['param1'], param2=self.p['param2'], minRadius=self.p['minRadius'], maxRadius=self.p['maxRadius'])
        circles_list = [] if circles is None else [(int(x), int(y), int(r)) for x, y, r in circles[0]]
        print(f"[Hough] Found {len(circles_list)} circle candidates")
        return circles_list, src, edges

def select_best_circle(circles: List[Tuple[int, int, int]], roi_shape: Tuple[int, int, int], w_radius: float, w_center: float, w_edge: float) -> Optional[Tuple[int, int, int]]:
    if not circles:
        print("[Hough] No circles to select from")
        return None
    h, w = roi_shape[:2]
    cxr, cyr = w / 2.0, h / 2.0
    dmax = math.hypot(cxr, cyr)
    rmax = min(h, w) / 2.0
    best, best_score = None, -1e9
    for cx, cy, r in circles:
        R = r / (rmax + 1e-6)
        D = 1.0 - (math.hypot(cx - cxr, cy - cyr) / (dmax + 1e-6))
        margin = min(cx, w - cx, cy, h - cy)
        edge_frac = max(0.0, 1.0 - margin / (r + 1e-6))
        score = w_radius * R + w_center * D - w_edge * edge_frac
        if score > best_score:
            best_score, best = score, (cx, cy, r)
    if best is not None:
        print(f"[Hough] Best circle center=({best[0]},{best[1]}), r={best[2]}")
    return best

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def draw_bbox_overlay(img: np.ndarray, det: Det) -> np.ndarray:
    x1, y1, x2, y2 = det.bbox
    out = img.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(out, f"conf={det.conf:.2f} cls={det.cls}", (x1, max(0, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out

def render_hough_candidates(roi_shape: Tuple[int, int, int], circles: List[Tuple[int, int, int]]) -> np.ndarray:
    h, w = roi_shape[:2]
    canvas = np.zeros((h, w), dtype=np.uint8)
    for (cx, cy, r) in circles:
        cv2.circle(canvas, (int(cx), int(cy)), int(r), 255, 1)
        cv2.circle(canvas, (int(cx), int(cy)), 2, 255, -1)
    return canvas

def overlay_best_circle(roi: np.ndarray, best: Optional[Tuple[int, int, int]]) -> np.ndarray:
    out = roi.copy()
    if best is None:
        cv2.putText(out, 'NO CIRCLE', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return out
    cx, cy, r = best
    cv2.circle(out, (cx, cy), r, (0, 255, 255), 2)
    cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)
    return out

def overlay_kp_and_center(roi: np.ndarray, kp: Optional[Tuple[int, int]], center: Optional[Tuple[int, int]]) -> np.ndarray:
    out = roi.copy()
    if center is not None:
        cv2.circle(out, center, 3, (0, 0, 255), -1)
    if kp is not None:
        cv2.circle(out, kp, 5, (255, 0, 0), -1)
    if kp is None:
        cv2.putText(out, 'NO KP', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    if center is None:
        cv2.putText(out, 'NO CIRCLE', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out

def overlay_direction(roi: np.ndarray, kp: Optional[Tuple[int, int]], center: Optional[Tuple[int, int]]) -> np.ndarray:
    out = overlay_kp_and_center(roi, kp, center)
    # Arrow should point from keypoint -> center (kp -> center)
    if kp is not None and center is not None:
        cv2.arrowedLine(out, kp, center, (0, 255, 0), 3, tipLength=0.25)
    return out

class CameraSystem:
    def __init__(self):
        self.cameras = {}
        self.images = {}
        self.points3D = {}
        self.dense_pcd = None
        self.dense_pts = None
        self.dense_kd = None
        self.dense_aabb = None
        self.scene_diag = 1.0
        self.dense_proj_max_pix = 4.0
        self.dense_max_proj = 300000
        self.ray_radius = 0.02
        self.ray_samples = 24
        self.ray_near_frac = 0.02
        self.ray_far_frac = 0.98
        self._dense_proj_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    def load(self, model_dir: str):
        cams, imgs, pts = read_model(path=model_dir, ext='.bin')
        self.cameras = cams
        self.images = imgs
        self.points3D = pts
        print(f"[COLMAP] Loaded {len(self.cameras)} cams, {len(self.images)} images, {len(self.points3D)} 3D pts")
    def configure_dense(self, dense_proj_max_pix: float = 4.0, dense_max_proj: int = 300000, ray_radius: float = 0.02, ray_samples: int = 24, ray_near_frac: float = 0.02, ray_far_frac: float = 0.98):
        self.dense_proj_max_pix = float(dense_proj_max_pix)
        self.dense_max_proj = int(dense_max_proj)
        self.ray_radius = float(ray_radius)
        self.ray_samples = int(ray_samples)
        self.ray_near_frac = float(ray_near_frac)
        self.ray_far_frac = float(ray_far_frac)
        print(f"[DenseCfg] proj_max_pix={self.dense_proj_max_pix}, max_proj={self.dense_max_proj}, ray_r={self.ray_radius}, samples={self.ray_samples}")
    def load_dense(self, pcd_path: Optional[str], voxel: float = 0.0):
        if o3d is None or pcd_path is None or not os.path.exists(pcd_path):
            print("[Dense] No dense point cloud loaded")
            return
        try:
            pcd = o3d.io.read_point_cloud(pcd_path)
            if voxel and voxel > 0.0:
                pcd = pcd.voxel_down_sample(voxel_size=float(voxel))
            self.dense_pcd = pcd
            self.dense_pts = np.asarray(pcd.points)
            if self.dense_pts.size == 0:
                print("[Dense] Loaded PCD is empty")
                self.dense_pcd = None
                self.dense_pts = None
                self.dense_kd = None
                return
            self.dense_kd = o3d.geometry.KDTreeFlann(pcd)
            self.dense_aabb = pcd.get_axis_aligned_bounding_box()
            self.scene_diag = float(np.linalg.norm(self.dense_aabb.get_extent()))
            print(f"[Dense] Loaded {self.dense_pts.shape[0]} points, scene_diag={self.scene_diag:.3f}")
        except Exception as e:
            print(f"[Dense] Failed to read PCD: {e}")
    def _img_entry(self, name: str):
        for im in self.images.values():
            if im.name == name:
                return im
        print(f"[COLMAP] Image entry not found for name={name}")
        return None
    @staticmethod
    def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
        w, x, y, z = qvec
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
        ], dtype=np.float64)
    def _image_pose(self, im) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        R = self._qvec_to_rotmat(im.qvec)
        t = im.tvec.reshape(3)
        C = -R.T @ t
        return R, t, C
    def _intrinsics(self, cam) -> Tuple[float, float, float, float, Optional[Tuple[float, ...]]]:
        model = cam.model
        p = cam.params
        fx = fy = 1.0
        cx = cy = 0.0
        dist = None
        if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            f, cx, cy = p[0], p[1], p[2]
            fx = fy = f
            if model == "SIMPLE_RADIAL" and len(p) >= 4:
                dist = (p[3],)
        elif model in ("PINHOLE", "OPENCV"):
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
            if model == "OPENCV" and len(p) >= 8:
                dist = (p[4], p[5], p[6], p[7])
        elif model == "RADIAL":
            f, cx, cy, k1, k2 = p[0], p[1], p[2], p[3], p[4]
            fx = fy = f
            dist = (k1, k2)
        else:
            if len(p) >= 3:
                f, cx, cy = p[0], p[1], p[2]
                fx = fy = f
        return float(fx), float(fy), float(cx), float(cy), dist
    @staticmethod
    def _apply_distortion(x: np.ndarray, y: np.ndarray, dist: Optional[Tuple[float, ...]]):
        if dist is None:
            return x, y
        k1 = dist[0] if len(dist) > 0 else 0.0
        k2 = dist[1] if len(dist) > 1 else 0.0
        p1 = dist[2] if len(dist) > 2 else 0.0
        p2 = dist[3] if len(dist) > 3 else 0.0
        r2 = x*x + y*y
        radial = 1.0 + k1*r2 + k2*r2*r2
        x_d = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
        y_d = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y
        return x_d, y_d
    def _project_dense_for_image(self, img_name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.dense_pts is None:
            return None
        if img_name in self._dense_proj_cache:
            return self._dense_proj_cache[img_name]
        im = self._img_entry(img_name)
        if im is None:
            return None
        cam = self.cameras.get(im.camera_id, None)
        if cam is None:
            print(f"[Proj] Camera not found for image")
            return None
        R, t, _ = self._image_pose(im)
        fx, fy, cx, cy, dist = self._intrinsics(cam)
        Xw = self.dense_pts
        if Xw.shape[0] > self.dense_max_proj:
            idx = np.linspace(0, Xw.shape[0]-1, self.dense_max_proj).astype(np.int64)
            Xw = Xw[idx]
            back_idx = idx
        else:
            back_idx = np.arange(Xw.shape[0])
        Xc = (R @ Xw.T + t.reshape(3,1)).T
        z = Xc[:, 2]
        valid = z > 1e-6
        if not np.any(valid):
            print("[Proj] No points in front of camera")
            return None
        Xc = Xc[valid]
        back_idx = back_idx[valid]
        x = Xc[:, 0] / Xc[:, 2]
        y = Xc[:, 1] / Xc[:, 2]
        x, y = self._apply_distortion(x, y, dist)
        u = fx * x + cx
        v = fy * y + cy
        uv = np.stack([u, v], axis=1)
        self._dense_proj_cache[img_name] = (uv, back_idx)
        return self._dense_proj_cache[img_name]
    def _lift_with_dense_projection(self, pix: Tuple[int,int], img_name: str) -> Optional[np.ndarray]:
        proj = self._project_dense_for_image(img_name)
        if proj is None:
            return None
        uv, idx = proj
        p = np.array(pix, dtype=np.float64)
        d2 = np.sum((uv - p.reshape(1,2))**2, axis=1)
        j = int(np.argmin(d2))
        pix_err = float(np.sqrt(d2[j]))
        if pix_err <= self.dense_proj_max_pix:
            k = int(idx[j])
            print(f"[LiftProj] Hit with reproj_err={pix_err:.2f} px")
            return np.asarray(self.dense_pts[k])
        print(f"[LiftProj] No hit within {self.dense_proj_max_pix}px (best={pix_err:.2f}px)")
        return None
    def _lift_with_ray(self, pix: Tuple[int,int], img_name: str) -> Optional[np.ndarray]:
        if self.dense_pts is None or self.dense_kd is None:
            print("[LiftRay] Dense KD-tree not available")
            return None
        im = self._img_entry(img_name)
        if im is None:
            return None
        cam = self.cameras.get(im.camera_id, None)
        if cam is None:
            print("[LiftRay] Camera not found")
            return None
        R, t, C = self._image_pose(im)
        fx, fy, cx, cy, _ = self._intrinsics(cam)
        u, v = float(pix[0]), float(pix[1])
        x = (u - cx) / fx
        y = (v - cy) / fy
        dir_cam = np.array([x, y, 1.0], dtype=np.float64)
        dir_cam = dir_cam / (np.linalg.norm(dir_cam) + 1e-12)
        dir_world = R.T @ dir_cam
        near = self.ray_near_frac * self.scene_diag
        far = self.ray_far_frac * self.scene_diag
        ts = np.linspace(near, far, num=self.ray_samples)
        candidates = set()
        for t_s in ts:
            p_s = C + t_s * dir_world
            k, idx, _ = self.dense_kd.search_radius_vector_3d(p_s, self.ray_radius)
            if k > 0:
                for ii in idx:
                    candidates.add(int(ii))
        if not candidates:
            print("[LiftRay] No candidates along ray")
            return None
        Pw = self.dense_pts[list(candidates)]
        vcp = Pw - C.reshape(1,3)
        t_param = vcp @ dir_world
        proj = np.outer(t_param, dir_world)
        perp = vcp - proj
        dists = np.linalg.norm(perp, axis=1)
        j = int(np.argmin(dists))
        if float(dists[j]) <= self.ray_radius:
            print(f"[LiftRay] Hit with perp_dist={float(dists[j]):.3f}")
            return Pw[j]
        print(f"[LiftRay] Closest perp_dist={float(dists[j]):.3f} > ray_radius={self.ray_radius}")
        return None
    def _ray_dir_cam(self, pix: Tuple[int,int], cam) -> np.ndarray:
        fx, fy, cx, cy, _ = self._intrinsics(cam)
        u, v = float(pix[0]), float(pix[1])
        x = (u - cx) / fx
        y = (v - cy) / fy
        d = np.array([x, y, 1.0], dtype=np.float64)
        return d / (np.linalg.norm(d) + 1e-12)
    def correct_sphere_center(self, circ2d: Tuple[int,int], S_world: np.ndarray, img_name: str, circle_radius_px: Optional[float]) -> np.ndarray:
        im = self._img_entry(img_name)
        if im is None or circle_radius_px is None or circle_radius_px <= 0:
            print("[Center] Missing image entry or invalid pixel radius; using surface point as center")
            return S_world
        cam = self.cameras.get(im.camera_id, None)
        if cam is None:
            print("[Center] Camera not found; using surface point as center")
            return S_world
        R, t, _ = self._image_pose(im)
        fx, fy, cx, cy, _ = self._intrinsics(cam)
        f = 0.5 * (fx + fy)
        dir_cam = self._ray_dir_cam(circ2d, cam)  # unit
        S_cam = R @ S_world + t
        s = float(np.dot(S_cam, dir_cam))  # distance along ray to surface
        dir_z = float(dir_cam[2])
        den = (f - circle_radius_px * dir_z)
        if abs(den) <= 1e-9:
            print("[Center] Ill-conditioned solve (den≈0); using surface point as center")
            return S_world
        R_est = (circle_radius_px * s * dir_z) / den
        if not np.isfinite(R_est) or R_est <= 0:
            print(f"[Center] Invalid radius estimate R={R_est}; using surface point as center")
            return S_world
        C_cam = (s + R_est) * dir_cam
        C_world = R.T @ (C_cam - t)
        shift = float(np.linalg.norm(C_world - S_world))
        print(f"[Center] Solved sphere center: R≈{R_est:.4f}, shift={shift:.4f}")
        return C_world
    def vec_and_points_from_2D_pair(self, kp2d: Tuple[int, int], circ2d: Tuple[int, int], img_name: str, circle_radius_px: Optional[float] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        print(f"[Lift] kp2d={kp2d}, circ2d={circ2d}, img={img_name}")
        im = self._img_entry(img_name)
        if im is None:
            return None, None, None
        feats = im.xys
        ids = im.point3D_ids
        p_kp = p_cc = None
        # Sparse
        if feats.size > 0:
            kp_id = ids[np.argmin(np.linalg.norm(feats - np.array(kp2d), axis=1))]
            cc_id = ids[np.argmin(np.linalg.norm(feats - np.array(circ2d), axis=1))]
            if kp_id >= 0 and kp_id in self.points3D:
                p_kp = self.points3D[kp_id].xyz
                print("[Sparse] Found kp3d via track")
            if cc_id >= 0 and cc_id in self.points3D:
                p_cc = self.points3D[cc_id].xyz
                print("[Sparse] Found cc3d via track")
        # Dense proj
        if p_kp is None:
            p_kp = self._lift_with_dense_projection(kp2d, img_name)
        if p_cc is None:
            p_cc = self._lift_with_dense_projection(circ2d, img_name)
        # Ray
        if p_kp is None:
            p_kp = self._lift_with_ray(kp2d, img_name)
        if p_cc is None:
            p_cc = self._lift_with_ray(circ2d, img_name)
        if p_kp is None:
            print("[Lift] FAILED to lift keypoint to 3D")
        if p_cc is None:
            print("[Lift] FAILED to lift circle center to 3D")
        if p_kp is None or p_cc is None:
            return None, p_kp, p_cc
        # Correct center from surface -> geometric
        p_cc_corrected = self.correct_sphere_center(circ2d, p_cc, img_name, circle_radius_px)
        v = p_cc_corrected - p_kp
        n = np.linalg.norm(v)
        if not np.isfinite(n) or n == 0:
            print("[Vec] Degenerate vector (n=0 or NaN)")
            return None, p_kp, p_cc_corrected
        print(f"[Vec] Orientation vector computed, length={n:.4f}")
        return v / n, p_kp, p_cc_corrected

class Visualizer3D:
    def __init__(self, cams: CameraSystem, pcd_path: Optional[str] = None, point_size_mult: float = 0.5, arrow_width_mult: float = 0.1, arrow_length: float = 0.0, arrow_radius: float = 0.0, arrow_mode: str = 'line', line_width: float = 1.0):
        self.cams = cams
        self.pcd_path = pcd_path
        self.point_size_mult = point_size_mult
        self.arrow_width_mult = float(arrow_width_mult)
        self.arrow_length = float(arrow_length)
        self.arrow_radius = float(arrow_radius)
        self.arrow_mode = str(arrow_mode)
        self.line_width = float(line_width)
        self._scene_diag = 1.0
    def _load_pointcloud(self):
        if o3d is None:
            return None
        if self.pcd_path and os.path.exists(self.pcd_path):
            try:
                pc = o3d.io.read_point_cloud(self.pcd_path)
                if not pc.is_empty():
                    return pc
            except Exception:
                pass
        all_pts = np.array([p.xyz for p in self.cams.points3D.values()]) if len(self.cams.points3D) > 0 else np.zeros((0, 3))
        if all_pts.size == 0:
            return None
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(all_pts)
        pc.colors = o3d.utility.Vector3dVector(np.tile([[0.5, 0.5, 0.5]], (all_pts.shape[0], 1)))
        return pc
    @staticmethod
    def _get_rotation_from_z(v: np.ndarray) -> np.ndarray:
        v = v / (np.linalg.norm(v) + 1e-12)
        z = np.array([0.0, 0.0, 1.0])
        axis = np.cross(z, v)
        s = np.linalg.norm(axis)
        if s < 1e-12:
            dot = float(np.dot(z, v))
            if dot > 0.0:
                return np.eye(3)
            return o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]) * np.pi)
        axis /= s
        angle = math.acos(np.clip(float(np.dot(z, v)), -1.0, 1.0))
        return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
    def _arrow_dims(self) -> Tuple[float, float, float]:
        if self.arrow_radius > 0:
            cyl_r = float(self.arrow_radius)
        else:
            base = max(self._scene_diag * 0.001, 1e-5)
            cyl_r = base * float(self.arrow_width_mult)
        cone_r = cyl_r * 1.2
        cone_len = self._scene_diag * 0.02
        return cyl_r, cone_r, cone_len
    def _make_arrow(self, start: np.ndarray, end: np.ndarray, color: Tuple[float, float, float]):
        v = end - start
        L = float(np.linalg.norm(v))
        if L < 1e-9:
            return None
        cyl_r, cone_r, cone_len = self._arrow_dims()
        cone_len = min(cone_len, 0.5 * L)
        cyl_h = max(L - cone_len, L * 0.66)
        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=cyl_r, cone_radius=cone_r, cylinder_height=cyl_h, cone_height=cone_len, resolution=20, cylinder_split=1, cone_split=1)
        R = self._get_rotation_from_z(v)
        arrow.rotate(R, center=(0, 0, 0))
        arrow.translate(start)
        arrow.compute_vertex_normals()
        arrow.paint_uniform_color(np.asarray(color))
        return arrow
    def show(self, preds: List[Pred3D]):
        if o3d is None:
            print("[o3d] Open3D not available")
            return
        geoms = []
        pc = self._load_pointcloud()
        if pc is not None and not pc.is_empty():
            geoms.append(pc)
            aabb = pc.get_axis_aligned_bounding_box()
            self._scene_diag = float(np.linalg.norm(aabb.get_extent()))
        else:
            self._scene_diag = 1.0
        L = self.arrow_length if self.arrow_length > 0 else 0.045 * self._scene_diag
        added = 0
        if self.arrow_mode == 'line':
            pts = []
            lines = []
            colors = []
            idx = 0
            for p in preds:
                if p.class_id == 1:
                    continue
                if p.kp3d is None or p.cc3d is None:
                    continue
                v = p.cc3d - p.kp3d
                n = float(np.linalg.norm(v))
                if n <= 1e-9:
                    continue
                u = v / n
                start = p.kp3d
                end = p.kp3d + u * L
                pts.append(start)
                pts.append(end)
                lines.append([idx, idx + 1])
                colors.append([0.0, 1.0, 0.0])
                idx += 2
                added += 1
            if pts:
                ls = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(np.asarray(pts)),
                    lines=o3d.utility.Vector2iVector(np.asarray(lines))
                )
                ls.colors = o3d.utility.Vector3dVector(np.asarray(colors))
                geoms.append(ls)
        else:
            for p in preds:
                if p.class_id == 1:
                    continue
                if p.kp3d is None or p.cc3d is None:
                    continue
                v = p.cc3d - p.kp3d
                n = float(np.linalg.norm(v))
                if n <= 1e-9:
                    continue
                u = v / n
                end = p.kp3d + u * L
                mesh = self._make_arrow(p.kp3d, end, color=(0.0, 1.0, 0.0))
                if mesh is not None:
                    geoms.append(mesh)
                    added += 1
        print(f"[o3d] Will render {added} orientation primitives")
        if not geoms:
            print("[o3d] No geometries to render")
            return
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for _g in geoms:
            vis.add_geometry(_g)
        opt = vis.get_render_option()
        opt.point_size = max(0.1, opt.point_size * float(self.point_size_mult))
        if hasattr(opt, 'line_width'):
            opt.line_width = float(self.line_width)
        vis.run()
        vis.destroy_window()

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--yolo_weights', required=True)
    ap.add_argument('--unet_weights', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--conf_thresh', type=float, default=0.6)
    ap.add_argument('--kp_input_size', type=int, default=256)
    ap.add_argument('--kp_heatmap_size', type=int, default=64)
    ap.add_argument('--kp_gn_groups', type=int, default=16)
    ap.add_argument('--kp_no_se', action='store_true')
    ap.add_argument('--kp_deep_supervision', action='store_true')
    ap.add_argument('--kp_thresh', type=float, default=0.3)
    ap.add_argument('--canny1', type=int, default=100)
    ap.add_argument('--canny2', type=int, default=200)
    ap.add_argument('--dilate_iter', type=int, default=0)
    ap.add_argument('--dp', type=float, default=1.2)
    ap.add_argument('--minDist', type=int, default=15)
    ap.add_argument('--param1', type=int, default=100)
    ap.add_argument('--param2', type=int, default=5)
    ap.add_argument('--minRadius', type=int, default=12)
    ap.add_argument('--maxRadius', type=int, default=90)
    ap.add_argument('--equalize', action='store_true')
    ap.add_argument('--median_blur', type=int, default=5)
    ap.add_argument('--hough_on_gray', action='store_true')
    ap.add_argument('--w_radius', type=float, default=0.6)
    ap.add_argument('--w_center', type=float, default=0.4)
    ap.add_argument('--w_edge', type=float, default=0.5)
    ap.add_argument('--model_dir', type=str, required=True)
    ap.add_argument('--pcd_path', type=str, default=None)
    ap.add_argument('--dense_voxel', type=float, default=0.0)
    ap.add_argument('--dense_proj_max_pix', type=float, default=4.0)
    ap.add_argument('--dense_max_proj', type=int, default=300000)
    ap.add_argument('--ray_radius', type=float, default=0.02)
    ap.add_argument('--ray_samples', type=int, default=24)
    ap.add_argument('--ray_near_frac', type=float, default=0.02)
    ap.add_argument('--ray_far_frac', type=float, default=0.98)
    ap.add_argument('--point_size_mult', type=float, default=0.5)
    ap.add_argument('--show_o3d', action='store_true')
    ap.add_argument('--save_pointcloud', type=str, default=None)
    ap.add_argument('--arrow_samples', type=int, default=50)
    ap.add_argument('--arrow_width_mult', type=float, default=0.1)
    ap.add_argument('--arrow_length', type=float, required=True)
    ap.add_argument('--arrow_radius', type=float, default=0.0)
    ap.add_argument('--arrow_mode', choices=['line','mesh'], default='line')
    ap.add_argument('--line_width', type=float, default=10.0)
    ap.add_argument('--save_lines', type=str, default=None)
    return ap.parse_args()

def main():
    args = parse_args()
    out_root = Path(args.out_dir)
    ensure_dir(out_root)
    yolo = YOLODetector(args.yolo_weights, args.conf_thresh)
    kpdet = KeypointDetector(args.unet_weights, args.device, args.kp_thresh, input_size=args.kp_input_size, heatmap_size=args.kp_heatmap_size, base_ch=None, gn_groups=(None if args.kp_gn_groups is None else int(args.kp_gn_groups)), use_se=(not args.kp_no_se), deep_supervision=args.kp_deep_supervision)
    circle = CircleDetector({'canny1': args.canny1, 'canny2': args.canny2, 'dilate_iter': args.dilate_iter, 'dp': args.dp, 'minDist': args.minDist, 'param1': args.param1, 'param2': args.param2, 'minRadius': args.minRadius, 'maxRadius': args.maxRadius, 'equalize': args.equalize, 'median_blur': args.median_blur, 'hough_on_gray': args.hough_on_gray})
    img_path = args.image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    dets = yolo.detect(img_path)
    if not dets:
        print("[Main] No detections above threshold.")
        return
    cams = CameraSystem()
    cams.load(args.model_dir)
    cams.configure_dense(dense_proj_max_pix=args.dense_proj_max_pix, dense_max_proj=args.dense_max_proj, ray_radius=args.ray_radius, ray_samples=args.ray_samples, ray_near_frac=args.ray_near_frac, ray_far_frac=args.ray_far_frac)
    cams.load_dense(args.pcd_path, voxel=args.dense_voxel)
    vis3d = Visualizer3D(cams, pcd_path=args.pcd_path, point_size_mult=args.point_size_mult, arrow_width_mult=args.arrow_width_mult, arrow_length=args.arrow_length, arrow_radius=args.arrow_radius, arrow_mode=args.arrow_mode, line_width=args.line_width)
    image_name = Path(img_path).name
    pred_list: List[Pred3D] = []
    valid_count = 0
    for idx, det in enumerate(dets):
        print(f"[Det] #{idx} cls={det.cls} conf={det.conf:.2f} bbox={det.bbox}")
        det_dir = out_root / f"det_{idx:03d}_cls{det.cls}_conf{det.conf:.2f}"
        ensure_dir(det_dir)
        overlay = draw_bbox_overlay(img, det)
        cv2.imwrite(str(det_dir / '01_bbox_overlay.png'), overlay)
        x1, y1, x2, y2 = det.bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img.shape[1], x2), min(img.shape[0], y2)
        roi = img[y1:y2, x1:x2]
        cv2.imwrite(str(det_dir / '02_roi.png'), roi)
        circles, hough_src, edges = circle.detect(roi)
        src_img = edges if not args.hough_on_gray else hough_src
        if src_img.ndim == 2:
            cv2.imwrite(str(det_dir / '03_hough_src_bw.png'), src_img)
        else:
            cv2.imwrite(str(det_dir / '03_hough_src_bw.png'), cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY))
        cand_img = render_hough_candidates(roi.shape, circles)
        cv2.imwrite(str(det_dir / '03_hough_candidates.png'), cand_img)
        best = select_best_circle(circles, roi.shape, args.w_radius, args.w_center, args.w_edge)
        best_circle_img = overlay_best_circle(roi, best)
        cv2.imwrite(str(det_dir / '04_best_circle.png'), best_circle_img)
        kp_xy, kp_conf, heat_np = kpdet.detect(roi)
        center_xy = (best[0], best[1]) if best is not None else None
        if center_xy is None:
            print("[Main] No circle center; skipping orientation for this detection")
        kp_and_center_img = overlay_kp_and_center(roi, kp_xy, center_xy)
        cv2.imwrite(str(det_dir / '05_kp_and_center.png'), kp_and_center_img)
        direction_img = overlay_direction(roi, kp_xy, center_xy)
        cv2.imwrite(str(det_dir / '06_direction.png'), direction_img)
        meta = {
            'bbox': {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)},
            'center_full': {'x': int(det.center[0]), 'y': int(det.center[1])},
            'confidence': float(det.conf),
            'class_id': int(det.cls),
            'hough_candidates': [{'cx': int(cx), 'cy': int(cy), 'r': int(r)} for (cx, cy, r) in circles],
            'best_circle': None if best is None else {'cx': int(best[0]), 'cy': int(best[1]), 'r': int(best[2])},
            'keypoint': None if kp_xy is None else {'x': int(kp_xy[0]), 'y': int(kp_xy[1]), 'conf': float(kp_conf)}
        }
        with open(det_dir / '00_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        kp_img = None if kp_xy is None else (x1 + kp_xy[0], y1 + kp_xy[1])
        circ_img = None if center_xy is None else (x1 + center_xy[0], y1 + center_xy[1])
        r_px = None if best is None else float(best[2])
        if kp_img is not None and circ_img is not None:
            v, kp3d, cc3d = cams.vec_and_points_from_2D_pair(
                kp_img,
                circ_img,
                image_name,
                circle_radius_px=r_px
            )
            pred_list.append(Pred3D(kp3d=kp3d, cc3d=cc3d, class_id=det.cls))
            if v is None:
                print("[Main] Orientation vector not available for this detection")
            else:
                valid_count += 1
        else:
            if kp_img is None:
                print("[Main] Missing keypoint; skipping 3D orientation")
            if circ_img is None:
                print("[Main] Missing circle; skipping 3D orientation")
            pred_list.append(Pred3D(kp3d=None, cc3d=None, class_id=det.cls))
    print(f"[Main] Valid 3D orientations: {valid_count}/{len(dets)}")
    if args.show_o3d and o3d is not None:
        vis3d.show(pred_list)
    if args.save_pointcloud and o3d is not None:
        base_pc = vis3d._load_pointcloud()
        if base_pc is None:
            base_pc = o3d.geometry.PointCloud()
        new_pts = []
        new_cols = []
        L = float(args.arrow_length)
        added = 0
        for p in pred_list:
            if p.class_id == 1:
                continue
            if p.kp3d is None or p.cc3d is None:
                continue
            v = p.cc3d - p.kp3d
            n = float(np.linalg.norm(v))
            if n <= 1e-9:
                continue
            u = v / n
            start = p.kp3d
            end = p.kp3d + u * L
            for t in np.linspace(0.0, 1.0, num=max(2, int(args.arrow_samples))):
                q = (1.0 - t) * start + t * end
                new_pts.append(q)
                new_cols.append([0.0, 1.0, 0.0])
            added += 1
        if new_pts:
            pp = np.asarray(new_pts)
            cc = np.asarray(new_cols)
            bp = np.asarray(base_pc.points)
            if bp.size == 0:
                base_pc.points = o3d.utility.Vector3dVector(pp)
                base_pc.colors = o3d.utility.Vector3dVector(cc)
            else:
                base_pc.points = o3d.utility.Vector3dVector(np.vstack([bp, pp]))
                if base_pc.has_colors():
                    bc = np.asarray(base_pc.colors)
                else:
                    bc = np.tile([[0.5, 0.5, 0.5]], (bp.shape[0], 1))
                base_pc.colors = o3d.utility.Vector3dVector(np.vstack([bc, cc]))
        out_path = Path(args.save_pointcloud)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(out_path), base_pc)
        print(f"[SavePCD] Wrote point cloud with {added} arrows to {out_path}")
    if args.save_lines and o3d is not None:
        pts = []
        lines = []
        colors = []
        idx = 0
        L = float(args.arrow_length)
        added = 0
        for p in pred_list:
            if p.class_id == 1:
                continue
            if p.kp3d is None or p.cc3d is None:
                continue
            v = p.cc3d - p.kp3d
            n = float(np.linalg.norm(v))
            if n <= 1e-9:
                continue
            u = v / n
            start = p.kp3d
            end = p.kp3d + u * L
            pts.append(start)
            pts.append(end)
            lines.append([idx, idx + 1])
            colors.append([0.0, 1.0, 0.0])
            idx += 2
            added += 1
        if pts:
            ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(np.asarray(pts)), lines=o3d.utility.Vector2iVector(np.asarray(lines)))
            ls.colors = o3d.utility.Vector3dVector(np.asarray(colors))
            out_path = Path(args.save_lines)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_line_set(str(out_path), ls)
            print(f"[SaveLines] Wrote {added} arrows to {out_path}")
    print(f"Exported {len(dets)} detection folders to: {out_root}")

if __name__ == '__main__':
    main()
