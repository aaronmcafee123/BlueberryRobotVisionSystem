import os
import cv2
import glob
import math
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
try:
    import open3d as o3d
except ImportError:
    o3d = None
from read_write_model import read_model


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


@dataclass
class Prediction:
    image_name: str
    det_id: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    conf: float
    class_id: int
    keypoint_2d: Optional[Tuple[int, int]]
    keypoint_conf: float
    circle_2d: Optional[Tuple[int, int]]
    circle_radius: Optional[int]
    orientation_3d: Optional[np.ndarray]
    kp_point3D: Optional[np.ndarray] = None
    circle_point3D: Optional[np.ndarray] = None


@dataclass
class GroundTruth:
    berry_id: str
    base: np.ndarray
    tip: np.ndarray
    orientation_3d: np.ndarray


class YOLODetector:
    def __init__(self, weights: str, conf_thresh: float):
        self.model = YOLO(weights, verbose=False)
        self.conf_thresh = conf_thresh

    def detect(self, img_path: str) -> List[Dict]:
        r = self.model(img_path)[0]
        out = []
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            if float(conf) < self.conf_thresh:
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            out.append({
                'bbox': (x1, y1, x2, y2),
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'conf': float(conf),
                'cls': int(cls)
            })
        return out


class CircleDetector:
    def __init__(self, p: Dict):
        self.p = p

    def detect(self, roi: np.ndarray) -> List[Tuple[int, int, int]]:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if self.p.get('median_blur', 5) > 0:
            gray = cv2.medianBlur(gray, self.p['median_blur'])
        gray_eq = cv2.equalizeHist(gray) if self.p.get('equalize', True) else gray
        edges = cv2.Canny(gray_eq, self.p['canny1'], self.p['canny2'])
        if self.p.get('dilate_iter', 0) > 0:
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=self.p['dilate_iter'])
        src = gray_eq if self.p.get('hough_on_gray', True) else edges
        circles = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, dp=self.p['dp'], minDist=self.p['minDist'],
                                   param1=self.p['param1'], param2=self.p['param2'],
                                   minRadius=self.p['minRadius'], maxRadius=self.p['maxRadius'])
        if circles is None:
            return []
        return [(int(x), int(y), int(r)) for x, y, r in circles[0]]


def select_best_circle(circles: List[Tuple[int, int, int]], roi_shape: Tuple[int, int, int], w_radius: float, w_center: float, w_edge: float) -> Optional[Tuple[int, int, int]]:
    if not circles:
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
            best_score = score
            best = (cx, cy, r)
    return best


class KeypointDetector:
    def __init__(self, weights: str, device: str, thresh: float,
                 input_size: int = 256, heatmap_size: int = 64,
                 base_ch: int = 64, gn_groups: Optional[int] = 16,
                 use_se: bool = True, deep_supervision: bool = False):
        self.device = torch.device(device)
        self.thresh = float(thresh)
        self.input_size = int(input_size)
        self.heatmap_size = int(heatmap_size)
        self.deep_supervision = bool(deep_supervision)
        # If gn_groups <= 0 => use BatchNorm by passing None
        gn = None if (gn_groups is None or gn_groups <= 0) else int(gn_groups)
        self.model = UNetPlus(in_ch=3, base=base_ch, gn_groups=gn, out_ch=1, use_se=use_se).to(self.device)
        sd = torch.load(weights, map_location=self.device)
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()
        self.tf = T.Compose([
            T.ToPILImage(),
            T.Resize((self.input_size, self.input_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @staticmethod
    def _to_numpy_heatmap(t: torch.Tensor) -> np.ndarray:
        # Ensure shape (H, W)
        if t.dim() == 4:
            t = t[0, 0]
        elif t.dim() == 3:
            t = t[0]
        return t.detach().cpu().numpy()

    def _forward_single_heatmap(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.model(x, return_multi=self.deep_supervision)
            if isinstance(out, list):
                # Average multi-scale heads to a single heatmap
                out = torch.stack(out, dim=0).mean(dim=0)
            # Optionally resize to desired heatmap size for peak picking
            if self.heatmap_size > 0 and (out.shape[-2] != self.heatmap_size or out.shape[-1] != self.heatmap_size):
                out = F.interpolate(out, size=(self.heatmap_size, self.heatmap_size), mode='bilinear', align_corners=False)
            return out

    def detect(self, roi: np.ndarray) -> Tuple[Optional[Tuple[int, int]], float]:
        if roi is None or roi.size == 0:
            return None, 0.0
        roi_h, roi_w = roi.shape[:2]
        inp = self.tf(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
        heat = self._forward_single_heatmap(inp)
        heat_np = self._to_numpy_heatmap(heat)
        conf = float(heat_np.max())
        if conf < self.thresh:
            return None, conf
        py, px = np.unravel_index(int(heat_np.argmax()), heat_np.shape)
        # Map heatmap coords back to ROI coords
        kx = int(px * roi_w / heat_np.shape[1])
        ky = int(py * roi_h / heat_np.shape[0])
        return (kx, ky), conf


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
        self.dense_voxel = 0.0
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

    def configure_dense(self, dense_proj_max_pix: float = 4.0, dense_max_proj: int = 300000, ray_radius: float = 0.02, ray_samples: int = 24, ray_near_frac: float = 0.02, ray_far_frac: float = 0.98):
        self.dense_proj_max_pix = float(dense_proj_max_pix)
        self.dense_max_proj = int(dense_max_proj)
        self.ray_radius = float(ray_radius)
        self.ray_samples = int(ray_samples)
        self.ray_near_frac = float(ray_near_frac)
        self.ray_far_frac = float(ray_far_frac)

    def load_dense(self, pcd_path: Optional[str], voxel: float = 0.0):
        if o3d is None or pcd_path is None or not os.path.exists(pcd_path):
            return
        try:
            pcd = o3d.io.read_point_cloud(pcd_path)
            if voxel and voxel > 0.0:
                pcd = pcd.voxel_down_sample(voxel_size=float(voxel))
            self.dense_pcd = pcd
            self.dense_pts = np.asarray(pcd.points)
            if self.dense_pts.size == 0:
                self.dense_pcd = None
                self.dense_pts = None
                self.dense_kd = None
                return
            self.dense_kd = o3d.geometry.KDTreeFlann(pcd)
            self.dense_aabb = pcd.get_axis_aligned_bounding_box()
            self.scene_diag = float(np.linalg.norm(self.dense_aabb.get_extent()))
        except Exception as e:
            print(f"[WARN] Failed to load dense point cloud '{pcd_path}': {e}")

    def _img_entry(self, name: str):
        for im in self.images.values():
            if im.name == name:
                return im
        return None

    @staticmethod
    def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
        w, x, y, z = qvec
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
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
        if float(np.sqrt(d2[j])) <= self.dense_proj_max_pix:
            k = int(idx[j])
            return np.asarray(self.dense_pts[k])
        return None

    def _lift_with_ray(self, pix: Tuple[int,int], img_name: str) -> Optional[np.ndarray]:
        if self.dense_pts is None or self.dense_kd is None:
            return None
        im = self._img_entry(img_name)
        if im is None:
            return None
        cam = self.cameras.get(im.camera_id, None)
        if cam is None:
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
            return None
        Pw = self.dense_pts[list(candidates)]
        vcp = Pw - C.reshape(1,3)
        t_param = vcp @ dir_world
        proj = np.outer(t_param, dir_world)
        perp = vcp - proj
        dists = np.linalg.norm(perp, axis=1)
        j = int(np.argmin(dists))
        if float(dists[j]) <= self.ray_radius:
            return Pw[j]
        return None

    def _ray_dir_cam(self, pix: Tuple[int,int], cam) -> np.ndarray:
        """Calculate ray direction in camera coordinates (unit vector)"""
        fx, fy, cx, cy, _ = self._intrinsics(cam)
        u, v = float(pix[0]), float(pix[1])
        x = (u - cx) / fx
        y = (v - cy) / fy
        d = np.array([x, y, 1.0], dtype=np.float64)
        return d / (np.linalg.norm(d) + 1e-12)

    def correct_sphere_center(self, circ2d: Tuple[int,int], surface_world: np.ndarray, 
                            img_name: str, circle_radius_px: float) -> np.ndarray:
        """
        Correct surface point to geometric sphere center using:
        1. Camera pose and intrinsics
        2. Projected circle radius in pixels
        3. Surface point location

        Args:
            circ2d: 2D circle center in image coordinates
            surface_world: 3D surface point of the sphere
            img_name: Image identifier
            circle_radius_px: Radius of detected circle in pixels

        Returns:
            Corrected 3D sphere center
        """
        im = self._img_entry(img_name)
        if im is None or circle_radius_px <= 0:
            return surface_world  # Fallback to surface point

        cam = self.cameras.get(im.camera_id, None)
        if cam is None:
            return surface_world

        R, t, _ = self._image_pose(im)
        fx, fy, cx, cy, _ = self._intrinsics(cam)
        f = 0.5 * (fx + fy)  # Average focal length

        # Calculate ray direction from camera to circle center
        dir_cam = self._ray_dir_cam(circ2d, cam)  # Unit vector in camera coords
        
        # Transform surface point to camera coordinates
        surface_cam = R @ surface_world + t
        surface_dist = float(np.dot(surface_cam, dir_cam))  # Distance along ray
        
        # Calculate correction factor using similar triangles
        dir_z = float(dir_cam[2])
        denominator = (f - circle_radius_px * dir_z)
        if abs(denominator) <= 1e-9:
            return surface_world  # Ill-conditioned case
        
        # Estimate sphere radius in 3D space
        sphere_radius_3d = (circle_radius_px * surface_dist * dir_z) / denominator
        
        # Calculate geometric center in camera coordinates
        center_cam = (surface_dist + sphere_radius_3d) * dir_cam
        
        # Convert back to world coordinates
        center_world = R.T @ (center_cam - t)
        return center_world

    def vec_and_points_from_2D_pair(self, kp2d: Tuple[int, int], circ2d: Tuple[int, int], 
                                  img_name: str, circle_radius_px: Optional[float] = None
                                  ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        im = self._img_entry(img_name)
        if im is None:
            return None, None, None
        feats = im.xys
        ids = im.point3D_ids
        p_kp = p_cc = None
        if feats.size > 0:
            kp_id = ids[np.argmin(np.linalg.norm(feats - np.array(kp2d), axis=1))]
            cc_id = ids[np.argmin(np.linalg.norm(feats - np.array(circ2d), axis=1))]
            if kp_id >= 0 and kp_id in self.points3D:
                p_kp = self.points3D[kp_id].xyz
            if cc_id >= 0 and cc_id in self.points3D:
                p_cc = self.points3D[cc_id].xyz
        if p_kp is None:
            p_kp = self._lift_with_dense_projection(kp2d, img_name)
        if p_cc is None:
            p_cc = self._lift_with_dense_projection(circ2d, img_name)
        if p_kp is None:
            p_kp = self._lift_with_ray(kp2d, img_name)
        if p_cc is None:
            p_cc = self._lift_with_ray(circ2d, img_name)
            
        # CORRECT SPHERE CENTER USING GEOMETRY
        if circle_radius_px is not None and circle_radius_px > 0 and p_cc is not None:
            p_cc = self.correct_sphere_center(circ2d, p_cc, img_name, circle_radius_px)
        
        # Calculate orientation vector
        if p_kp is None or p_cc is None:
            return None, p_kp, p_cc
        
        v = p_cc - p_kp
        n = np.linalg.norm(v)
        if n == 0:
            return None, p_kp, p_cc
        return v / n, p_kp, p_cc


def unit(v: np.ndarray) -> Optional[np.ndarray]:
    if v is None:
        return None
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return None
    return v / n


def vec_to_unit_quat(v: Optional[np.ndarray], ref: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    if v is None or np.linalg.norm(v) == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    v = v / np.linalg.norm(v)
    r = ref / np.linalg.norm(ref)
    dot = float(np.clip(np.dot(r, v), -1.0, 1.0))
    ang = math.acos(dot)
    if ang < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = np.cross(r, v)
    n = np.linalg.norm(axis)
    if n < 1e-8:
        axis = np.array([1.0, 0.0, 0.0]) if abs(r[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    else:
        axis /= n
    h = ang / 2.0
    s = math.sin(h)
    return np.array([math.cos(h), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_unsigned_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = abs(float(np.dot(q1, q2)))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * math.degrees(math.acos(dot))


def build_cost_matrix(preds: List[Prediction], gts: List[GroundTruth]) -> np.ndarray:
    M = np.zeros((len(preds), len(gts)), float)
    for i, p in enumerate(preds):
        if p.orientation_3d is None:
            M[i, :] = 999.0
            continue
        qp = vec_to_unit_quat(p.orientation_3d)
        for j, g in enumerate(gts):
            qg = vec_to_unit_quat(g.orientation_3d)
            M[i, j] = quat_unsigned_angle_deg(qp, qg)
    return M


def unsigned_angle_between(u: np.ndarray, w: np.ndarray) -> float:
    u = unit(u); w = unit(w)
    if u is None or w is None:
        return float('nan')
    d = float(np.clip(abs(np.dot(u, w)), -1.0, 1.0))
    return math.degrees(math.acos(d))


def point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[float, np.ndarray, float]:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        cp = a
        return float(np.linalg.norm(p - cp)), cp, 0.0
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    cp = a + t * ab
    return float(np.linalg.norm(p - cp)), cp, t


class GroundTruthManager:
    def __init__(self, csv_path: str):
        self.csv = csv_path
        self.data: List[GroundTruth] = []

    def load(self):
        df = pd.read_csv(self.csv)
        for _, r in df.iterrows():
            base = np.array([r['base_x'], r['base_y'], r['base_z']], float)
            tip = np.array([r['tip_x'], r['tip_y'], r['tip_z']], float)
            v = tip - base
            n = np.linalg.norm(v)
            if n == 0:
                v = np.array([0.0, 0.0, 1.0])
            else:
                v = v / n
            self.data.append(GroundTruth(str(r['berry_id']), base, tip, v))


class Visualizer2D:
    def __init__(self, out_dir: str):
        self.dir = os.path.join(out_dir, 'overlays')
        os.makedirs(self.dir, exist_ok=True)

    def draw(self, img_path: str, preds: List[Prediction]):
        img = cv2.imread(img_path)
        if img is None:
            return
        for p in preds:
            x1, y1, x2, y2 = p.bbox
            color = (0, 255, 0) if p.class_id == 0 else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"P{p.det_id} C:{p.conf:.2f}", (x1, y1 - 7), 0, 0.5, color, 2)
            if p.class_id == 0:
                if p.keypoint_2d is not None:
                    cv2.circle(img, p.keypoint_2d, 6, (255, 0, 0), -1)
                else:
                    cv2.putText(img, 'NO KP', (x1, y2 + 15), 0, 0.45, (255, 0, 0), 1)
                if p.circle_2d is not None:
                    cv2.circle(img, p.circle_2d, 4, (0, 0, 255), -1)
                    if p.circle_radius:
                        cv2.circle(img, p.circle_2d, p.circle_radius, (0, 255, 255), 1)
                else:
                    cv2.putText(img, 'NO CIRCLE', (x1, y2 + 30), 0, 0.45, (0, 0, 255), 1)
                if p.keypoint_2d is not None and p.circle_2d is not None:
                    cv2.arrowedLine(img, p.circle_2d, p.keypoint_2d, (0, 255, 0), 3, tipLength=0.2)
                elif p.orientation_3d is not None:
                    end = (
                        int(p.center[0] + p.orientation_3d[0] * 60),
                        int(p.center[1] + p.orientation_3d[1] * 60)
                    )
                    cv2.arrowedLine(img, p.center, end, (0, 255, 0), 3, tipLength=0.2)
        out_path = os.path.join(self.dir, Path(img_path).stem + '_overlay.png')
        cv2.imwrite(out_path, img)


class Visualizer3D:
    def __init__(self, cams: CameraSystem, gt_mgr: GroundTruthManager, pcd_path: Optional[str] = None, point_size_mult: float = 0.5):
        self.cams = cams
        self.gt_mgr = gt_mgr
        self.pcd_path = pcd_path
        self.point_size_mult = point_size_mult
        self._scene_diag = 1.0
        if o3d is None:
            print("[WARN] open3d is not installed. 3D visualization will be skipped.")

    def _load_pointcloud(self):
        if o3d is None:
            return None
        if self.pcd_path and os.path.exists(self.pcd_path):
            try:
                pc = o3d.io.read_point_cloud(self.pcd_path)
                if pc.is_empty():
                    print(f"[WARN] Point cloud at {self.pcd_path} is empty; falling back to COLMAP points.")
                else:
                    return pc
            except Exception as e:
                print(f"[WARN] Failed to read {self.pcd_path}: {e}. Falling back to COLMAP points.")
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
        cyl_r = max(self._scene_diag * 0.003, 1e-4)
        cone_r = cyl_r * 2.0
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
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=cyl_r,
            cone_radius=cone_r,
            cylinder_height=cyl_h,
            cone_height=cone_len,
            resolution=20,
            cylinder_split=1,
            cone_split=1,
        )
        R = self._get_rotation_from_z(v)
        arrow.rotate(R, center=(0, 0, 0))
        arrow.translate(start)
        arrow.compute_vertex_normals()
        arrow.paint_uniform_color(np.asarray(color))
        return arrow

    def show(self, preds: List[Prediction]):
        if o3d is None:
            return
        geoms = []
        pc = self._load_pointcloud()
        if pc is not None and not pc.is_empty():
            geoms.append(pc)
            aabb = pc.get_axis_aligned_bounding_box()
            self._scene_diag = float(np.linalg.norm(aabb.get_extent()))
        else:
            self._scene_diag = 1.0
        for g in self.gt_mgr.data:
            mesh = self._make_arrow(g.base, g.tip, color=(1.0, 1.0, 0.0))
            if mesh is not None:
                geoms.append(mesh)
        for p in preds:
            if p.class_id != 0:
                continue
            if p.kp_point3D is None or p.circle_point3D is None:
                continue
            mesh = self._make_arrow(p.circle_point3D, p.kp_point3D, color=(0.0, 1.0, 0.0))
            if mesh is not None:
                geoms.append(mesh)
        if geoms:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            for _g in geoms:
                vis.add_geometry(_g)
            opt = vis.get_render_option()
            opt.point_size = max(0.1, opt.point_size * float(self.point_size_mult))
            vis.run()
            vis.destroy_window()


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.yolo = YOLODetector(args.yolo_weights, args.conf_thresh)
        self.circle = CircleDetector({
            'canny1': args.canny1,
            'canny2': args.canny2,
            'dilate_iter': args.dilate_iter,
            'dp': args.dp,
            'minDist': args.minDist,
            'param1': args.param1,
            'param2': args.param2,
            'minRadius': args.minRadius,
            'maxRadius': args.maxRadius,
            'equalize': args.equalize,
            'median_blur': args.median_blur,
            'hough_on_gray': args.hough_on_gray
        })
        self.kp = KeypointDetector(
            args.unet_weights,
            args.device,
            args.kp_thresh,
            input_size=args.kp_input_size,
            heatmap_size=args.kp_heatmap_size,
            base_ch=32,  # Change from 64 to 32 to match your weights
            gn_groups=(None if args.kp_gn_groups is None else int(args.kp_gn_groups)),
            use_se=(not args.kp_no_se),
            deep_supervision=args.kp_deep_supervision,
        )
        self.cams = CameraSystem(); self.cams.load(args.model_dir)
        self.cams.configure_dense(
            dense_proj_max_pix=args.dense_proj_max_pix,
            dense_max_proj=args.dense_max_proj,
            ray_radius=args.ray_radius,
            ray_samples=args.ray_samples,
            ray_near_frac=args.ray_near_frac,
            ray_far_frac=args.ray_far_frac,
        )
        self.cams.load_dense(args.pcd_path, voxel=args.dense_voxel)
        self.gt_mgr = GroundTruthManager(args.gt_csv); self.gt_mgr.load()
        self.vis2d = Visualizer2D(args.out_dir)
        self.vis3d = Visualizer3D(self.cams, self.gt_mgr, pcd_path=self.args.pcd_path, point_size_mult=self.args.point_size_mult)
        os.makedirs(args.out_dir, exist_ok=True)
        self.predictions: List[Prediction] = []
        self.per_image: Dict[str, List[Prediction]] = {}
        self.assignment: Dict[int, int] = {}

    def run(self):
        imgs = sorted(glob.glob(os.path.join(self.args.images, '*.jpg')) +
                      glob.glob(os.path.join(self.args.images, '*.png')))
        for img_path in imgs:
            name = Path(img_path).name
            dets = self.yolo.detect(img_path)
            img = cv2.imread(img_path)
            if img is None:
                continue
            preds_here: List[Prediction] = []
            for det_idx, d in enumerate(dets):
                x1, y1, x2, y2 = d['bbox']
                cls = d['cls']
                roi = img[y1:y2, x1:x2]
                kp2d = None
                kconf = 0.0
                circ_img = None
                radius = None
                orient = None
                kp3d = None
                circ3d = None
                if cls == 0:
                    kp2d, kconf = self.kp.detect(roi)
                    circles = self.circle.detect(roi)
                    if circles:
                        best = select_best_circle(circles, roi.shape,
                                                  self.args.w_radius, self.args.w_center, self.args.w_edge)
                        if best is not None:
                            cx, cy, r = best
                            circ_img = (x1 + cx, y1 + cy)
                            radius = r
                    if kp2d is not None and circ_img is not None:
                        kp_img = (x1 + kp2d[0], y1 + kp2d[1])
                        # PASS RADIUS TO GEOMETRIC CORRECTION
                        orient, kp3d, circ3d = self.cams.vec_and_points_from_2D_pair(
                            kp_img, 
                            circ_img, 
                            name,
                            circle_radius_px=radius  # <-- CRITICAL CORRECTION
                        )
                pred = Prediction(
                    name,
                    len(preds_here),
                    d['bbox'],
                    d['center'],
                    d['conf'],
                    cls,
                    None if kp2d is None else (x1 + kp2d[0], y1 + kp2d[1]),
                    kconf,
                    circ_img,
                    radius,
                    orient,
                    kp_point3D=kp3d,
                    circle_point3D=circ3d
                )
                preds_here.append(pred)
            self.per_image[name] = preds_here
            self.predictions.extend(preds_here)
        self.evaluate_save()

    def _evaluate_save_quat(self, csv_path: str):
        gts = self.gt_mgr.data
        C = build_cost_matrix(self.predictions, gts) if len(gts) > 0 else np.zeros((len(self.predictions), 0))
        assignment: Dict[int, int] = {}
        if C.size > 0:
            r, c = linear_sum_assignment(C)
            assignment = {int(i): int(j) for i, j in zip(r, c)}
        self.assignment = assignment
        rows = []
        for i, p in enumerate(self.predictions):
            if p.class_id != 0:
                continue
            qp = vec_to_unit_quat(p.orientation_3d)
            for j, g in enumerate(gts):
                qg = vec_to_unit_quat(g.orientation_3d)
                ang = quat_unsigned_angle_deg(qp, qg) if p.orientation_3d is not None else np.nan
                rows.append({
                    'prediction_id': f"{p.image_name}_P{p.det_id}",
                    'image_name': p.image_name,
                    'class_id': p.class_id,
                    'ground_truth_id': g.berry_id,
                    'angular_difference_deg': ang,
                    'is_best_match': int(assignment.get(i, -1) == j),
                    'pred_vec_x': p.orientation_3d[0] if p.orientation_3d is not None else np.nan,
                    'pred_vec_y': p.orientation_3d[1] if p.orientation_3d is not None else np.nan,
                    'pred_vec_z': p.orientation_3d[2] if p.orientation_3d is not None else np.nan,
                    'gt_vec_x': g.orientation_3d[0],
                    'gt_vec_y': g.orientation_3d[1],
                    'gt_vec_z': g.orientation_3d[2],
                    'pred_q_w': qp[0], 'pred_q_x': qp[1], 'pred_q_y': qp[2], 'pred_q_z': qp[3],
                    'gt_q_w': qg[0], 'gt_q_x': qg[1], 'gt_q_y': qg[2], 'gt_q_z': qg[3],
                })
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        for im, preds in self.per_image.items():
            self.vis2d.draw(os.path.join(self.args.images, im), preds)
        if self.args.show_o3d and o3d is not None:
            self.vis3d.show(self.predictions)

    def _evaluate_save_3d(self, csv_path: str):
        gts = self.gt_mgr.data
        rows = []
        self.assignment = {}
        skipped_count = 0
        total_predictions = 0
        
        for i, p in enumerate(self.predictions):
            if p.class_id != 0:
                continue
                
            total_predictions += 1
            
            if p.kp_point3D is None or p.circle_point3D is None:
                continue
                
            p_start = p.circle_point3D
            p_end = p.kp_point3D
            p_vec = unit(p_end - p_start)
            if p_vec is None:
                continue
                
            best_j, best_dist = None, float('inf')
            for j, g in enumerate(gts):
                dist, _, _ = point_to_segment_distance(p_start, g.base, g.tip)
                if dist < best_dist:
                    best_dist, best_j = dist, j
                    
            if best_j is None:
                continue
                
            # Apply distance threshold filter
            if best_dist > self.args.gt_max_dist:
                skipped_count += 1
                continue
                
            self.assignment[i] = best_j
            g = gts[best_j]
            g_vec = unit(g.tip - g.base)
            ang = unsigned_angle_between(p_vec, g_vec) if g_vec is not None else np.nan
            
            rows.append({
                'prediction_id': f"{p.image_name}_P{p.det_id}",
                'image_name': p.image_name,
                'class_id': p.class_id,
                'matched_ground_truth_id': g.berry_id,
                'angular_difference_deg': ang,
                'distance_to_gt_segment': best_dist,
                'pred_start_x': p_start[0], 'pred_start_y': p_start[1], 'pred_start_z': p_start[2],
                'pred_end_x': p_end[0],   'pred_end_y': p_end[1],   'pred_end_z': p_end[2],
                'gt_base_x': g.base[0], 'gt_base_y': g.base[1], 'gt_base_z': g.base[2],
                'gt_tip_x':  g.tip[0],  'gt_tip_y':  g.tip[1],  'gt_tip_z':  g.tip[2],
            })
            
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        
        # Print statistics
        print(f"[Evaluation] Total berry predictions: {total_predictions}")
        print(f"[Evaluation] Valid matches: {len(rows)}")
        print(f"[Evaluation] Skipped due to distance > {self.args.gt_max_dist}m: {skipped_count}")
        
        for im, preds in self.per_image.items():
            self.vis2d.draw(os.path.join(self.args.images, im), preds)
        if self.args.show_o3d and o3d is not None:
            self.vis3d.show(self.predictions)

    def evaluate_save(self):
        out_csv = os.path.join(self.args.out_dir, 'angular_differences.csv')
        if self.args.eval_mode == '3d':
            self._evaluate_save_3d(out_csv)
        else:
            self._evaluate_save_quat(out_csv)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--yolo_weights', required=True)
    ap.add_argument('--unet_weights', required=True)
    ap.add_argument('--gt_csv', required=True)
    ap.add_argument('--out_dir', default='evaluation_results')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--conf_thresh', type=float, default=0.6)
    ap.add_argument('--kp_thresh', type=float, default=0.2)

    # --- New UNetPlus keypoint detector params ---
    ap.add_argument('--kp_input_size', type=int, default=256, help='Resize ROI to this square size for UNetPlus input')
    ap.add_argument('--kp_heatmap_size', type=int, default=64, help='Optionally resize output heatmap to this size for peak picking')
    ap.add_argument('--kp_base_ch', type=int, default=64, help='Base channels used when training the UNetPlus')
    ap.add_argument('--kp_gn_groups', type=int, default=16, help='<=0 to use BatchNorm; must match training')
    ap.add_argument('--kp_no_se', action='store_true', help='Disable SE attention (must match training)')
    ap.add_argument('--kp_deep_supervision', action='store_true', help='Model was trained with deep supervision heads')

    # Circle/Hough params
    ap.add_argument('--canny1', type=int, default=100)
    ap.add_argument('--canny2', type=int, default=200)
    ap.add_argument('--dilate_iter', type=int, default=0)
    ap.add_argument('--dp', type=float, default=1.2)
    ap.add_argument('--minDist', type=int, default=15)
    ap.add_argument('--param1', type=int, default=100)
    ap.add_argument('--param2', type=int, default=5)
    ap.add_argument('--minRadius', type=int, default=12)
    ap.add_argument('--maxRadius', type=int, default=90)
    ap.add_argument('--equalize', action='store_true', default=True)
    ap.add_argument('--median_blur', type=int, default=5)
    ap.add_argument('--hough_on_gray', action='store_true', default=True)
    ap.add_argument('--w_radius', type=float, default=0.6)
    ap.add_argument('--w_center', type=float, default=0.4)
    ap.add_argument('--w_edge', type=float, default=0.5)

    # Dense lifting / visualization
    ap.add_argument('--pcd_path', type=str, default=None, help='Optional path to a dense point cloud (e.g., .ply, .pcd) used for the 3D overlay & 2D->3D lifting fallbacks')
    ap.add_argument('--dense_voxel', type=float, default=0.0, help='Voxel size for optional downsampling of the dense point cloud before KD-tree (0=off)')
    ap.add_argument('--dense_proj_max_pix', type=float, default=4.0, help='Max pixel distance for dense projection nearest match')
    ap.add_argument('--dense_max_proj', type=int, default=300000, help='Cap on number of dense points to project per image (subsampled evenly)')
    ap.add_argument('--ray_radius', type=float, default=0.02, help='Ray-nearest radius (scene units) for dense KD-tree queries')
    ap.add_argument('--ray_samples', type=int, default=24, help='How many samples along the camera ray for dense radius searches')
    ap.add_argument('--ray_near_frac', type=float, default=0.02, help='Near sampling distance as a fraction of scene diagonal')
    ap.add_argument('--ray_far_frac', type=float, default=0.98, help='Far sampling distance as a fraction of scene diagonal')
    ap.add_argument('--point_size_mult', type=float, default=0.5, help='Multiply default point size (0.5 halves it)')

    # Evaluation
    ap.add_argument('--eval_mode', choices=['3d', 'quat'], default='3d', help='Evaluate using 3D overlay (nearest GT segment) or original quaternion-based method')
    ap.add_argument('--show_o3d', action='store_true', help='Pop up an interactive Open3D window at the end')
    ap.add_argument('--gt_max_dist', type=float, default=0.3, help='Max distance (in meters) to accept a GT match; otherwise skip the prediction')
    return ap.parse_args()


def main():
    args = parse_args()
    pipe = Pipeline(args)
    pipe.run()


if __name__ == '__main__':
    main()