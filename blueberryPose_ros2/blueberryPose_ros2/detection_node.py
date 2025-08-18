#!/usr/bin/env python3

"""
BlueberryDetectorV2 — ROS 2 node (rclpy)

Implements the "new protocol" from the evaluation pipeline, but tailored for
an online ROS workflow using RGB + Depth:

• YOLO for 2D detections (class 0 = blueberry by default)
• UNetPlus heatmap keypoint detector (configurable to match training)
• Robust circle detection (Canny/Hough + heuristic selection)
• Lift 2D keypoint & circle-center pixels into 3D using the depth map
• Correct the circle surface point to the geometric sphere center using the
  camera model and the detected circle radius in pixels (no Open3D needed)
• Publish:
    - Overlay image with bboxes, keypoint, circle, and 2D arrow
    - Detection2DArray for detections
    - Marker ARROW in camera frame from corrected sphere center → keypoint

Notes
-----
- Assumes the depth image is aligned to the RGB image. If your topics are not
  already aligned, please perform registration upstream and remap this node to
  use the aligned depth topic.
- Depth is expected in meters. If your depth topic publishes uint16 in
  millimeters, set the parameter `depth_scale` accordingly (default auto).
- The UNetPlus configuration must match your training setup: base channels,
  group norm groups, use of SE, deep supervision, etc.
"""

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from ultralytics import YOLO

from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import (
    Detection2DArray,
    Detection2D,
    ObjectHypothesisWithPose,
    BoundingBox2D,
)
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration

# ──────────────────────────────────────────────────────────────────────────────
# Model components (UNetPlus + helpers) — mirrors the eval pipeline
# ──────────────────────────────────────────────────────────────────────────────

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
        # Bottleneck with dilation
        self.bot = nn.Sequential(
            nn.Conv2d(base * 4, base * 8, 3, padding=2, dilation=2, bias=False),
            (nn.GroupNorm(gn_groups, base * 8) if gn_groups else nn.BatchNorm2d(base * 8)),
            nn.SiLU(inplace=True),
            nn.Conv2d(base * 8, base * 8, 3, padding=4, dilation=4, bias=False),
            (nn.GroupNorm(gn_groups, base * 8) if gn_groups else nn.BatchNorm2d(base * 8)),
            nn.SiLU(inplace=True),
        )
        # Decoder
        self.up3 = UpBlock(base * 8, base * 4, base * 4, gn_groups=gn_groups, use_se=use_se)
        self.up2 = UpBlock(base * 4, base * 2, base * 2, gn_groups=gn_groups, use_se=use_se)
        self.up1 = UpBlock(base * 2, base, base, gn_groups=gn_groups, use_se=use_se)
        # Heads
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

        # Deep supervision
        h3 = F.interpolate(self.head3(d3), size=d1.shape[-2:], mode='bilinear', align_corners=False)
        h2 = F.interpolate(self.head2(d2), size=d1.shape[-2:], mode='bilinear', align_corners=False)
        h1 = self.head1(d1)

        if return_multi:
            return [torch.sigmoid(h) for h in (h1, h2, h3)]
        return torch.sigmoid(h1)


class KeypointDetector:
    def __init__(
        self,
        weights: str,
        device: str,
        thresh: float,
        input_size: int = 256,
        heatmap_size: int = 64,
        base_ch: int = 32,
        gn_groups: Optional[int] = 16,
        use_se: bool = True,
        deep_supervision: bool = False,
    ):
        self.device = torch.device(device)
        self.thresh = float(thresh)
        self.input_size = int(input_size)
        self.heatmap_size = int(heatmap_size)
        self.deep_supervision = bool(deep_supervision)
        gn = None if (gn_groups is None or gn_groups <= 0) else int(gn_groups)
        self.model = UNetPlus(in_ch=3, base=base_ch, gn_groups=gn, out_ch=1, use_se=use_se).to(self.device)
        sd = torch.load(weights, map_location=self.device)
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()
        self.tf = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((self.input_size, self.input_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

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
            if self.heatmap_size > 0 and (
                out.shape[-2] != self.heatmap_size or out.shape[-1] != self.heatmap_size
            ):
                out = F.interpolate(
                    out, size=(self.heatmap_size, self.heatmap_size), mode='bilinear', align_corners=False
                )
            return out

    def detect(self, roi_bgr: np.ndarray) -> Tuple[Optional[Tuple[int, int]], float]:
        if roi_bgr is None or roi_bgr.size == 0:
            return None, 0.0
        h, w = roi_bgr.shape[:2]
        inp = self.tf(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
        heat = self._forward_single_heatmap(inp)
        heat_np = self._to_numpy_heatmap(heat)
        conf = float(heat_np.max())
        if conf < self.thresh:
            return None, conf
        py, px = np.unravel_index(int(heat_np.argmax()), heat_np.shape)
        kx = int(px * w / heat_np.shape[1])
        ky = int(py * h / heat_np.shape[0])
        return (kx, ky), conf


class CircleDetector:
    def __init__(self, p: Dict):
        self.p = p

    def detect(self, roi_bgr: np.ndarray) -> List[Tuple[int, int, int]]:
        if roi_bgr is None or roi_bgr.size == 0:
            return []
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        if self.p.get("median_blur", 5) > 0:
            gray = cv2.medianBlur(gray, self.p["median_blur"])
        gray_eq = cv2.equalizeHist(gray) if self.p.get("equalize", True) else gray
        edges = cv2.Canny(gray_eq, self.p["canny1"], self.p["canny2"])
        if self.p.get("dilate_iter", 0) > 0:
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=self.p["dilate_iter"])
        src = gray_eq if self.p.get("hough_on_gray", True) else edges
        circles = cv2.HoughCircles(
            src,
            cv2.HOUGH_GRADIENT,
            dp=self.p["dp"],
            minDist=self.p["minDist"],
            param1=self.p["param1"],
            param2=self.p["param2"],
            minRadius=self.p["minRadius"],
            maxRadius=self.p["maxRadius"],
        )
        if circles is None:
            return []
        return [(int(x), int(y), int(r)) for x, y, r in circles[0]]


def select_best_circle(
    circles: List[Tuple[int, int, int]],
    roi_shape: Tuple[int, int, int],
    w_radius: float,
    w_center: float,
    w_edge: float,
) -> Optional[Tuple[int, int, int]]:
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


# ──────────────────────────────────────────────────────────────────────────────
# Utility math for projection with depth
# ──────────────────────────────────────────────────────────────────────────────

def ray_dir_cam(u: float, v: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    x = (u - cx) / fx
    y = (v - cy) / fy
    d = np.array([x, y, 1.0], dtype=np.float32)
    n = float(np.linalg.norm(d))
    return d / (n + 1e-12)


def backproject(u: float, v: float, z: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float32)


def correct_sphere_center_from_depth(
    u_circ: float,
    v_circ: float,
    z_surface: float,
    r_pixels: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Correct the *surface* point at the circle center pixel to the geometric
    sphere center along the camera ray using the same geometry as the eval node.

    Returns center in **camera coordinates**.
    """
    # Ray direction for circle center pixel
    d = ray_dir_cam(u_circ, v_circ, fx, fy, cx, cy)  # unit vector
    # Surface point in camera coords from pinhole backprojection (Z-depth)
    p_surf = backproject(u_circ, v_circ, z_surface, fx, fy, cx, cy)
    # Distance along ray to surface point
    s = float(np.dot(p_surf, d))
    # Use average focal length
    f = 0.5 * (fx + fy)
    denom = (f - r_pixels * float(d[2]))
    if abs(denom) <= 1e-9:
        return p_surf  # ill-conditioned; return surface point
    # Estimate sphere radius in 3D and shift along the ray
    R3D = (r_pixels * s * float(d[2])) / denom
    center_cam = (s + R3D) * d
    return center_cam.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# ROS 2 Node
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DetOut:
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    conf: float
    cls: int
    kp2d: Optional[Tuple[int, int]]
    kpconf: float
    circ2d: Optional[Tuple[int, int]]
    radius: Optional[int]
    kp3d: Optional[np.ndarray]
    cc3d: Optional[np.ndarray]
    vec3d: Optional[np.ndarray]


class BlueberryDetectorV2(Node):
    def __init__(self):
        super().__init__('detection_node')
        self.bridge = CvBridge()

        # Parameters — topics
        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")

        # Parameters — models & thresholds
        self.declare_parameter("processing_rate", 1.0)
        self.declare_parameter("confidence_threshold", 0.1)
        self.declare_parameter("yolo_model_path", "yolo_best.pt")
        self.declare_parameter("unet_weights", "unet_epoch200.pth")
        self.declare_parameter("kp_thresh", 0.1)
        self.declare_parameter("kp_input_size", 256)
        self.declare_parameter("kp_heatmap_size", 64)
        self.declare_parameter("kp_base_ch", 32)  # match training by default
        self.declare_parameter("kp_gn_groups", 16)
        self.declare_parameter("kp_use_se", True)
        self.declare_parameter("kp_deep_supervision", False)

        # Depth handling
        self.declare_parameter("depth_scale", 0.0)  # 0.0 => auto from dtype (uint16->*1e-3)
        self.declare_parameter("depth_patch", 3)    # odd window to median-filter depth lookup

        # Circle/Hough params (+ selection weights)
        self.declare_parameter("canny1", 100)
        self.declare_parameter("canny2", 200)
        self.declare_parameter("dilate_iter", 0)
        self.declare_parameter("dp", 1.2)
        self.declare_parameter("minDist", 15)
        self.declare_parameter("param1", 100)
        self.declare_parameter("param2", 5)
        self.declare_parameter("minRadius", 12)
        self.declare_parameter("maxRadius", 90)
        self.declare_parameter("equalize", True)
        self.declare_parameter("median_blur", 5)
        self.declare_parameter("hough_on_gray", True)
        self.declare_parameter("w_radius", 0.6)
        self.declare_parameter("w_center", 0.4)
        self.declare_parameter("w_edge", 0.5)
        self.declare_parameter("roi_margin", 6)
        self.declare_parameter("circle_alpha", 0.6)
        self.declare_parameter("circle_max_miss", 3)
        self.declare_parameter("circle_search_margin", 12)
        self.declare_parameter("track_max_center_px", 60)

        # Publish topics & marker namespace
        self.declare_parameter("overlay_topic", "/camera/detections/image_raw")
        self.declare_parameter("detections_topic", "/detections")
        self.declare_parameter("marker_topic", "/detections/orientation_marker")
        self.declare_parameter("marker_ns", "berry_orientation")
        self.declare_parameter("marker_lifetime", 0.5)

        # Retrieve params
        self.color_topic = self.get_parameter("color_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.caminfo_topic = self.get_parameter("camera_info_topic").value

        self.rate = float(self.get_parameter("processing_rate").value)
        self.conf_th = float(self.get_parameter("confidence_threshold").value)
        self.kp_thresh = float(self.get_parameter("kp_thresh").value)
        self.kp_input_size = int(self.get_parameter("kp_input_size").value)
        self.kp_heatmap_size = int(self.get_parameter("kp_heatmap_size").value)
        self.kp_base_ch = int(self.get_parameter("kp_base_ch").value)
        self.kp_gn_groups = int(self.get_parameter("kp_gn_groups").value)
        self.kp_use_se = bool(self.get_parameter("kp_use_se").value)
        self.kp_deep_supervision = bool(self.get_parameter("kp_deep_supervision").value)

        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.depth_patch = int(self.get_parameter("depth_patch").value)

        cpar = {
            "canny1": int(self.get_parameter("canny1").value),
            "canny2": int(self.get_parameter("canny2").value),
            "dilate_iter": int(self.get_parameter("dilate_iter").value),
            "dp": float(self.get_parameter("dp").value),
            "minDist": int(self.get_parameter("minDist").value),
            "param1": int(self.get_parameter("param1").value),
            "param2": int(self.get_parameter("param2").value),
            "minRadius": int(self.get_parameter("minRadius").value),
            "maxRadius": int(self.get_parameter("maxRadius").value),
            "equalize": bool(self.get_parameter("equalize").value),
            "median_blur": int(self.get_parameter("median_blur").value),
            "hough_on_gray": bool(self.get_parameter("hough_on_gray").value),
        }
        self.w_radius = float(self.get_parameter("w_radius").value)
        self.w_center = float(self.get_parameter("w_center").value)
        self.w_edge = float(self.get_parameter("w_edge").value)

        self.ns = str(self.get_parameter("marker_ns").value)
        self.overlay_topic = str(self.get_parameter("overlay_topic").value)
        self.dets_topic = str(self.get_parameter("detections_topic").value)
        self.marker_topic = str(self.get_parameter("marker_topic").value)

        yolo_path = str(self.get_parameter("yolo_model_path").value)
        unet_path = str(self.get_parameter("unet_weights").value)

        # Models
        self.yolo = YOLO(yolo_path, verbose=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass
        self.kp = KeypointDetector(
            unet_path,
            device=str(self.device),
            thresh=self.kp_thresh,
            input_size=self.kp_input_size,
            heatmap_size=self.kp_heatmap_size,
            base_ch=self.kp_base_ch,
            gn_groups=self.kp_gn_groups,
            use_se=self.kp_use_se,
            deep_supervision=self.kp_deep_supervision,
        )
        self.circ = CircleDetector(cpar)

        # Internal state
        self.color_msg: Optional[Image] = None
        self.depth_img: Optional[np.ndarray] = None
        self.fx = self.fy = self.cx = self.cy = None

        # ROS I/O
        self.create_subscription(Image, self.color_topic, self._cb_color, qos_profile_sensor_data)
        self.create_subscription(Image, self.depth_topic, self._cb_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.caminfo_topic, self._cb_info, 10)

        self.pub_img = self.create_publisher(Image, self.overlay_topic, 10)
        self.pub_det = self.create_publisher(Detection2DArray, self.dets_topic, 10)
        self.pub_mk = self.create_publisher(Marker, self.marker_topic, 10)

        # Track active marker ids (to delete stale ones)
        self.prev_marker_ids = set()
        # Simple per-detection tracking for continuity (IDs + circle EMA)
        self.tracks: Dict[int, Dict] = {}
        self.next_track_id: int = 1

        # Timer
        self.create_timer(1.0 / max(1e-3, self.rate), self._process_once)
        self.get_logger().info(f"detection_node running @ {self.rate:.2f} Hz")

    # ─────────────────────────── Callbacks ────────────────────────────
    def _cb_color(self, msg: Image):
        self.color_msg = msg

    def _cb_depth(self, msg: Image):
        arr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if arr is None:
            return
        if self.depth_scale > 0.0:
            # User-provided scale: raw * depth_scale => meters
            self.depth_img = arr.astype(np.float32) * float(self.depth_scale)
        else:
            # Auto: if uint16, assume millimeters
            if arr.dtype == np.uint16:
                self.depth_img = arr.astype(np.float32) * 1e-3
            else:
                self.depth_img = arr.astype(np.float32)

    def _cb_info(self, msg: CameraInfo):
        K = msg.k
        self.fx, self.fy = float(K[0]), float(K[4])
        self.cx, self.cy = float(K[2]), float(K[5])

    # ───────────────────────────── Utils ──────────────────────────────
    def _valid_cam(self) -> bool:
        return (
            self.color_msg is not None
            and self.depth_img is not None
            and None not in (self.fx, self.fy, self.cx, self.cy)
        )

    def _depth_at(self, u: int, v: int, patch: int) -> float:
        h, w = self.depth_img.shape[:2]
        u = int(np.clip(u, 0, w - 1))
        v = int(np.clip(v, 0, h - 1))
        if patch <= 1:
            z = float(self.depth_img[v, u])
            return z if np.isfinite(z) and z > 0 else 0.0
        k = patch // 2
        u0, v0 = max(0, u - k), max(0, v - k)
        u1, v1 = min(w, u + k + 1), min(h, v + k + 1)
        win = self.depth_img[v0:v1, u0:u1]
        valid = win[(win > 0) & np.isfinite(win)]
        if valid.size == 0:
            return 0.0
        return float(np.median(valid))

    # Helper: match a detection to an existing track by 2D center proximity
    def _match_track(self, center: Tuple[int,int]) -> Optional[int]:
        if not self.tracks:
            return None
        th = float(self.get_parameter("track_max_center_px").value)
        best_id, best_d = None, 1e9
        for tid, st in self.tracks.items():
            if "center" not in st:
                continue
            d = math.hypot(center[0] - st["center"][0], center[1] - st["center"][1])
            if d < best_d and d <= th:
                best_id, best_d = tid, d
        return best_id

    # Helper: update per-track circle with EMA + miss tolerance
    def _update_circle_state(self, tid: int, measured: Optional[Tuple[float, float, float]]):
        # measured = (u, v, r) in **image** pixels, or None if not found
        alpha = float(self.get_parameter("circle_alpha").value)
        max_miss = int(self.get_parameter("circle_max_miss").value)
        st = self.tracks.setdefault(tid, {})
        if measured is None:
            st["miss"] = int(st.get("miss", 0)) + 1
            if st.get("miss", 0) <= max_miss and all(k in st for k in ("circ_u","circ_v","radius")):
                return (st["circ_u"], st["circ_v"], st["radius"])  # hold last good for a bit
            return None
        # got a measurement
        u, v, r = measured
        if all(k in st for k in ("circ_u","circ_v","radius")):
            u = alpha * u + (1.0 - alpha) * float(st["circ_u"])
            v = alpha * v + (1.0 - alpha) * float(st["circ_v"])
            r = alpha * r + (1.0 - alpha) * float(st["radius"])
        st["circ_u"], st["circ_v"], st["radius"] = float(u), float(v), float(r)
        st["miss"] = 0
        return (st["circ_u"], st["circ_v"], st["radius"])

    # ─────────────────────────── Main loop ────────────────────────────
    def _process_once(self):
        if self.color_msg is None:
            return
        if not self._valid_cam():
            return

        msg = self.color_msg
        # consume frame (best-effort drop old frames under load)
        self.color_msg = None

        color = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if color is None:
            return
        overlay = color.copy()

        dets_msg = Detection2DArray()
        dets_msg.header = msg.header
        current_marker_ids = set()

        # Run YOLO
        try:
            results = self.yolo(color, verbose=False)[0]
        except Exception as e:
            self.get_logger().warning(f"YOLO inference failed: {e}")
            return

        outputs: List[DetOut] = []
        for di, (box, conf, cls) in enumerate(zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls)):
            conf = float(conf)
            cls = int(cls)
            if conf < self.conf_th:
                continue

            x1, y1, x2, y2 = map(int, box.tolist())
            cx2d, cy2d = (x1 + x2) // 2, (y1 + y2) // 2

            # Track ID continuity based on detection center proximity
            tid = self._match_track((cx2d, cy2d))
            if tid is None:
                tid = self.next_track_id
                self.next_track_id += 1
                self.tracks[tid] = {}
            self.tracks[tid]["center"] = (cx2d, cy2d)

            # Draw bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0) if cls == 0 else (0, 0, 255), 2)

            kp2d: Optional[Tuple[int, int]] = None
            kpconf: float = 0.0
            circ2d: Optional[Tuple[int, int]] = None
            radius: Optional[int] = None
            kp3d = None
            cc3d = None
            vec3d = None

            h_img, w_img = color.shape[:2]
            margin = int(self.get_parameter("roi_margin").value)
            x1e = max(0, x1 - margin); y1e = max(0, y1 - margin)
            x2e = min(w_img, x2 + margin); y2e = min(h_img, y2 + margin)
            roi = color[y1e:y2e, x1e:x2e]
            if cls == 0 and roi.size > 0:
                # Keypoint via UNetPlus heatmap
                kp_local, kpconf = self.kp.detect(roi)
                if kp_local is not None:
                    kp2d = (x1e + int(kp_local[0]), y1e + int(kp_local[1]))
                    cv2.circle(overlay, kp2d, 5, (255, 0, 0), -1)
                else:
                    cv2.putText(overlay, "NO KP", (x1, y2 + 15), 0, 0.5, (255, 0, 0), 1)

                # Circle via Hough + selection
                circles = self.circ.detect(roi)
                best = select_best_circle(circles, roi.shape, self.w_radius, self.w_center, self.w_edge)

                measured = None
                if best is not None:
                    cx, cy, r = best
                    measured = (float(x1e + cx), float(y1e + cy), float(int(r)))
                # Update per-track circle with EMA + miss hold
                sm = self._update_circle_state(tid, measured)
                if sm is not None:
                    cu, cv, rr = sm
                    circ2d = (int(round(cu)), int(round(cv)))
                    radius = int(round(rr))
                    cv2.circle(overlay, circ2d, 3, (0, 0, 255), -1)
                    cv2.circle(overlay, circ2d, radius, (0, 255, 255), 1)
                else:
                    cv2.putText(overlay, "NO CIRCLE", (x1, y2 + 30), 0, 0.5, (0, 0, 255), 1)

                # If we have both 2D points, lift to 3D using depth map
                if kp2d is not None and circ2d is not None and radius is not None:
                    kx, ky = kp2d
                    cxu, cyv = circ2d
                    z_k = self._depth_at(kx, ky, self.depth_patch)
                    z_c = self._depth_at(cxu, cyv, self.depth_patch)

                    if z_k > 0 and z_c > 0:
                        # 3D for keypoint (surface)
                        kp3d = backproject(kx, ky, z_k, self.fx, self.fy, self.cx, self.cy)
                        # Correct circle surface → sphere center using pixel radius
                        cc3d = correct_sphere_center_from_depth(
                            float(cxu), float(cyv), float(z_c), float(radius), self.fx, self.fy, self.cx, self.cy
                        )
                        # Orientation vector from sphere center → keypoint
                        v = kp3d - cc3d
                        n = float(np.linalg.norm(v))
                        if n > 1e-9:
                            vec3d = (v / n).astype(np.float32)

                        # Draw 2D arrow on overlay
                        cv2.arrowedLine(overlay, circ2d, kp2d, (0, 255, 0), 2, tipLength=0.2)

                        # Publish Marker in camera frame
                        mk = Marker()
                        mk.header = msg.header
                        mk.ns = self.ns
                        mk.id = tid
                        mk.type = Marker.ARROW
                        mk.action = Marker.ADD
                        mk.points = [
                            Point(x=float(cc3d[0]), y=float(cc3d[1]), z=float(cc3d[2])),
                            Point(x=float(kp3d[0]), y=float(kp3d[1]), z=float(kp3d[2])),
                        ]
                        mk.scale.x = 0.005  # shaft diameter
                        mk.scale.y = 0.01   # head diameter
                        mk.scale.z = 0.02   # head length
                        mk.color.r, mk.color.g, mk.color.b, mk.color.a = 0.0, 1.0, 0.0, 1.0
                        # Lifetime so old markers auto-expire if not refreshed
                        life = float(self.get_parameter("marker_lifetime").value)
                        mk.lifetime.sec = int(life)
                        mk.lifetime.nanosec = int((life - int(life)) * 1e9)
                        self.pub_mk.publish(mk)
                        current_marker_ids.add(tid)

            # Build Detection2D message for every detection
            det = Detection2D()
            det.header = msg.header
            bb = BoundingBox2D()
            bb.center.position.x = (x1 + x2) / 2.0
            bb.center.position.y = (y1 + y2) / 2.0
            bb.center.theta = 0.0
            bb.size_x = float(x2 - x1)
            bb.size_y = float(y2 - y1)
            det.bbox = bb

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(cls)
            hyp.hypothesis.score = conf
            det.results.append(hyp)
            dets_msg.detections.append(det)

            outputs.append(
                DetOut(
                    bbox=(x1, y1, x2, y2),
                    center=(cx2d, cy2d),
                    conf=conf,
                    cls=cls,
                    kp2d=kp2d,
                    kpconf=kpconf,
                    circ2d=circ2d,
                    radius=radius,
                    kp3d=kp3d,
                    cc3d=cc3d,
                    vec3d=vec3d,
                )
            )

        # Delete stale markers that were drawn in previous frame but not this one
        stale_ids = self.prev_marker_ids - current_marker_ids
        for sid in stale_ids:
            delmk = Marker()
            delmk.header = msg.header
            delmk.ns = self.ns
            delmk.id = sid
            delmk.action = Marker.DELETE
            self.pub_mk.publish(delmk)
        self.prev_marker_ids = current_marker_ids

        # Publish overlay & detections
        out_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        out_msg.header = msg.header
        self.pub_img.publish(out_msg)
        self.pub_det.publish(dets_msg)

        # Optional: log predicted roll/pitch per detection (camera frame)
        for i, o in enumerate(outputs):
            if o.vec3d is None:
                continue
            vx, vy, vz = float(o.vec3d[0]), float(o.vec3d[1]), float(o.vec3d[2])
            pred_pitch = float(math.asin(-vz))
            pred_roll = float(math.atan2(vy, vx))
            self.get_logger().debug(
                f"[det {i}] roll={pred_roll:.3f} rad, pitch={pred_pitch:.3f} rad"
            )


def main(args=None):
    rclpy.init(args=args)
    node = BlueberryDetectorV2()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
