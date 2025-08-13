#!/usr/bin/env python3
"""
infer_and_visualize_with_save.py

- Reads COLMAP’s binary sparse model via read_write_model.py
- Loads your fused bush PLY
- Runs YOLO12 + SmallUNet on every image in IMAGE_DIR to detect ripe vs. unripe blueberries
- For each detection in each image:
    • If unripe (class 1): just draw the YOLO box
    • If ripe   (class 0):
        – Run UNet → get (kx,ky) and draw it
        – Run HoughCircles → if found, draw it
        – Draw a bright-green arrow from keypoint → circle center
        – If UNet conf < KEYPOINT_THRESH or no circle: overlay “cant determine angle”
        – Else: match 2D keypoint & circle center to COLMAP features → get 3D points
- Saves predicted base & tip arrays for error analysis
- Visualizes the full bush + coordinate frame + colored 3D spheres + orientation arrows
- Lets you interactively remove background points and writes cleaned_cloud.ply
- Finally launches a free-fly camera view of the cleaned cloud
"""
import os
import glob
import numpy as np
import open3d as o3d
import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision import transforms as T
from read_write_model import read_model, qvec2rotmat
from pathlib import Path
from open3d.visualization import gui, rendering

# ─── CONFIG ───────────────────────────────────────────────────────────────
IMAGE_DIR         = "/home/aaronmcafee/Documents/special2/images"
MODEL_DIR         = "/home/aaronmcafee/Documents/special2/0"
POINTCLOUD_PLY    = "/home/aaronmcafee/Downloads/blueberry1Edited_rgb.ply"
YOLO12_WEIGHTS    = "/home/aaronmcafee/Documents/blueberryDetect/runs/sweep/yolo12n_lr0.010_idx0/weights/best.pt"
UNET_WEIGHTS      = "/home/aaronmcafee/ros2_ws/src/yolov11_ros2/yolov11_ros2/unet_epoch200.pth"
CONF_THRESH       = 0.5
KEYPOINT_THRESH   = 0.3
DEBUG_OVERLAY_DIR = "debug_overlays"
# ─────────────────────────────────────────────────────────────────────────

def build_intrinsics(camera):
    m, p = camera.model, camera.params
    if m == "SIMPLE_PINHOLE":
        fx = fy = p[0]; cx, cy = p[1], p[2]
    elif m == "PINHOLE":
        fx, fy, cx, cy = p
    elif m in ("SIMPLE_RADIAL", "RADIAL"):
        fx = fy = p[0]; cx, cy = p[1], p[2]
    else:
        raise ValueError(f"Unsupported camera model {m}")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)

def detect_circle(gray,
                  canny1=100, canny2=200, dilate_iter=1,
                  dp=1.5, minDist=20, param1=100, param2=10,
                  minRadius=3, maxRadius=100):
    edges = cv2.Canny(gray, canny1, canny2)
    if dilate_iter:
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=dilate_iter)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT,
                               dp=dp, minDist=minDist,
                               param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    if circles is not None and len(circles[0]) > 0:
        x, y, r = circles[0][0]
        return int(x), int(y), int(r)
    return None

class SmallUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.ReLU(True),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1), nn.ReLU(True),
        )
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*4, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base_ch*4, base_ch*4, 3, padding=1), nn.ReLU(True),
        )
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_ch*4, base_ch*2, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1), nn.ReLU(True),
        )
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.ReLU(True),
        )
        self.final = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        b  = self.bottleneck(p2)
        u2 = self.up2(b); d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2); d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.final(d1)


def visualize_all(pcd, all_p1, all_p2, all_cls, cls_names):
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geoms = [pcd, axis]
    for p1, p2, cid in zip(all_p1, all_p2, all_cls):
        color = [1, 0, 0] if cls_names[cid] == "0" else [0, 0, 1]
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sph.paint_uniform_color(color)
        sph.translate(p2)
        geoms.append(sph)
        vec = p2 - p1
        L   = np.linalg.norm(vec)
        if L > 1e-6:
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.01,
                cone_radius=0.02,
                cylinder_height=0.7*L,
                cone_height=0.3*L
            )
            arrow.paint_uniform_color([0, 1, 0])
            default = np.array([0,0,1]); d = vec / L
            v = np.cross(default, d); c = np.dot(default, d)
            K = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]])
            R = np.eye(3) + K + K @ K * (1/(1+c))
            arrow.rotate(R, center=(0,0,0))
            arrow.translate(p1)
            geoms.append(arrow)
    vis = o3d.visualization.Visualizer()
    vis.create_window("Bush & Berries", 1024, 768)
    for g in geoms: vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1,0.1,0.1])
    opt.point_size = 2.0; opt.line_width = 5.0
    ctr = vis.get_view_control()
    ctr.set_front([0,-1,0.5]); ctr.set_lookat([0,0,0]); ctr.set_up([0,0,1]); ctr.set_zoom(0.8)
    vis.run(); vis.destroy_window()


def free_camera_view(pcd):
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window("Free-Fly View", 1024, 768)
    scene  = gui.SceneWidget(); scene.scene = rendering.Open3DScene(window.renderer)
    scene.scene.add_geometry("cleaned", pcd, rendering.MaterialRecord())
    aabb = pcd.get_axis_aligned_bounding_box()
    scene.setup_camera(60.0, aabb, aabb.get_center())
    window.add_child(scene)
    gui.Application.instance.run()


def main():
    cams, imgs, points3D = read_model(path=MODEL_DIR, ext=".bin")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo = YOLO(YOLO12_WEIGHTS, verbose=False)
    cls_names = yolo.names
    unet = SmallUNet().to(device)
    unet.load_state_dict(torch.load(UNET_WEIGHTS, map_location=device))
    unet.eval()
    tf = T.Compose([T.ToPILImage(), T.Resize((64,64)), T.ToTensor(),
                    T.Normalize([.485,.456,.406],[.229,.224,.225])])
    pcd = o3d.io.read_point_cloud(POINTCLOUD_PLY)
    os.makedirs(DEBUG_OVERLAY_DIR, exist_ok=True)
    all_p1, all_p2, all_cls = [], [], []
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")) +
                         glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None: continue
        name = Path(img_path).name; overlay = img.copy()
        meta = next((m for m in imgs.values() if m.name == name), None)
        if meta is None: continue
        cam = cams[meta.camera_id]
        K = build_intrinsics(cam)
        feats = np.array(meta.xys); p3d_ids = np.array(meta.point3D_ids)
        results = yolo(img_path)[0]
        ripe_ct = unripe_ct = 0
        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            cid = int(cls)
            if float(conf) < CONF_THRESH: continue
            x1,y1,x2,y2 = map(int, box.tolist())
            if cid == 1:
                unripe_ct += 1
                cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(overlay, "unripe", (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                continue
            ripe_ct += 1
            cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(overlay, "ripe", (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            roi = img[y1:y2, x1:x2]
            inp = tf(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
            with torch.no_grad(): hm = unet(inp).squeeze().cpu().numpy()
            kp_conf = float(hm.max())
            py,px = np.unravel_index(hm.argmax(), hm.shape)
            kx = x1 + int(px * (x2-x1) / hm.shape[1])
            ky = y1 + int(py * (y2-y1) / hm.shape[0])
            cv2.circle(overlay, (kx,ky), 5, (0,0,255), -1)
            circ = detect_circle(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
            if circ:
                crx,cry,_ = circ
                cx,cy = x1+crx, y1+cry
                cv2.circle(overlay, (cx,cy), _, (255,0,0), 2)
                cv2.arrowedLine(overlay, (kx,ky), (cx,cy), (0,255,0), 2, tipLength=0.3)
            else:
                cx = cy = None
            if kp_conf < KEYPOINT_THRESH or circ is None: continue
            d1s = np.linalg.norm(feats - [kx,ky], axis=1)
            id1 = int(p3d_ids[np.argmin(d1s)])
            d2s = np.linalg.norm(feats - [cx,cy], axis=1)
            id2 = int(p3d_ids[np.argmin(d2s)])
            if id1 < 0 or id2 < 0: continue
            p1 = points3D[id1].xyz; p2 = points3D[id2].xyz
            all_p1.append(p1); all_p2.append(p2); all_cls.append(cid)
        cv2.putText(overlay, f"Ripe:{ripe_ct} Unripe:{unripe_ct}", (overlay.shape[1]-220,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)
        cv2.imwrite(os.path.join(DEBUG_OVERLAY_DIR, name), overlay)
    if not all_p2:
        print("No valid detections."); return
    np.save("pred_p1.npy", np.vstack(all_p1))
    np.save("pred_p2.npy", np.vstack(all_p2))
    print(f"Saved pred arrays ({len(all_p1)} entries)")
    visualize_all(pcd, all_p1, all_p2, all_cls, cls_names)
    print("Pick background points then close window.")
    picked = o3d.visualization.draw_geometries_with_editing([pcd])
    cleaned = pcd.select_by_index(picked, invert=True)
    o3d.io.write_point_cloud("cleaned_cloud.ply", cleaned)
    free_camera_view(cleaned)

if __name__ == "__main__":
    main()
