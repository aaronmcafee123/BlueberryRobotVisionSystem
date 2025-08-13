#!/usr/bin/env python3
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from torch.utils.data import DataLoader
from torchvision import transforms as T
from small_unet import SmallUNet
from detect_utils import detect_circle

# Configuration
IMAGE_DIR = "/path/to/images"
OUTPUT_DIR = "/path/to/output"
YOLO_MODEL_PATH = "/home/aaronmcafee/ros2_ws/src/yolov11_ros2/yolov11_ros2/best.pt"
HEATMAP_PATH = "/home/aaronmcafee/ros2_ws/src/yolov11_ros2/yolov11_ros2/unet_epoch200.pth"
CONF_THRESHOLD = 0.6

# Create output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

yolo = YOLO(YOLO_MODEL_PATH, verbose=False)

# Heatmap model
hm_model = SmallUNet()
hm_model.load_state_dict(torch.load(HEATMAP_PATH, map_location=device))
hm_model.to(device).eval()
hm_tf = T.Compose([
    T.ToPILImage(),
    T.Resize((64,64)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# Processing
for img_path in Path(IMAGE_DIR).glob('*.*'):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    overlay = img.copy()

    # YOLO detections
    results = yolo(img, verbose=False)[0]
    for idx, (box, conf, cls) in enumerate(zip(
            results.boxes.xyxy,
            results.boxes.conf,
            results.boxes.cls)):

        if float(conf) < CONF_THRESHOLD or int(cls) != 0:
            continue

        x1,y1,x2,y2 = map(int, box.tolist())
        cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 2)

        # heatmap keypoint
        roi = img[y1:y2, x1:x2]
        inp = hm_tf(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        with torch.no_grad():
            hm = hm_model(inp).squeeze().cpu().numpy()
        py,px = np.unravel_index(hm.argmax(), hm.shape)
        kx = int(px*(x2-x1)/hm.shape[1] + x1)
        ky = int(py*(y2-y1)/hm.shape[0] + y1)
        cv2.circle(overlay, (kx,ky), 5, (0,0,255), -1)

        # circle detection
        circ = detect_circle(
            cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        )
        if circ:
            crx,cry,r = circ
            cx_img, cy_img = x1+crx, y1+cry
            cv2.circle(overlay, (cx_img,cy_img), r, (255,0,0), 2)
            cv2.arrowedLine(overlay, (kx,ky), (cx_img,cy_img),
                            (0,255,255), 2, tipLength=0.1)

    # Save overlay
    out_path = Path(OUTPUT_DIR) / img_path.name
    cv2.imwrite(str(out_path), overlay)
    print(f"Processed and saved: {out_path}")
