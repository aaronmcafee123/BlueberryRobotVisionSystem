import numpy as np
import cv2
import open3d as o3d
import xml.etree.ElementTree as ET
import os
import glob

def depth_to_point_cloud(depth, fx, fy, cx, cy):
    H, W = depth.shape
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    z = depth
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy
    points = np.stack((x, y, z), axis=-1)
    return points.reshape(-1, 3)

def get_calyx_point_from_xml(xml_path, image_name):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for img_node in root.findall('image'):
        if os.path.basename(img_node.get('name')) == os.path.basename(image_name):
            for child in img_node:
                if child.tag.endswith('points'):
                    pts_str = child.get('points')
                    if not pts_str:
                        continue
                    pair = pts_str.strip().split(';')[0]
                    x_str, y_str = pair.split(',')
                    return (float(x_str), float(y_str))
    return None

# ==== SET THESE ====
img_dir = '/home/aaronmcafee/Documents/blueberryDetect/OriginalDataBlueberryReflect/train'
depth_dir = '/home/aaronmcafee/Documents/blueberryDetect/OriginalDataBlueberryReflect/depth'
xml_path = '/home/aaronmcafee/Documents/blueberryDetect/OriginalDataBlueberryReflect/predictions.xml'
# ===================

img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
print(f"Found {len(img_files)} images.")

for idx, img_path in enumerate(img_files):
    base = os.path.splitext(os.path.basename(img_path))[0]
    possible_depths = [os.path.join(depth_dir, base + ext) for ext in ['.png', '.tiff', '.tif', '.exr']]
    depth_path = None
    for pd in possible_depths:
        if os.path.exists(pd):
            depth_path = pd
            break
    if depth_path is None:
        print(f"[WARN] No depth map for {img_path}")
        continue

    # Load data
    rgb = cv2.imread(img_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        print(f"[WARN] Cannot read depth map: {depth_path}")
        continue
    depth = depth.astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]

    fx, fy = rgb.shape[1], rgb.shape[0]
    cx, cy = rgb.shape[1] / 2, rgb.shape[0] / 2

    # Make point cloud
    points = depth_to_point_cloud(depth, fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if rgb is not None:
        rgb_vis = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB) / 255.0
        colors = rgb_vis.reshape(-1, 3)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Get calyx point from XML (match by full relative path if possible, else just basename)
    calyx_2d = get_calyx_point_from_xml(xml_path, img_path)
    calyx_pcd = None
    calyx_3d = None
    if calyx_2d is not None:
        u, v = int(round(calyx_2d[0])), int(round(calyx_2d[1]))
        if 0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]:
            z = depth[v, u]
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            calyx_3d = np.array([x, y, z])
            calyx_pcd = o3d.geometry.PointCloud()
            calyx_pcd.points = o3d.utility.Vector3dVector([calyx_3d])
            calyx_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
            print(f"[{idx+1}/{len(img_files)}] {img_path}: Calyx 3D = {calyx_3d}")
        else:
            print(f"[WARN] Calyx point ({u},{v}) out of depth map bounds for {img_path}")
    else:
        print(f"[WARN] No calyx for {img_path}")

    # --- Find center with HoughCircles ---
    center_3d = None
    try:
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                   param1=50, param2=30, minRadius=10, maxRadius=0)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            main_circle = max(circles[0, :], key=lambda x: x[2])
            center_u, center_v, center_r = main_circle
            if (0 <= center_v < depth.shape[0]) and (0 <= center_u < depth.shape[1]):
                center_z = depth[center_v, center_u]
                center_x = (center_u - cx) * center_z / fx
                center_y = (center_v - cy) * center_z / fy
                center_3d = np.array([center_x, center_y, center_z])
            else:
                print(f"[WARN] Hough center out of depth bounds for {img_path}")
        else:
            print(f"[WARN] No circle found for {img_path}")
    except Exception as e:
        print(f"[WARN] Hough transform failed: {e}")

    # VISUALIZE
    to_show = [pcd]
    if calyx_pcd is not None:
        to_show.append(calyx_pcd)
    if center_3d is not None:
        # Show center as a green point
        center_pcd = o3d.geometry.PointCloud()
        center_pcd.points = o3d.utility.Vector3dVector([center_3d])
        center_pcd.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
        to_show.append(center_pcd)
    if calyx_3d is not None and center_3d is not None:
        points_line = np.vstack([center_3d, calyx_3d])
        lines = [[0, 1]]
        colors = [[0, 0, 1]]  # Blue
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points_line)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        to_show.append(line_set)

    print("Close window or press N/ESC for next, Q to quit...")
    o3d.visualization.draw_geometries(to_show, window_name=os.path.basename(img_path))
