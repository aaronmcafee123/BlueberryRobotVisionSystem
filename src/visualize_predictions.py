import os
import cv2
import xml.etree.ElementTree as ET
import argparse

def draw_points_on_image(img, points, color=(0,255,0), radius=5, thickness=2):
    for pt in points:
        cv2.circle(img, (int(round(pt[0])), int(round(pt[1]))), radius, color, thickness)
    return img

def main(pred_xml, img_dir, vis_dir):
    tree = ET.parse(pred_xml)
    root = tree.getroot()
    os.makedirs(vis_dir, exist_ok=True)

    for img_node in root.findall('image'):
        img_name = img_node.get('name')
        img_path = os.path.join(img_dir, os.path.basename(img_name))
        if not os.path.exists(img_path):
            print(f"[WARN] Image not found: {img_path}")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue

        # Get predicted points
        pred_points = []
        for child in img_node:
            if 'points' in child.tag:
                pts_str = child.get('points')
                if not pts_str:
                    continue
                for pair in pts_str.split(';'):
                    pair = pair.strip()
                    if not pair:
                        continue
                    if ',' not in pair:
                        continue
                    x_str, y_str = pair.split(',')
                    pred_points.append((float(x_str), float(y_str)))

        if not pred_points:
            continue

        img_vis = draw_points_on_image(img, pred_points, color=(0,255,0), radius=5, thickness=2)
        out_path = os.path.join(vis_dir, os.path.basename(img_name))
        cv2.imwrite(out_path, img_vis)
        print(f"Saved visualization: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_xml', required=True, help='Predicted XML file (from HarrisPredict.py)')
    parser.add_argument('--img_dir', required=True, help='Directory containing images')
    parser.add_argument('--vis_dir', required=True, help='Output directory for visualized images')
    args = parser.parse_args()
    main(args.pred_xml, args.img_dir, args.vis_dir)
