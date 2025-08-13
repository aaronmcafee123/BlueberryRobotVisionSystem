#!/usr/bin/env python3
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
import plotly.graph_objects as go

# -- Orientation Estimation --------------------------------------------------
def estimate_berry_orientation(center_2d, calyx_2d, radius_px):
    x = (calyx_2d[0] - center_2d[0]) / float(radius_px)
    y = (calyx_2d[1] - center_2d[1]) / float(radius_px)
    r2 = min(x*x + y*y, 1.0)
    z = np.sqrt(1.0 - r2)
    d = np.array([x, y, z], dtype=float)
    d /= np.linalg.norm(d)
    roll = np.arctan2(d[0], d[2])
    pitch = np.arcsin(d[1])
    return {
        "direction_vector": d,
        "roll_deg": np.degrees(roll),
        "pitch_deg": np.degrees(pitch),
        "roll_rad": roll,
        "pitch_rad": pitch,
    }

# -- Circle Detection --------------------------------------------------------
def circle_from_3pts(p1, p2, p3):
    x1,y1 = p1; x2,y2 = p2; x3,y3 = p3
    A = np.array([[2*(x2-x1), 2*(y2-y1)], [2*(x3-x1), 2*(y3-y1)]], dtype=float)
    b = np.array([
        x2*x2 - x1*x1 + y2*y2 - y1*y1,
        x3*x3 - x1*x1 + y3*y3 - y1*y1
    ], dtype=float)
    try:
        cx, cy = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    r = np.hypot(x1-cx, y1-cy)
    return (cx, cy, r)

def detect_berry_circle_ransac(gray, calyx_pt, iterations=200, thresh=2, min_ratio=0.3):
    edges = cv2.Canny(gray, 100, 200)
    ys, xs = np.where(edges > 0)
    pts = np.column_stack((xs, ys))
    if pts.shape[0] < 50:
        return None
    cx, cy = map(int, map(round, calyx_pt))
    pts = pts[np.hypot(pts[:,0]-cx, pts[:,1]-cy) > 10]
    if pts.shape[0] < 50:
        return None
    best_circle, best_inliers = None, 0
    for _ in range(iterations):
        idx = np.random.choice(pts.shape[0], 3, replace=False)
        tri = [tuple(pts[i]) for i in idx]
        circ = circle_from_3pts(*tri)
        if circ is None:
            continue
        xc, yc, r = circ
        inliers = np.sum(
            np.abs(np.hypot(pts[:,0]-xc, pts[:,1]-yc) - r) < thresh
        )
        if inliers > best_inliers:
            best_circle, best_inliers = circ, inliers
    if best_circle and best_inliers > min_ratio * pts.shape[0]:
        xc, yc, r = best_circle
        return (int(round(xc)), int(round(yc)), int(round(r)))
    return None

def detect_berry_circle(gray, calyx_pt=None):
    if calyx_pt is not None:
        c = detect_berry_circle_ransac(gray, calyx_pt)
        if c is not None:
            return c
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=20,
        param1=100, param2=30,
        minRadius=10, maxRadius=0
    )
    if circles is not None:
        x, y, r = circles[0,0]
        return (int(round(x)), int(round(y)), int(round(r)))
    return None

# -- XML Parsing -------------------------------------------------------------
def get_calyx_point_from_xml(img_node):
    for child in img_node:
        if 'points' in child.tag and child.get('points'):
            x_str, y_str = child.get('points').split(';')[0].split(',')
            return (float(x_str), float(y_str))
    return None

# -- 2D Annotation -----------------------------------------------------------
def annotate(img, center, calyx, stem, orient):
    # Ensure pure Python ints for drawing
    # center: (x0, y0, r)
    x0, y0, r = center
    cx0 = int(x0)
    cy0 = int(y0)
    radius_int = int(r)
    # original calyx scar
    calx = int(round(calyx[0]))
    caly = int(round(calyx[1]))
    # stem/opposite point
    stx = int(round(stem[0]))
    sty = int(round(stem[1]))

    vis = img.copy()
    # Draw circle outline
    cv2.circle(vis, (cx0, cy0), radius_int, (255, 0, 0), 2)
    # Mark center, scar, and stem
    cv2.circle(vis, (cx0, cy0), 6, (0, 255, 0), -1)
    cv2.circle(vis, (calx, caly),    6, (0, 0, 255), -1)
    cv2.circle(vis, (stx, sty),      6, (255, 255, 0), -1)
    # Axis arrow from scar to stem
    cv2.arrowedLine(vis, (calx, caly), (stx, sty), (0, 255, 255), 2, tipLength=0.12)
    return vis

# -- Interactive 3D Visualization -------------------------------------------
def visualize_orientation_interactive(direction, image_name, output_dir):
    phi = np.linspace(0, np.pi, 30)
    theta = np.linspace(0, 2 * np.pi, 60)
    phi, theta = np.meshgrid(phi, theta)
    X = np.sin(phi) * np.cos(theta)
    Y = np.sin(phi) * np.sin(theta)
    Z = np.cos(phi)
    vec = direction / np.linalg.norm(direction)
    fig = go.Figure()
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=0.3, showscale=False))
    fig.add_trace(go.Cone(
        x=[0], y=[0], z=[0],
        u=[vec[0]], v=[vec[1]], w=[vec[2]],
        sizemode='absolute', sizeref=0.5, anchor='tail'
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1, 1]), yaxis=dict(range=[-1, 1]), zaxis=dict(range=[-1, 1])
        ),
        title=f'3D Orientation: {image_name}',
        width=700,
        height=700
    )
    out_dir = os.path.join(output_dir, 'interactive_3d')
    os.makedirs(out_dir, exist_ok=True)
    fig.write_html(os.path.join(
        out_dir,
        f"3d_{os.path.splitext(image_name)[0]}.html"
    ))

# -- Main Processing ---------------------------------------------------------
def process_image(img_path, calyx, output_dir, verbose=False):
    img = cv2.imread(img_path)
    if img is None:
        if verbose:
            print(f"[WARN] Cannot read: {img_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    c0 = detect_berry_circle(gray, calyx)
    if c0 is None:
        if verbose:
            print(f"[WARN] No circle in {os.path.basename(img_path)}")
        return
    x0, y0, r = c0
    # project calyx onto circle edge if outside
    dx, dy = calyx[0] - x0, calyx[1] - y0
    dist = np.hypot(dx, dy)
    if dist > r:
        dx *= r / dist; dy *= r / dist
        calyx_proj = (x0 + dx, y0 + dy)
    else:
        calyx_proj = calyx
    orient = estimate_berry_orientation((x0, y0), calyx_proj, r)
    stem_pt = (2*x0 - int(round(calyx[0])), 2*y0 - int(round(calyx[1])))

    # Annotate and skip on drawing error
    try:
        vis2d = annotate(img, (x0, y0, r), calyx, stem_pt, orient)
    except cv2.error as e:
        if verbose:
            print(f"[WARN] Annotation failed for {os.path.basename(img_path)}: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), vis2d)
    # interactive visualization
    try:
        visualize_orientation_interactive(orient['direction_vector'], os.path.basename(img_path), output_dir)
    except Exception:
        if verbose:
            print(f"[WARN] 3D viz failed for {os.path.basename(img_path)}")
        # continue without crash

    if verbose:
        print(f"{os.path.basename(img_path)} -> Roll {orient['roll_deg']:+.1f}, Pitch {orient['pitch_deg']:+.1f}")


def main(pred_xml, img_dir, output_dir, verbose=False):
    tree = ET.parse(pred_xml)
    root = tree.getroot()
    for img_node in root.findall('image'):
        calyx = get_calyx_point_from_xml(img_node)
        if calyx is None:
            continue
        img_name = img_node.get('name')
        img_path = os.path.join(img_dir, os.path.basename(img_name))
        process_image(img_path, calyx, output_dir, verbose)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Blueberry orientation')
    parser.add_argument('--pred_xml',   required=True)
    parser.add_argument('--img_dir',    required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--verbose',    action='store_true')
    args = parser.parse_args()
    main(args.pred_xml, args.img_dir, args.output_dir, args.verbose)
