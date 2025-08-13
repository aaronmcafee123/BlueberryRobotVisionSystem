import cv2
import numpy as np
import trimesh

# ========== USER INPUTS ==========
obj_path = '/home/aaronmcafee/Documents/blueberryDetect/OriginalDataBlueberryReflect/fr_gourmetBlueberry.obj'       # <<< FILL IN >>>
output_path = 'overlayed_result.jpg'  # <<< FILL IN >>>

# 3D coordinates (in object/model space) for center and calyx
center_3d = [-0.000407, 0.007439, 0.000024]                 # <<< FILL IN >>>
calyx_3d = [-0.001597, 0.001289, -0.00007]                  # <<< FILL IN >>>
#fill1 = [0, 0, 0]
#fill1 = [1, 1, 1]

# 2D coordinates in the image for center and calyx
center_2d = [320, 240]                # <<< FILL IN >>>
calyx_2d = [350, 180]                 # <<< FILL IN >>>
#fill1 = [0, 0, 0]
#fill1 = [1, 1, 1]

import cv2
import numpy as np

# ========== USER INPUTS ==========
image_path = '/home/aaronmcafee/Documents/blueberryDetect/OriginalDataBlueberryReflect/train/6a449ddf497e8a9a7986a209ad8ddd50_jpg.rf.62946297b6e17a59869f560d5053b65c_blueberry_1.jpg'         # <<< FILL IN >>>
output_path = 'berry_direction_out.jpg'  # <<< FILL IN >>>
# 2D image coordinates (pixels)
center_2d = [1, 1]                  # <<< FILL IN >>>
calyx_2d = [1, 0]                  # <<< FILL IN >>>
radius_px = 80                        # <<< FILL IN (pixels, circle fit radius) >>>
# ==================================

def estimate_calyx_3d_direction(center_2d, calyx_2d, radius_px):
    x = (calyx_2d[0] - center_2d[0]) / radius_px
    y = (calyx_2d[1] - center_2d[1]) / radius_px
    xy_sq = x**2 + y**2
    if xy_sq > 1:
        print("Calyx point is outside the sphere's projection!")
        return None
    z = np.sqrt(1 - xy_sq)
    direction = np.array([x, y, z])
    direction /= np.linalg.norm(direction)
    theta = np.arccos(z)                 # polar angle (radians)
    phi = np.arctan2(y, x)               # azimuth (radians)
    return {
        'direction_vector': direction,
        'theta_deg': np.degrees(theta),
        'phi_deg': np.degrees(phi),
        'theta_rad': theta,
        'phi_rad': phi
    }

if __name__ == "__main__":
    # 1. Estimate direction
    result = estimate_calyx_3d_direction(center_2d, calyx_2d, radius_px)
    if result is None:
        print("Estimation failed.")
        exit()

    print("3D direction (from berry center to calyx, camera frame):", result['direction_vector'])
    print(f"Polar angle theta: {result['theta_deg']:.2f} deg")
    print(f"Azimuth angle phi: {result['phi_deg']:.2f} deg")

    # 2. Visualize on image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image {image_path}")

    # Draw berry circle
    center_int = tuple(map(int, center_2d))
    cv2.circle(img, center_int, int(radius_px), (255, 0, 0), 2)

    # Draw center point
    cv2.circle(img, center_int, 7, (0,255,0), -1)
    cv2.putText(img, 'Center', (center_int[0]+10, center_int[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Draw calyx point
    calyx_int = tuple(map(int, calyx_2d))
    cv2.circle(img, calyx_int, 7, (0,0,255), -1)
    cv2.putText(img, 'Calyx', (calyx_int[0]+10, calyx_int[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # Draw arrow from center to calyx
    cv2.arrowedLine(img, center_int, calyx_int, (0,0,255), 2, tipLength=0.18)

    # Show theta/phi
    cv2.putText(img, f"Theta: {result['theta_deg']:.1f} deg", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.putText(img, f"Phi: {result['phi_deg']:.1f} deg", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    # Save result
    cv2.imwrite(output_path, img)
    print(f"Saved visualization to {output_path}")
