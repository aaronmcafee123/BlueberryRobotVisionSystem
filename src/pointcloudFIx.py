# pointcloud_fix_binary.py

import numpy as np
from plyfile import PlyData, PlyElement

def spherical_harmonics_to_rgb(f_dc):
    # Clamp to [0,1], gamma-correct, scale to 0–255
    rgb = np.clip(f_dc, 0, 1)
    rgb = np.power(rgb, 1/2.2)
    return (rgb * 255).astype(np.uint8)

# 1) Read the binary PLY
ply = PlyData.read("/home/aaronmcafee/Downloads/blueberry3Edited.ply")
vertex = ply['vertex'].data  # numpy structured array

# 2) Extract positions and SH DC coefficients
xyz = np.vstack((vertex['x'], vertex['y'], vertex['z'])).T
f_dc = np.vstack((vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2'])).T

# 3) Convert SH → RGB
rgb_u8 = spherical_harmonics_to_rgb(f_dc)

# 4) Build new vertex array (with explicit red/green/blue)
new_dtype = [
    ('x','f4'),('y','f4'),('z','f4'),
    ('red','u1'),('green','u1'),('blue','u1')
]
new_verts = np.empty(xyz.shape[0], dtype=new_dtype)
new_verts['x'], new_verts['y'], new_verts['z'] = xyz.T
new_verts['red'], new_verts['green'], new_verts['blue'] = rgb_u8.T

# 5) Write the new ASCII PLY (text=True) or binary (text=False)
new_ply = PlyData([PlyElement.describe(new_verts, 'vertex')], text=True)
new_ply.write("/home/aaronmcafee/Downloads/blueberry3Edited_rgb.ply")
