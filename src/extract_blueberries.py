import os
import cv2
import numpy as np
import shutil

def yolo_to_pixels(bbox, img_width, img_height):
    """Convert YOLO format [x_center, y_center, width, height] to pixel coordinates [x_min, y_min, x_max, y_max]"""
    x_center, y_center, width, height = bbox
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    x_min = int(x_center - width/2)
    y_min = int(y_center - height/2)
    x_max = int(x_center + width/2)
    y_max = int(y_center + height/2)
    
    return x_min, y_min, x_max, y_max

def main():
    base_dir = "OriginalData"
    output_base = "OriginalDataBlueberryReflect"
    
    # Process all splits: train, val, test
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(base_dir, "images", split)
        
        # Skip if directory doesn't exist
        if not os.path.exists(img_dir):
            continue
            
        depth_dir = os.path.join(base_dir, "depth", split)  # Depth map directory
        label_dir = os.path.join(base_dir, "labels", split)
        output_dir = os.path.join(output_base, split)
        depth_output_dir = os.path.join(output_base, split, "depth")  # Depth output directory
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(depth_output_dir, exist_ok=True)
        
        # Process each image
        for img_file in os.listdir(img_dir):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(img_dir, img_file)
                base_name = os.path.splitext(img_file)[0]
                label_path = os.path.join(label_dir, f"{base_name}.txt")
                
                # Skip if no label file exists
                if not os.path.exists(label_path):
                    continue
                
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                img_height, img_width = img.shape[:2]
                
                # Load corresponding depth map if exists
                depth_path = os.path.join(depth_dir, img_file.replace(".jpg", ".png"))
                depth_map = None
                if os.path.exists(depth_path):
                    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                
                # Process each label
                with open(label_path, "r") as f:
                    lines = f.readlines()
                
                blueberry_count = 0
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    bbox = list(map(float, parts[1:5]))
                    
                    # Only process ripe blueberries (class 0)
                    if class_id == 0:
                        x_min, y_min, x_max, y_max = yolo_to_pixels(
                            bbox, img_width, img_height
                        )
                        
                        # Crop blueberry
                        blueberry = img[y_min:y_max, x_min:x_max]
                        
                        if blueberry.size > 0:
                            # Save cropped blueberry
                            output_path = os.path.join(
                                output_dir, 
                                f"{base_name}_blueberry_{blueberry_count}.jpg"
                            )
                            cv2.imwrite(output_path, blueberry)
                            
                            # Save cropped depth map if exists
                            if depth_map is not None:
                                depth_blueberry = depth_map[y_min:y_max, x_min:x_max]
                                if depth_blueberry.size > 0:
                                    depth_output_path = os.path.join(
                                        depth_output_dir, 
                                        f"{base_name}_blueberry_{blueberry_count}.png"
                                    )
                                    cv2.imwrite(depth_output_path, depth_blueberry)
                            
                            blueberry_count += 1

if __name__ == "__main__":
    main()
