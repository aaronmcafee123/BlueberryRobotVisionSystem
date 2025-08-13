import os
import shutil
from pathlib import Path

def merge_datasets():
    # Define paths
    blueberry_path = Path("/home/aaronmcafee/Documents/blueberryDetect/original/Blueberry.v3i.yolov11")
    archive_path = Path("/home/aaronmcafee/Documents/blueberryDetect/original/archive")
    output_path = Path("/home/aaronmcafee/merged_dataset1")
    
    # Create output directories
    for split in ["train", "test", "valid"]:
        for folder in ["images", "labels"]:
            os.makedirs(output_path / split / folder, exist_ok=True)
    
    # Process Blueberry dataset
    process_dataset(blueberry_path, output_path, "blueberry")
    
    # Process Archive datasets (all subdirectories)
    for dataset_dir in archive_path.iterdir():
        if dataset_dir.is_dir():
            process_dataset(dataset_dir, output_path, "archive")

def process_dataset(source_path, output_path, prefix):
    # Define label mappings
    if "blueberry" in prefix:
        # Blueberry dataset mapping: ripe=0, unripe=1, background=2
        label_map = {0: 0, 1: 1, 2: 2}
    else:
        # Archive datasets mapping: ripe=1 → 0, unripe=0 → 1, background=2 → 2
        label_map = {0: 1, 1: 0, 2: 2}

    for split in ["train", "test", "valid"]:
        # Process images
        image_dir = source_path / split / "images"
        if image_dir.exists():
            for image_file in image_dir.iterdir():
                if image_file.is_file() and image_file.suffix in ['.jpg', '.png', '.jpeg']:
                    new_name = f"{prefix}_{image_file.name}"
                    dest = output_path / split / "images" / new_name
                    shutil.copy2(image_file, dest)
        
        # Process corresponding labels with mapping
        label_dir = source_path / split / "labels"
        if label_dir.exists():
            for label_file in label_dir.iterdir():
                if label_file.is_file() and label_file.suffix == '.txt':
                    new_name = f"{prefix}_{label_file.name}"
                    dest = output_path / split / "labels" / new_name
                    
                    # Read and update labels
                    updated_lines = []
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.split()
                            if parts:  # Skip empty lines
                                class_id = int(parts[0])
                                # Map class ID using label_map, default to original if not found
                                mapped_id = label_map.get(class_id, class_id)
                                parts[0] = str(mapped_id)
                                updated_lines.append(" ".join(parts))
                    
                    # Write updated labels
                    with open(dest, 'w') as f:
                        f.write("\n".join(updated_lines))

if __name__ == "__main__":
    merge_datasets()
    print("Datasets merged successfully!")
