#!/usr/bin/env python3
"""
Simplified RF-DETR Evaluation Script
This version focuses on loading custom RT-DETR models and provides better debugging
"""
import os
import json
import tempfile
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Configuration
ORIG_CLASS_IDS = [0, 1]
CLASS_COLORS = {0: 'red', 1: 'blue'}
COCO_ID_MAP = {0: 1, 1: 2}

def load_and_analyze_checkpoint(model_path):
    """Load and analyze checkpoint structure"""
    print(f"Loading checkpoint: {model_path}")
    
    try:
        # First try with weights_only=True (safer)
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            print(f"✓ Checkpoint loaded successfully (weights_only=True)")
        except Exception as e:
            print(f"Failed with weights_only=True: {e}")
            print("Trying with weights_only=False (less secure but more compatible)...")
            
            # Add safe globals for common objects found in checkpoints
            torch.serialization.add_safe_globals([argparse.Namespace])
            
            # Try with weights_only=False
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            print(f"✓ Checkpoint loaded successfully (weights_only=False)")
        
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # Look for model weights
            model_keys = ['model', 'state_dict', 'model_state_dict', 'net', 'network', 'model_state', 'ema']
            model_data = None
            
            for key in model_keys:
                if key in checkpoint:
                    print(f"Found model data under key: '{key}'")
                    model_data = checkpoint[key]
                    break
            
            if model_data is None:
                print("Model data not found in standard keys, checking all keys...")
                # Check if any key contains a state dict
                for key, value in checkpoint.items():
                    if isinstance(value, dict) and any('weight' in k or 'bias' in k for k in value.keys()):
                        print(f"Found potential model data under key: '{key}'")
                        model_data = value
                        break
                
                if model_data is None:
                    print("Using checkpoint directly as model data")
                    model_data = checkpoint
            
            if isinstance(model_data, dict):
                param_names = list(model_data.keys())
                print(f"Model has {len(param_names)} parameters")
                print(f"First few parameters: {param_names[:5]}")
                
                # Try to infer model type from parameter names
                backbone_indicators = ['backbone', 'resnet', 'swin', 'vit']
                decoder_indicators = ['decoder', 'transformer', 'detr']
                
                has_backbone = any(any(ind in name.lower() for ind in backbone_indicators) for name in param_names)
                has_decoder = any(any(ind in name.lower() for ind in decoder_indicators) for name in param_names)
                
                print(f"Model analysis:")
                print(f"  - Has backbone components: {has_backbone}")
                print(f"  - Has decoder/transformer components: {has_decoder}")
                
                # Look for RT-DETR specific components
                rtdetr_indicators = ['rtdetr', 'hybrid_encoder', 'intra_scale_feat', 'cross_scale_feat']
                has_rtdetr = any(any(ind in name.lower() for ind in rtdetr_indicators) for name in param_names)
                print(f"  - Has RT-DETR specific components: {has_rtdetr}")
                
            # Check for additional info
            if 'epoch' in checkpoint:
                print(f"Training epoch: {checkpoint['epoch']}")
            if 'optimizer' in checkpoint:
                print(f"Has optimizer state")
            if 'lr_scheduler' in checkpoint:
                print(f"Has learning rate scheduler state")
            if 'args' in checkpoint:
                print(f"Has training arguments")
                if hasattr(checkpoint['args'], '__dict__'):
                    args_dict = vars(checkpoint['args'])
                    print(f"  Some args: {list(args_dict.keys())[:5]}")
                
        return checkpoint
        
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        return None

def create_dummy_model(num_classes=2):
    """Create a dummy model for testing"""
    class DummyRTDETR(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.num_classes = num_classes
            
        def forward(self, x):
            batch_size = x.size(0)
            device = x.device
            
            # Return some dummy detections for testing
            # In practice, you'd want to remove this and implement real inference
            num_detections = 2  # dummy detections per image
            
            boxes = torch.tensor([
                [100, 100, 50, 50],  # [x1, y1, w, h]
                [200, 200, 60, 60]
            ], dtype=torch.float32, device=device).repeat(batch_size, 1, 1)
            
            scores = torch.tensor([0.8, 0.7], dtype=torch.float32, device=device).repeat(batch_size, 1)
            labels = torch.tensor([0, 1], dtype=torch.long, device=device).repeat(batch_size, 1)
            
            return {
                'boxes': boxes.view(-1, 4),
                'scores': scores.view(-1),
                'labels': labels.view(-1)
            }
    
    return DummyRTDETR(num_classes)

def load_rtdetr_model(model_path, device='cuda'):
    """Load RT-DETR model with multiple fallback options"""
    
    # First analyze the checkpoint
    checkpoint = load_and_analyze_checkpoint(model_path)
    if checkpoint is None:
        raise ValueError("Could not load checkpoint")
    
    # Try method 1: Ultralytics RT-DETR
    try:
        from ultralytics import RTDETR
        print("Trying Ultralytics RT-DETR...")
        model = RTDETR(model_path)
        print("✓ Successfully loaded with Ultralytics")
        return model, 'ultralytics'
    except Exception as e:
        print(f"Ultralytics failed: {e}")
    
    # Try method 2: Official RT-DETR
    try:
        print("Trying official RT-DETR...")
        import sys
        # Add RT-DETR repo path - adjust this path as needed
        rtdetr_paths = ['RT-DETR', '../RT-DETR', '../../RT-DETR', 'rtdetr_pytorch']
        for path in rtdetr_paths:
            if os.path.exists(path):
                sys.path.insert(0, path)
                break
        
        from rtdetr_pytorch import RTDETR as RTDETRModel
        
        model = RTDETRModel()
        
        # Try different checkpoint loading methods
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'ema' in checkpoint:
            model.load_state_dict(checkpoint['ema'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        print("✓ Successfully loaded with official RT-DETR")
        return model, 'official'
        
    except Exception as e:
        print(f"Official RT-DETR failed: {e}")
    
    # Try method 3: YOLOv8 style loading (for ultralytics checkpoints)
    try:
        print("Trying YOLOv8 style loading...")
        from ultralytics import YOLO
        
        # Sometimes RT-DETR checkpoints are compatible with YOLO loading
        model = YOLO(model_path)
        print("✓ Successfully loaded with YOLO interface")
        return model, 'yolo'
        
    except Exception as e:
        print(f"YOLO loading failed: {e}")
    
    # Try method 4: Custom/Generic approach
    try:
        print("Trying custom model loading...")
        
        # For now, create a dummy model for testing
        # You would replace this with your actual model architecture
        print("⚠️  Using dummy model - replace with actual model architecture")
        model = create_dummy_model(num_classes=2)
        model.to(device)
        model.eval()
        
        return model, 'dummy'
        
    except Exception as e:
        print(f"Custom loading failed: {e}")
        raise e

def preprocess_image(image_path, target_size=(640, 640)):
    """Preprocess image for inference"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Standard preprocessing for RT-DETR
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, original_size

def run_inference(model, model_type, image_path, device='cuda'):
    """Run inference on a single image"""
    
    if model_type in ['ultralytics', 'yolo']:
        # Use ultralytics interface
        results = model(image_path)
        return results[0]
    
    elif model_type == 'official':
        # Use official RT-DETR interface
        image_tensor, original_size = preprocess_image(image_path)
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
        
        return outputs
    
    elif model_type == 'dummy':
        # Use dummy model
        image_tensor, original_size = preprocess_image(image_path)
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
        
        return outputs
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def test_single_image(model, model_type, image_path, device='cuda'):
    """Test inference on a single image"""
    print(f"Testing inference on: {image_path}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"✗ Image not found: {image_path}")
        return False
    
    try:
        start_time = time.time()
        outputs = run_inference(model, model_type, image_path, device)
        inference_time = time.time() - start_time
        
        print(f"✓ Inference completed in {inference_time:.3f}s")
        
        # Analyze outputs
        if model_type in ['ultralytics', 'yolo']:
            if hasattr(outputs, 'boxes') and outputs.boxes is not None:
                num_detections = len(outputs.boxes)
                print(f"  Detections: {num_detections}")
                if num_detections > 0:
                    print(f"  Confidence scores: {outputs.boxes.conf.cpu().numpy()}")
                    print(f"  Classes: {outputs.boxes.cls.cpu().numpy()}")
            else:
                print("  No detections found")
        
        else:
            if isinstance(outputs, dict):
                if 'boxes' in outputs:
                    num_detections = len(outputs['boxes'])
                    print(f"  Detections: {num_detections}")
                    if num_detections > 0:
                        print(f"  Confidence scores: {outputs['scores'].cpu().numpy()}")
                        print(f"  Classes: {outputs['labels'].cpu().numpy()}")
                else:
                    print("  No detections in output")
            else:
                print(f"  Output type: {type(outputs)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Simplified RF-DETR Evaluation')
    parser.add_argument('--model', required=True, help='Path to .pth model file')
    parser.add_argument('--test-image', help='Path to single test image')
    parser.add_argument('--image', help='Path to single test image (alternative flag)')
    parser.add_argument('--images', help='Path to test images directory')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Handle both --test-image and --image flags
    test_image = args.test_image or args.image
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    print("=== Simplified RF-DETR Evaluation ===")
    print(f"Device: {args.device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Load model
    try:
        model, model_type = load_rtdetr_model(args.model, args.device)
        print(f"✓ Model loaded successfully (type: {model_type})")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test with single image
    if test_image:
        success = test_single_image(model, model_type, test_image, args.device)
        if success:
            print("✓ Single image test passed")
        else:
            print("✗ Single image test failed")
    
    # Test with image directory
    if args.images:
        image_dir = Path(args.images)
        if not image_dir.exists():
            print(f"✗ Image directory not found: {args.images}")
            return
            
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpeg'))
        
        if image_files:
            print(f"Found {len(image_files)} images")
            # Test first few images
            for i, img_path in enumerate(image_files[:3]):
                print(f"\nTesting image {i+1}/{min(3, len(image_files))}: {img_path.name}")
                success = test_single_image(model, model_type, str(img_path), args.device)
                if not success:
                    break
        else:
            print("No images found in directory")
    
    if not test_image and not args.images:
        print("No test images specified. Use --image or --images flag.")

if __name__ == '__main__':
    main()