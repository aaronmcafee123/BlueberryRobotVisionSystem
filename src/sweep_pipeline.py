import os
import sys
import time
import csv
import random

# Get absolute path to current directory and add yolov5 to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
yolov5_path = os.path.join(current_dir, 'yolov5')
sys.path.append(yolov5_path)
import shutil
import argparse
from pathlib import Path
import yaml
import gc
import psutil
import warnings

# Configure for deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------------------------
# 1) DATASET CONFIGURATION
# ------------------------------------------------------------------------------

def generate_data_yaml(dataset_dir: Path):
    """
    Generate data.yaml for an existing split dataset
    """
    dataset_dir = Path(dataset_dir)
    class_ids = set()
    
    # Find class IDs from train labels
    train_label_dir = dataset_dir / 'labels' / 'train'
    for lbl_file in train_label_dir.glob('*.txt'):
        with open(lbl_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_ids.add(int(line.split()[0]))
    
    if not class_ids:
        raise RuntimeError(f"No labels found in {train_label_dir}; cannot build data.yaml")

    nc = max(class_ids) + 1
    names = [str(i) for i in range(nc)]

    data_cfg = {
        'train': str((dataset_dir / 'images' / 'train').resolve()),
        'val':   str((dataset_dir / 'images' / 'val').resolve()),
        'test':  str((dataset_dir / 'images' / 'test').resolve()),
        'nc':    nc,
        'names': names
    }
    yaml_path = dataset_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_cfg, f)
    return str(yaml_path)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_dir', required=True,
                   help='Root directory of pre-split dataset (must contain images/train, images/val, etc.)')
    p.add_argument('--epochs', type=int, default=5,
                   help='Number of training epochs (default: 5)')
    p.add_argument('--batch_size', type=int, default=8,
                   help='Training batch size (default: 8)')
    p.add_argument('--models', type=str, default=None,
                   help='Comma-separated list of specific models to train (e.g., "yolov5n,yolov5s"). If not provided, train all models.')
    p.add_argument('--skip_trained', action='store_true',
                   help='Skip models that already have entries in results.csv')
    return p.parse_args()

# ------------------------------------------------------------------------------
# 2) YOLOv5, YOLOv7/11/12 API SELECTION
# ------------------------------------------------------------------------------
import torch

# Unified YOLO adapter
class YOLOAdapter:
    def __init__(self, model_name: str, ckpt_path: str, device: str = 'cuda'):
        self.model_name = model_name
        self.ckpt_path = ckpt_path
        self.device = device if ':' in device else f'cuda:{device}' if device.isdigit() else device
        self.backend = self._determine_backend()
        self.model = self._load_model()

    def _determine_backend(self):
        # For YOLOv8, YOLOv9, YOLO11, YOLO12, use the ultralytics backend
        if any(v in self.model_name for v in ["yolov8", "yolov9", "yolo11", "yolo12"]):
            return "ultralytics"
        else:
            # Default to v5 for older versions (v5, v7)
            return "v5"

    def _load_model(self):
        if self.backend == "v5":
            # Corrected import paths for YOLOv5 in subdirectory
            from yolov5.models.experimental import attempt_load
            from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
            from yolov5.utils.torch_utils import select_device
            device = select_device(self.device)
            model = attempt_load(self.ckpt_path, device=device)
            return model
        else:
            from ultralytics import YOLO
            return YOLO(self.ckpt_path)

    def train(self, data, epochs, batch_size, project, name, modules_config=None, **kwargs):
        if modules_config:
            print(f"→ Applying module swap: {modules_config}")
            self._swap_modules(modules_config)
            
        if self.backend == "v5":
            from yolov5.train import run
            run(data=data, epochs=epochs, batch_size=batch_size, 
               project=project, name=name, weights=self.ckpt_path, **kwargs)
        else:
            # For Ultralytics models, use 'batch' instead of 'batch_size'
            self.model.train(data=data, epochs=epochs, batch=batch_size, 
                            project=project, name=name, **kwargs)

    def validate(self, data, batch_size, imgsz):
        if self.backend == "v5":
            from yolov5.val import run
            results = run(data=data, weights=self.ckpt_path, 
                         batch_size=batch_size, imgsz=imgsz, device=self.device)
            return results[0]  # (mp, mr, map50, map50_95, *losses)
        else:
            # Use 'batch' instead of 'batch_size' for Ultralytics models
            metrics = self.model.val(data=data, batch=batch_size, 
                                    imgsz=imgsz, device=self.device)
            return [
                metrics.box.map50, 
                metrics.box.map,
                metrics.box.p,
                metrics.box.r,
                metrics.box.fitness
            ]
            
    def _swap_modules(self, config):
        """Swap model modules based on configuration"""
        if self.backend != "ultralytics":
            print("⚠️ Module swapping only supported for Ultralytics models")
            return
            
        from ultralytics.nn.tasks import attempt_load_weights
        from ultralytics.nn.modules import C2f, SPPF, Detect
        
        # Mapping of module types to classes
        module_map = {
            'C2f': C2f,
            'SPPF': SPPF,
            'Detect': Detect
        }
        
        # Load model architecture
        ckpt = torch.load(self.ckpt_path)
        model = ckpt['model']
        
        # Apply module swaps
        for name, module in model.named_modules():
            if name in config:
                new_module = module_map[config[name]]()
                setattr(model, name, new_module)
                
        # Update model
        self.model.model = model
        print(f"✅ Swapped modules: {config}")

# ------------------------------------------------------------------------------
# 3) MAIN SWEEP LOGIC
# ------------------------------------------------------------------------------
def main():
    args = parse_args()

    print("→ Using pre-split dataset from:", args.dataset_dir)
    data_yaml = generate_data_yaml(Path(args.dataset_dir))
    print(f"→ data.yaml written to: {data_yaml}")

    RESULTS_CSV = 'sweep_results.csv'
    EPOCHS     = args.epochs
    BATCH_SIZE = args.batch_size
    DEVICE     = '0'     # use 'cpu' if no GPU

    # --------------------------------------------------------------------------
    # 3.a) AUTO-DISCOVER ALL .pt FILES AND MAP THEM TO “INTERNAL” NAMES
    # --------------------------------------------------------------------------
    global ckpt_map
    ckpt_map = {}  # maps internal_name -> actual file path (string)
    model_files = list(Path('.').glob("*.pt")) + list(Path('.').glob("yolov*.pt"))
    
    for p in model_files:
        stem = p.stem
        # Handle all known YOLO variants
        if stem.startswith("yolov5"):
            ckpt_map[stem] = p.as_posix()
        elif stem.startswith("yolov6"):
            ckpt_map[stem] = p.as_posix()
        elif stem.startswith("yolov7"):
            ckpt_map[stem] = p.as_posix()
        elif stem.startswith("yolov8"):
            ckpt_map[stem] = p.as_posix()
        elif stem.startswith("yolov9"):
            ckpt_map[stem] = p.as_posix()
        elif stem.startswith("yolo11") or stem.startswith("yolo12"):
            ckpt_map[stem] = p.as_posix()
        elif stem.startswith("yolol1"):
            ckpt_map[f"yolo11{stem[len('yolol1'):]}"] = p.as_posix()
        elif stem.startswith("yolol2"):
            ckpt_map[f"yolo12{stem[len('yolol2'):]}"] = p.as_posix()
        else:
            print(f"→ Found unknown model file: {p.name}")

    if not ckpt_map:
        print("→ No .pt files found in the current directory. Exiting.")
        return

    print("→ Discovered checkpoint files (internal_name → filepath):")
    for internal, path in ckpt_map.items():
        print(f"    {internal}  →  {path}")

    # --------------------------------------------------------------------------
    # 3.b) ORGANIZE MODELS BY VERSION
    # --------------------------------------------------------------------------
    version_groups = {
        'v12': [f"yolo12{sz}" for sz in ['n', 's', 'm', 'l', 'x']],
        'v11': [f"yolo11{sz}" for sz in ['n', 's', 'm', 'l', 'x']],
        'v9':  [f"yolov9{sz}" for sz in ['n', 's', 'm', 'l', 'x']],
        'v8':  [f"yolov8{sz}" for sz in ['n', 's', 'm', 'l', 'x']]
    }
    
    # Map aliases to actual model names
    alias_map = {
        'yolol1': 'yolo11',
        'yolol2': 'yolo12'
    }
    
    # Create version-specific model lists
    versioned_models = {version: [] for version in version_groups}
    
    # Process discovered model files
    for p in model_files:
        stem = p.stem.lower()
        
        # Handle aliases
        for alias, real_prefix in alias_map.items():
            if stem.startswith(alias):
                stem = stem.replace(alias, real_prefix)
                
        # Assign to version groups
        for version, prefixes in version_groups.items():
            for prefix in prefixes:
                if stem == prefix.lower():
                    versioned_models[version].append(prefix)
                    break
    
    # Create interleaved base_models list
    base_models = []
    max_len = max(len(v) for v in versioned_models.values())
    
    for i in range(max_len):
        for version in ['v12', 'v11', 'v9', 'v8']:
            if i < len(versioned_models[version]):
                model = versioned_models[version][i]
                if model not in base_models:  # Avoid duplicates
                    base_models.append(model)
                    
    # Filter models if specific models are requested
    if args.models:
        selected_models = [m.strip() for m in args.models.split(',')]
        base_models = [m for m in base_models if m in selected_models]
        if not base_models:
            print(f"→ No models matched your selection: {selected_models}")
            return
        print(f"→ Training selected models: {base_models}")
    
    if not base_models:
        print("→ No valid YOLO models found. Exiting.")
        return

    print("→ Interleaved base_models order:")
    print("   ", base_models)

    # --------------------------------------------------------------------------
    # 3.c) PAIR EACH base_model WITH BOTH HYPERPARAMETER SETS
    # --------------------------------------------------------------------------
    hparam_sets = [
        {'lr0': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005},
        {'lr0': 0.005,'momentum': 0.9,   'weight_decay': 0.0005},
    ]
    variants = [(m, hp) for m in base_models for hp in hparam_sets]

    print(f"→ Total variants to run: {len(variants)} (model × hyperparam). First three:")
    print("   ", variants[:3])

    # --------------------------------------------------------------------------
    # 3.d) VERSION-SPECIFIC MODULE SWAPPING CONFIGURATIONS
    # --------------------------------------------------------------------------
    # Only apply module swapping to compatible versions (v11/v12)
    module_configs = {
        'v11': [
            None,
            {'backbone': 'C2f', 'neck': 'SPPF'},
            {'backbone': 'C2f', 'head': 'Detect'},
            {'neck': 'SPPF', 'head': 'Detect'}
        ],
        'v12': [
            None,
            {'backbone': 'C2f', 'neck': 'SPPF'},
            {'backbone': 'C2f', 'head': 'Detect'},
            {'neck': 'SPPF', 'head': 'Detect'}
        ]
    }
    
    # For other versions, only use baseline
    default_configs = [None]
    
    # For other versions, only use baseline
    default_configs = [None]
    
    # --------------------------------------------------------------------------
    # 3.e) INITIALIZE RESULTS CSV (APPEND MODE)
    # --------------------------------------------------------------------------
    # Create CSV with header only if file doesn't exist
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'model', 'config', 'modules', 
                'map50', 'map50_95', 'precision', 'recall', 'fitness',
                'params', 'inference_time'
            ])
            
    # Load existing results to skip trained models
    trained_models = set()
    if args.skip_trained and os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Create unique identifier for model+config combination
                model_id = f"{row['model']}-{row['config']}-{row['modules']}"
                trained_models.add(model_id)
        print(f"→ Found {len(trained_models)} existing model configurations in results.csv")

    # --------------------------------------------------------------------------
    # 4) LOOP THROUGH EACH VARIANT
    # --------------------------------------------------------------------------
    variants_expanded = []
    for model_name, hparams in variants:
        # Determine version group
        version = None
        if model_name.startswith('yolo11'):
            version = 'v11'
        elif model_name.startswith('yolo12'):
            version = 'v12'
        elif model_name.startswith('yolov7'):
            version = 'v7'
        elif model_name.startswith('yolov8'):
            version = 'v8'
        elif model_name.startswith('yolov5'):
            version = 'v5'
            
        # Select appropriate configs
        configs = module_configs.get(version, default_configs)
        
        for mod_config in configs:
            variants_expanded.append((model_name, hparams, mod_config))
            
    print(f"→ Total variants to run: {len(variants_expanded)} (model × hyperparam × module_config). First three:")
    print("   ", variants_expanded[:3])

    # Track index separately since we might skip some variants
    run_idx = 0
    for variant in variants_expanded:
        model_name, hparams, mod_config = variant
        model_id = f"{model_name}-{str(hparams)}-{str(mod_config)}"
        
        # Skip already trained models if requested
        if args.skip_trained and model_id in trained_models:
            print(f"→ Skipping already trained: {model_id}")
            continue
            
        try:
            run_name = f"{model_name}_lr{hparams['lr0']:.3f}_idx{run_idx}"
            run_idx += 1
            run_dir = Path('runs/sweep') / run_name
            if run_dir.exists():
                shutil.rmtree(str(run_dir))

            device_str = DEVICE
            print(f"\n→ Training {model_name} with hparams={hparams} as run='{run_name}'")
            # Create YOLOAdapter instance with error handling
            try:
                adapter = YOLOAdapter(model_name, ckpt_map[model_name], device_str)
            except Exception as e:
                print(f"→ Error loading model {model_name}: {str(e)}")
                continue
            
            # Train with module configuration
            adapter.train(
                data=data_yaml,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                project='runs/sweep',
                name=run_name,
                exist_ok=False,
                imgsz=640,
                modules_config=mod_config,
                **hparams
            )

            best_weights = f"runs/sweep/{run_name}/weights/best.pt"
            if not Path(best_weights).exists():
                print(f"→ Warning: Best weights not found at {best_weights}, using last.pt")
                best_weights = f"runs/sweep/{run_name}/weights/last.pt"
                
            print(f"→ Validating {model_name} from {best_weights}")
            # Validate using the adapter
            metrics = adapter.validate(
                data=data_yaml,
                batch_size=BATCH_SIZE,
                imgsz=640
            )
            
            # Extract metrics
            if metrics:
                if adapter.backend == "v5":
                    # v5 returns: (mp, mr, map50, map50_95, *losses)
                    map50 = metrics[2]
                    map50_95 = metrics[3]
                    precision = metrics[0]
                    recall = metrics[1]
                    fitness = metrics[4] if len(metrics) > 4 else 0
                else:
                    # Ultralytics returns: [map50, map50_95, precision, recall, fitness]
                    map50 = metrics[0]
                    map50_95 = metrics[1]
                    precision = metrics[2]
                    recall = metrics[3]
                    fitness = metrics[4]
            else:
                print("→ Validation failed, using placeholder metrics")
                map50, map50_95, precision, recall, fitness = 0,0,0,0,0

        except Exception as e:
            print(f"→ Error training {model_name}: {str(e)}")
            # Skip to next model if this one fails
            continue

            # PARAM COUNT & INFERENCE TIME (with error handling)
            try:
                params = sum(p.numel() for p in adapter.model.model.parameters())
                # Get the actual device from model parameters
                device = next(adapter.model.model.parameters()).device
                x = torch.zeros((1, 3, 640, 640), device=device)

                # Warmup and measure inference time
                _ = adapter.model(x)  # Warmup

                times = []
                for _ in range(100):
                    t0 = time.time()
                    _ = adapter.model(x)
                    times.append(time.time() - t0)
                inf_time = sum(times) / len(times)
            except Exception as e:
                print(f"→ Error measuring performance: {str(e)}")
                params, inf_time = 0, 0

        # Clean up memory
        del adapter
        gc.collect()
        torch.cuda.empty_cache()
        
        # Check memory usage
        mem = psutil.virtual_memory()
        if mem.percent > 90:
            warnings.warn(f"High memory usage: {mem.percent}%. Consider reducing batch size.")
        
        # LOG TO CSV
        with open(RESULTS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                model_name, 
                str(hparams), 
                str(mod_config) if mod_config else 'baseline',
                map50, 
                map50_95,
                precision,
                recall,
                fitness,
                params, 
                inf_time
            ])

        print(f"→ Logged {model_name} ({mod_config}): mAP50={map50:.4f}, mAP50-95={map50_95:.4f}, params={params}, inf_time={inf_time:.4f}s")

    print(f"\nSweep complete! Results saved to {RESULTS_CSV}\n")


if __name__ == '__main__':
    main()
