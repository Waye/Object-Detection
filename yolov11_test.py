# Comprehensive Thermal Model Testing Framework - YOLOv11
# Testing YOLOv11 Models on FLIR Thermal Dataset

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import cv2
from pathlib import Path
import subprocess
import shutil
from datetime import datetime
import time
import glob
from tqdm import tqdm
import sys
import traceback
import yaml
import requests
import tempfile

# Set environment variable to avoid OpenMP errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# YOLO11 model URLs
YOLO11S_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"
YOLO11M_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt"

# Set paths to your dataset
DATA_DIR = os.path.join(os.getcwd(), "data")
OUTPUT_DIR = os.path.join(os.getcwd(), "results")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
ANNOTATIONS_FILE = os.path.join(DATA_DIR, "annotations.json")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

print(f"Data directory: {DATA_DIR}")
print(f"Images directory: {IMAGE_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# Global variables to track dataset and model preparation
DATASET_PREPARED = False
MODELS_DOWNLOADED = False
YOLO11_MODELS = {}
filtered_coco_data = None

# Define annotation file paths
train_annot_file = os.path.join(OUTPUT_DIR, "train_annotations.json")
val_annot_file = os.path.join(OUTPUT_DIR, "val_annotations.json")
filtered_annot_file = os.path.join(OUTPUT_DIR, "filtered_annotations.json")

def download_model_from_url(url, model_name):
    """Download a model from URL"""
    try:
        model_dir = os.path.join(OUTPUT_DIR, "models")
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"{model_name}.pt")
        
        # Check if model already exists
        if os.path.exists(model_path):
            print(f"Model {model_name} already exists at {model_path}")
            return model_path
        
        print(f"Downloading {model_name} from {url}...")
        
        # Download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as f, tqdm(
            desc=model_name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        
        print(f"Successfully downloaded {model_name} to {model_path}")
        return model_path
        
    except Exception as e:
        print(f"Error downloading {model_name}: {str(e)}")
        return None

def download_all_yolo11_models():
    """Download all required YOLOv11 models"""
    global MODELS_DOWNLOADED, YOLO11_MODELS
    
    if MODELS_DOWNLOADED:
        print("YOLOv11 models already downloaded, skipping.")
        return YOLO11_MODELS
    
    print("Downloading all required YOLOv11 models...")
    
    # Ensure ultralytics is installed
    try:
        from ultralytics import YOLO
        print("Ultralytics already installed")
    except ImportError:
        subprocess.run(["pip", "install", "--user", "ultralytics>=8.1.0"], check=False)
        try:
            from ultralytics import YOLO
            print("Installed ultralytics")
        except ImportError:
            print("Failed to install ultralytics")
            return {}
    
    models = {}
    model_configs = {
        'yolo11s': YOLO11S_URL,
        'yolo11m': YOLO11M_URL
    }
    
    for model_name, url in model_configs.items():
        try:
            model_path = download_model_from_url(url, model_name)
            if model_path and os.path.exists(model_path):
                model = YOLO(model_path)
                models[model_name] = model
                print(f"Successfully loaded {model_name}")
            else:
                print(f"Failed to download {model_name}")
        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")
    
    MODELS_DOWNLOADED = True
    YOLO11_MODELS = models
    
    print(f"Downloaded {len(models)} YOLOv11 models: {list(models.keys())}")
    return models

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    print(f"Free GPU memory: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Load the COCO annotations
print(f"Loading annotations from: {ANNOTATIONS_FILE}")
try:
    with open(ANNOTATIONS_FILE, 'r') as f:
        full_coco_data = json.load(f)
    print(f"Successfully loaded annotations with {len(full_coco_data.get('images', []))} images and {len(full_coco_data.get('annotations', []))} annotations")
except FileNotFoundError:
    print(f"Error: Annotations file not found at {ANNOTATIONS_FILE}")
    full_coco_data = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "object"}]}
except json.JSONDecodeError:
    print(f"Error: Invalid JSON format in annotations file {ANNOTATIONS_FILE}")
    full_coco_data = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "object"}]}
except Exception as e:
    print(f"Error loading annotations: {str(e)}")
    full_coco_data = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "object"}]}

def prepare_available_images_dataset():
    """Filter annotations to only include available images and create new COCO file"""
    global DATASET_PREPARED, filtered_coco_data, filtered_annot_file
    
    filtered_annot_file = os.path.join(OUTPUT_DIR, "filtered_annotations.json")
    
    if DATASET_PREPARED and filtered_coco_data is not None:
        print("Using already prepared dataset from memory.")
        return filtered_coco_data, filtered_annot_file
    
    if os.path.exists(filtered_annot_file):
        print("Loading existing filtered annotations file.")
        try:
            with open(filtered_annot_file, 'r') as f:
                filtered_coco_data = json.load(f)
            DATASET_PREPARED = True
            return filtered_coco_data, filtered_annot_file
        except Exception as e:
            print(f"Error loading prepared dataset file: {str(e)}")
    
    print("Preparing dataset with only available images...")
    
    available_files = set()
    
    if not os.path.exists(IMAGE_DIR):
        print(f"Warning: Image directory {IMAGE_DIR} does not exist. Creating it.")
        os.makedirs(IMAGE_DIR, exist_ok=True)
    
    print(f"Searching for images in: {IMAGE_DIR}")
    
    # List all image files in the directory
    for root, dirs, files in os.walk(IMAGE_DIR):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                relative_path = os.path.relpath(os.path.join(root, file), IMAGE_DIR)
                available_files.add(relative_path)
    
    print(f"Found {len(available_files)} image files in directory")
    
    # If no images found, check if there are images in the dataset directory above
    if len(available_files) == 0:
        print("No images found in the images directory. Checking if images exist in the data directory...")
        for root, dirs, files in os.walk(DATA_DIR):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    source_path = os.path.join(root, file)
                    dest_path = os.path.join(IMAGE_DIR, file)
                    print(f"Copying {source_path} to {dest_path}")
                    try:
                        shutil.copy2(source_path, dest_path)
                        relative_path = file
                        available_files.add(relative_path)
                    except Exception as e:
                        print(f"Error copying image: {str(e)}")
        
        # Check the img directory if it exists
        img_dir = os.path.join(os.getcwd(), "img")
        if os.path.exists(img_dir):
            print(f"Checking for images in {img_dir}...")
            for root, dirs, files in os.walk(img_dir):
                for file in files:
                    if file.endswith(('.jpg', '.jpeg', '.png')):
                        source_path = os.path.join(root, file)
                        dest_path = os.path.join(IMAGE_DIR, file)
                        print(f"Copying {source_path} to {dest_path}")
                        try:
                            shutil.copy2(source_path, dest_path)
                            relative_path = file
                            available_files.add(relative_path)
                        except Exception as e:
                            print(f"Error copying image: {str(e)}")
                            
        print(f"After searching additional directories, found {len(available_files)} image files")
    
    available_basenames = {os.path.basename(f) for f in available_files}
    
    # Filter images to only those available in the directory
    available_images = []
    available_image_ids = set()
    
    for img in full_coco_data['images']:
        file_name = img['file_name']
        basename = os.path.basename(file_name)
        
        if file_name in available_files or basename in available_basenames:
            img['file_name'] = basename
            available_images.append(img)
            available_image_ids.add(img['id'])
    
    print(f"Filtered to {len(available_images)} images with annotations")
    
    # Filter annotations to only those for available images
    available_annotations = [
        ann for ann in full_coco_data['annotations'] 
        if ann['image_id'] in available_image_ids
    ]
    
    print(f"Keeping {len(available_annotations)} annotations for available images")
    
    # Create new COCO data with only available images
    filtered_coco_data = {
        'images': available_images,
        'annotations': available_annotations,
        'categories': full_coco_data['categories']
    }
    
    # Save filtered annotation file
    with open(filtered_annot_file, 'w') as f:
        json.dump(filtered_coco_data, f)
    
    print(f"Created filtered annotation file with {len(available_images)} images")
    
    DATASET_PREPARED = True
    return filtered_coco_data, filtered_annot_file

# Filter annotations to available images
filtered_coco_data, filtered_annot_file = prepare_available_images_dataset()
coco_data = filtered_coco_data

# Extract categories
categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
print(f"Categories: {categories}")

def log_result(model_name, metrics):
    """Log results to file"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(OUTPUT_DIR, f"{model_name}_results_{timestamp}.txt")
    
    with open(log_file, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("-" * 50 + "\n")
        
        for metric_name, value in metrics.items():
            f.write(f"{metric_name}: {value}\n")
    
    print(f"Results saved to {log_file}")
    
    # Also append to master results file
    master_file = os.path.join(OUTPUT_DIR, "all_results.csv")
    metrics_df = pd.DataFrame({
        'Model': [model_name],
        'Timestamp': [timestamp],
        **{k: [v] for k, v in metrics.items()}
    })
    
    if os.path.exists(master_file):
        master_df = pd.read_csv(master_file)
        master_df = pd.concat([master_df, metrics_df], ignore_index=True)
    else:
        master_df = metrics_df
    
    master_df.to_csv(master_file, index=False)

# Check if images are grayscale
if len(coco_data['images']) > 0:
    sample_img_filename = coco_data['images'][0]['file_name']
    sample_img_path = os.path.join(IMAGE_DIR, sample_img_filename)
    
    if not os.path.exists(sample_img_path):
        for root, dirs, files in os.walk(IMAGE_DIR):
            for file in files:
                if file == os.path.basename(sample_img_filename):
                    sample_img_path = os.path.join(root, file)
                    break
    
    if os.path.exists(sample_img_path):
        sample_img = cv2.imread(sample_img_path)
        if sample_img is not None:
            print(f"Sample image shape: {sample_img.shape}")
            if len(sample_img.shape) == 2 or sample_img.shape[2] == 1:
                print("Images are already in grayscale format")
                is_grayscale = True
            else:
                print("Images are in RGB format, will convert to grayscale for thermal models")
                is_grayscale = False
        else:
            print(f"Could not read image at {sample_img_path}")
            is_grayscale = None
    else:
        print(f"Sample image not found at {sample_img_path}")
        is_grayscale = None
else:
    print("No images found in the filtered dataset")
    is_grayscale = None

# Prepare train/val splits and YAML configuration
yaml_file = os.path.join(OUTPUT_DIR, "thermal_dataset.yaml")

# Check if files already exist
if not (os.path.exists(train_annot_file) and os.path.exists(val_annot_file) and os.path.exists(yaml_file)):
    from sklearn.model_selection import train_test_split
    
    image_ids = [img['id'] for img in coco_data['images']]
    
    if len(image_ids) == 0:
        print("No images found in the dataset. Creating empty train/val sets.")
        train_ids = []
        val_ids = []
    else:
        train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
    
    # Create split annotation files
    train_data = {
        'images': [img for img in coco_data['images'] if img['id'] in train_ids],
        'annotations': [ann for ann in coco_data['annotations'] if ann['image_id'] in train_ids],
        'categories': coco_data['categories']
    }
    
    val_data = {
        'images': [img for img in coco_data['images'] if img['id'] in val_ids],
        'annotations': [ann for ann in coco_data['annotations'] if ann['image_id'] in val_ids],
        'categories': coco_data['categories']
    }
    
    # Save split files
    with open(train_annot_file, 'w') as f:
        json.dump(train_data, f)
    
    with open(val_annot_file, 'w') as f:
        json.dump(val_data, f)
    
    # Create YAML config
    yaml_content = f"""
# Dataset paths
path: {OUTPUT_DIR}  # Root directory

# Directory structure
train_dir: {IMAGE_DIR}  # Directory containing training images
val_dir: {IMAGE_DIR}  # Directory containing validation images
test_dir:  # Directory containing test images (optional)

# Annotations (COCO format)
train: {os.path.relpath(train_annot_file, OUTPUT_DIR)}  # Path to train annotations
val: {os.path.relpath(val_annot_file, OUTPUT_DIR)}  # Path to validation annotations
test:  # Path to test annotations (optional)

# Single class mode - all objects are treated as class 0
nc: 1  # Number of classes (1 for simplified single-class mode)
names: ['object']  # Class name

# Original class definitions (for reference)
original_classes:
  {os.linesep.join([f'  {i}: {categories[cat_id]}' for i, cat_id in enumerate(categories)])}
"""
    
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"Dataset prepared: {len(train_ids)} training images, {len(val_ids)} validation images")
else:
    print("Dataset split files already exist, skipping preparation")

# Global model performance tracker
all_results = {}

def test_yolo11(model_size='s'):
    """Test YOLOv11 model with specified size variant"""
    assert model_size in ['s', 'm'], "Only 's' and 'm' variants are supported"
    global YOLO11_MODELS
    
    # Install ultralytics if needed
    try:
        from ultralytics import YOLO
        print("Ultralytics already installed")
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "--user", "ultralytics"], check=False)
        try:
            from ultralytics import YOLO
            print("Installed ultralytics")
        except ImportError:
            print("Failed to install ultralytics, skipping this test")
            return {"mAP50": 0, "mAP50-95": 0, "Error": "Installation failed"}
    
    model_name = f"yolo11{model_size}"
    yolo_output = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(yolo_output, exist_ok=True)
    
    # Create YOLO dataset structure
    dataset_root = os.path.join(yolo_output, "dataset")
    os.makedirs(dataset_root, exist_ok=True)
    
    # Create train and val directories with their respective subdirectories
    for split in ['train', 'val']:
        split_images = os.path.join(dataset_root, split, 'images')
        split_labels = os.path.join(dataset_root, split, 'labels')
        os.makedirs(split_images, exist_ok=True)
        os.makedirs(split_labels, exist_ok=True)
    
    # Load and process annotations
    print("\nProcessing annotations...")
    for split in ['train', 'val']:
        annot_file = os.path.join(OUTPUT_DIR, f"{split}_annotations.json")
        print(f"Processing {split} annotations from {annot_file}")
        
        try:
            with open(annot_file, 'r') as f:
                split_data = json.load(f)
            
            print(f"Loaded {len(split_data['images'])} images and {len(split_data['annotations'])} annotations for {split}")
            
            # Create image to annotation mapping
            image_annotations = {}
            for ann in split_data['annotations']:
                img_id = ann['image_id']
                if img_id not in image_annotations:
                    image_annotations[img_id] = []
                image_annotations[img_id].append(ann)
            
            # Process each image
            for img in split_data['images']:
                img_id = img['id']
                img_file = img['file_name']
                img_width = img['width']
                img_height = img['height']
                
                # Source image path
                src_img = os.path.join(IMAGE_DIR, img_file)
                if not os.path.exists(src_img):
                    print(f"Warning: Image not found: {src_img}")
                    continue
                
                # Copy image to dataset
                dst_img = os.path.join(dataset_root, split, 'images', img_file)
                shutil.copy2(src_img, dst_img)
                
                # Create YOLO format label file
                label_file = os.path.join(dataset_root, split, 'labels', 
                                        os.path.splitext(img_file)[0] + '.txt')
                
                # Write annotations in YOLO format
                if img_id in image_annotations:
                    with open(label_file, 'w') as f:
                        for ann in image_annotations[img_id]:
                            # Get bbox coordinates
                            x, y, w, h = ann['bbox']
                            
                            # Convert to YOLO format (normalized coordinates)
                            x_center = (x + w/2) / img_width
                            y_center = (y + h/2) / img_height
                            width = w / img_width
                            height = h / img_height
                            
                            # Always use class 0 for single class detection
                            f.write(f"0 {x_center} {y_center} {width} {height}\n")
            
            # Verify the conversion
            n_images = len(glob.glob(os.path.join(dataset_root, split, 'images', '*')))
            n_labels = len(glob.glob(os.path.join(dataset_root, split, 'labels', '*.txt')))
            print(f"{split} set: {n_images} images, {n_labels} label files")
            
        except Exception as e:
            print(f"Error processing {split} annotations: {str(e)}")
            return {"mAP50": 0, "mAP50-95": 0, "Error": f"Annotation processing error: {str(e)}"}
    
    # Create YAML configuration
    yaml_content = {
        'path': os.path.abspath(dataset_root),
        'train': os.path.join('train', 'images'),
        'val': os.path.join('val', 'images'),
        'nc': 1,  # Single class
        'names': ['object']
    }
    
    yaml_path = os.path.join(yolo_output, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"\nCreated dataset configuration at {yaml_path}")
    
    # Load or download model
    try:
        if model_name in YOLO11_MODELS and YOLO11_MODELS[model_name] is not None:
            model = YOLO11_MODELS[model_name]
            print(f"Using cached {model_name} model")
        else:
            # Download model if not cached
            url = YOLO11S_URL if model_size == 's' else YOLO11M_URL
            model_path = download_model_from_url(url, model_name)
            if model_path:
                model = YOLO(model_path)
                print(f"Loaded {model_name} model")
            else:
                raise Exception(f"Failed to download {model_name}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return {"mAP50": 0, "mAP50-95": 0, "Error": f"Model loading error: {str(e)}"}
    
    # Train the model
    try:
        print("\nStarting YOLOv11 training...")
        start_time = time.time()
        
        results = model.train(
            data=yaml_path,
            epochs=30,
            imgsz=640,
            batch=16,
            device=0 if torch.cuda.is_available() else 'cpu',
            project=yolo_output,
            name='train',
            exist_ok=True,
            pretrained=True,
            single_cls=True,
            verbose=True,
            val=True,
            seed=42,
            patience=5,
            save_period=5,
            save=True,
            cache=True,
            amp=True,
            # Training hyperparameters
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            # Data augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
        )
        
        # Calculate training time
        training_time = (time.time() - start_time) / 60.0
        
        # Get metrics
        metrics = {}
        try:
            if hasattr(results, 'results_dict'):
                metrics_dict = results.results_dict
                print("\nYOLOv11 Training Results:")
                for key, value in metrics_dict.items():
                    print(f"{key}: {value}")
                
                metrics["mAP50"] = float(metrics_dict.get('metrics/mAP50(B)', 0))
                metrics["mAP50-95"] = float(metrics_dict.get('metrics/mAP50-95(B)', 0))
                metrics["Precision"] = float(metrics_dict.get('metrics/precision(B)', 0))
                metrics["Recall"] = float(metrics_dict.get('metrics/recall(B)', 0))
                metrics["Box Loss"] = float(metrics_dict.get('val/box_loss', 0))
            else:
                print("No metrics found in results dictionary")
                metrics = {
                    "mAP50": 0,
                    "mAP50-95": 0,
                    "Precision": 0,
                    "Recall": 0,
                    "Box Loss": 0,
                    "Error": "No metrics available"
                }
        except Exception as e:
            print(f"Error extracting metrics: {str(e)}")
            metrics = {
                "mAP50": 0,
                "mAP50-95": 0,
                "Precision": 0,
                "Recall": 0,
                "Box Loss": 0,
                "Error": f"Metric extraction error: {str(e)}"
            }
        
        # Calculate additional metrics
        metrics["Parameters (M)"] = sum(p.numel() for p in model.model.parameters() if p.requires_grad) / 1e6
        metrics["Training Time (min)"] = training_time
        metrics["GFLOPs"] = "N/A"
        
        # Calculate inference time
        if torch.cuda.is_available():
            # Warm up
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            for _ in range(5):
                model.predict(img)
            
            # Measure inference time
            t0 = time.time()
            iterations = 50
            for _ in range(iterations):
                model.predict(img)
            torch.cuda.synchronize()
            
            inference_time = (time.time() - t0) * 1000 / iterations
            metrics["Inference Time (ms)"] = inference_time
            metrics["GPU Memory (GB)"] = torch.cuda.max_memory_allocated() / 1e9
        else:
            metrics["Inference Time (ms)"] = "N/A"
            metrics["GPU Memory (GB)"] = "N/A"
        
        # Save metrics to yolo11.csv
        csv_path = os.path.join(OUTPUT_DIR, 'yolo11.csv')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare metrics row
        metrics_row = {
            'Timestamp': timestamp,
            'Model': model_name,
            'mAP50': metrics['mAP50'],
            'mAP50-95': metrics['mAP50-95'],
            'Parameters (M)': metrics['Parameters (M)'],
            'GFLOPs': metrics['GFLOPs'],
            'Inference Time (ms)': metrics['Inference Time (ms)'],
            'Training Time (min)': metrics['Training Time (min)'],
            'GPU Memory (GB)': metrics['GPU Memory (GB)'],
            'Error': metrics.get('Error', '')
        }
        
        # Read existing CSV or create new one
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            df = pd.DataFrame(columns=metrics_row.keys())
        
        # Append new row
        df = pd.concat([df, pd.DataFrame([metrics_row])], ignore_index=True)
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"\nMetrics saved to {csv_path}")
        
        # Print summary
        print("\nYOLOv11 Training Summary:")
        print(f"Model: {model_name}")
        print(f"mAP50: {metrics['mAP50']:.4f}")
        print(f"mAP50-95: {metrics['mAP50-95']:.4f}")
        print(f"Parameters (M): {metrics['Parameters (M)']:.2f}")
        print(f"Training Time: {metrics['Training Time (min)']:.2f} minutes")
        print(f"Inference Time: {metrics['Inference Time (ms)']}")
        print(f"GPU Memory: {metrics['GPU Memory (GB)']} GB")
        
        return metrics
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return {"mAP50": 0, "mAP50-95": 0, "Error": f"Training error: {str(e)}"}

def run_all_yolo11_tests():
    """Run all YOLOv11 model tests and compile results"""
    results = {}
    
    # Test YOLOv11 models
    try:
        print("\n--- Testing YOLOv11-s ---")
        results['YOLOv11-s'] = test_yolo11('s')
    except Exception as e:
        print(f"Error testing YOLOv11-s: {e}")
        results['YOLOv11-s'] = {"mAP50": 0, "mAP50-95": 0, "Error": str(e)}
    
    try:
        print("\n--- Testing YOLOv11-m ---")
        results['YOLOv11-m'] = test_yolo11('m')
    except Exception as e:
        print(f"Error testing YOLOv11-m: {e}")
        results['YOLOv11-m'] = {"mAP50": 0, "mAP50-95": 0, "Error": str(e)}
    
    return results

def visualize_yolo11_results(results):
    """Create visualization of YOLOv11 model performance"""
    # Create DataFrame from results
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'mAP50': [results[model].get('mAP50', 0) for model in results],
        'mAP50-95': [results[model].get('mAP50-95', 0) for model in results],
        'Parameters (M)': [results[model].get('Parameters (M)', 0) for model in results],
        'GFLOPs': [results[model].get('GFLOPs', 'N/A') for model in results],
        'Inference Time (ms)': [results[model].get('Inference Time (ms)', 0) for model in results],
        'Training Time (min)': [results[model].get('Training Time (min)', 'N/A') for model in results],
        'GPU Memory (GB)': [results[model].get('GPU Memory (GB)', 'N/A') for model in results],
        'Error': [results[model].get('Error', '') for model in results]
    })
    
    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, 'yolo11_model_comparison.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"YOLOv11 Results saved to {csv_path}")
    
    # Check if we have valid data to visualize
    if metrics_df.empty or metrics_df['mAP50'].max() == 0:
        print("No valid YOLOv11 performance metrics to visualize. Skipping visualization.")
        return metrics_df
    
    # Create visualizations
    try:
        plt.figure(figsize=(15, 10))
        
        # 1. Accuracy (mAP50) comparison
        plt.subplot(2, 2, 1)
        bars = plt.bar(metrics_df['Model'], metrics_df['mAP50'], color='darkblue')
        plt.title('YOLOv11 mAP50 Comparison', fontsize=14)
        plt.ylabel('mAP50')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', rotation=0)
        
        # 2. Model Size vs. Accuracy
        plt.subplot(2, 2, 2)
        plt.scatter(metrics_df['Parameters (M)'], metrics_df['mAP50'], 
                   s=100, color='darkblue', alpha=0.7)
        
        # Add model names as labels
        for i, model in enumerate(metrics_df['Model']):
            plt.annotate(model, (metrics_df['Parameters (M)'].iloc[i], metrics_df['mAP50'].iloc[i]),
                        textcoords="offset points", xytext=(0, 10), ha='center')
        
        plt.title('YOLOv11 Model Size vs. Accuracy', fontsize=14)
        plt.xlabel('Parameters (M)')
        plt.ylabel('mAP50')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 3. Inference Speed vs. Accuracy
        plt.subplot(2, 2, 3)
        plt.scatter(metrics_df['Inference Time (ms)'], metrics_df['mAP50'], 
                   s=100, color='darkblue', alpha=0.7)
        
        # Add model names as labels
        for i, model in enumerate(metrics_df['Model']):
            plt.annotate(model, (metrics_df['Inference Time (ms)'].iloc[i], metrics_df['mAP50'].iloc[i]),
                        textcoords="offset points", xytext=(0, 10), ha='center')
        
        plt.title('YOLOv11 Inference Speed vs. Accuracy', fontsize=14)
        plt.xlabel('Inference Time (ms)')
        plt.ylabel('mAP50')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 4. Training Time vs. Accuracy
        plt.subplot(2, 2, 4)
        plt.scatter(metrics_df['Training Time (min)'], metrics_df['mAP50'], 
                   s=100, color='darkblue', alpha=0.7)
        
        # Add model names as labels
        for i, model in enumerate(metrics_df['Model']):
            plt.annotate(model, (metrics_df['Training Time (min)'].iloc[i], metrics_df['mAP50'].iloc[i]),
                        textcoords="offset points", xytext=(0, 10), ha='center')
        
        plt.title('YOLOv11 Training Time vs. Accuracy', fontsize=14)
        plt.xlabel('Training Time (min)')
        plt.ylabel('mAP50')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'yolo11_model_comparison_charts.png'))
        plt.close()
        
        print("YOLOv11 visualization charts saved")
        
    except Exception as e:
        print(f"Error creating YOLOv11 visualizations: {e}")
        traceback.print_exc()
    
    return metrics_df

def main():
    """Main function to run YOLOv11 thermal vision model tests"""
    global DATASET_PREPARED, MODELS_DOWNLOADED
    
    # Setup output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if we can use CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        free_mem = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        print(f"Free GPU memory: {free_mem / 1e9:.2f} GB")
    
    # Prepare dataset
    if not DATASET_PREPARED:
        print("\n--- Preparing dataset for YOLOv11 ---")
        filtered_coco_data, filtered_annot_file = prepare_available_images_dataset()
        DATASET_PREPARED = True
    else:
        print("\n--- Dataset already prepared for YOLOv11 ---")
    
    # Download all YOLOv11 models
    if not MODELS_DOWNLOADED:
        print("\n--- Downloading YOLOv11 models ---")
        models = download_all_yolo11_models()
        MODELS_DOWNLOADED = True
        print(f"Available YOLOv11 models: {list(models.keys())}")
    else:
        print("\n--- YOLOv11 models already downloaded ---")
    
    # Define models to test
    selected_tests = ["yolo11s", "yolo11m"]  # Both YOLOv11 variants
    
    print("Running YOLOv11 tests...")
    
    # Store results of all model tests
    results = {}
    
    # Run tests
    for model_name in selected_tests:
        try:
            print(f"\n--- Testing {model_name.upper()} ---")
            
            if model_name == "yolo11s":
                model_results = test_yolo11('s')
            elif model_name == "yolo11m":
                model_results = test_yolo11('m')
            else:
                continue
            
            # Save results
            results[f"YOLOv11-{model_name[-1]}"] = model_results
            
        except KeyboardInterrupt:
            print("Testing interrupted by user")
            break
        except Exception as e:
            print(f"Error testing {model_name}: {str(e)}")
            traceback.print_exc()
            results[f"YOLOv11-{model_name[-1]}"] = {"Error": str(e)}
    
    # Generate summary plot and comparison
    if results:
        try:
            visualize_yolo11_results(results)
        except Exception as e:
            print(f"Error visualizing YOLOv11 results: {str(e)}")
            traceback.print_exc()
    
    print("\nYOLOv11 testing complete!")
    return results

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nYOLOv11 script interrupted by user")
    except Exception as e:
        print(f"Error in YOLOv11 main script: {str(e)}")
        traceback.print_exc()