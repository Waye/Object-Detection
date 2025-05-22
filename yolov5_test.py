# YOLOv5 Thermal Model Testing Framework
# Testing YOLOv5s and YOLOv5m Models on FLIR Thermal Dataset

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
from urllib.parse import urlparse
import pickle  # Add pickle module for model loading

# Set environment variable to avoid OpenMP errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
YOLOV5_MODELS = {}
filtered_coco_data = None

# YOLOv5 model URLs
YOLOV5_MODEL_URLS = {
    'yolov5s': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt',
    'yolov5m': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt'
}

def download_file(url, filename):
    """Download a file from URL"""
    print(f"Downloading {filename} from {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Successfully downloaded {filename}")
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")
        return False

def setup_yolov5_environment():
    """Setup YOLOv5 environment"""
    yolov5_dir = os.path.join(os.getcwd(), "yolov5")
    
    # Check if YOLOv5 directory exists
    if not os.path.exists(yolov5_dir):
        print("Cloning YOLOv5 repository...")
        try:
            subprocess.run([
                "git", "clone", "https://github.com/ultralytics/yolov5.git", yolov5_dir
            ], check=True)
            print("YOLOv5 repository cloned successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning YOLOv5 repository: {e}")
            return False
    else:
        print("YOLOv5 repository already exists")
    
    # Install requirements
    requirements_file = os.path.join(yolov5_dir, "requirements.txt")
    if os.path.exists(requirements_file):
        print("Installing YOLOv5 requirements...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", requirements_file
            ], check=False)  # Don't fail if some packages already exist
            print("YOLOv5 requirements installed")
        except Exception as e:
            print(f"Warning: Error installing requirements: {e}")
    
    # Add YOLOv5 to Python path
    if yolov5_dir not in sys.path:
        sys.path.append(yolov5_dir)
    
    return yolov5_dir

def download_yolov5_models():
    """Download YOLOv5 model weights"""
    global MODELS_DOWNLOADED, YOLOV5_MODELS
    
    if MODELS_DOWNLOADED:
        print("YOLOv5 models already downloaded, skipping.")
        return YOLOV5_MODELS
    
    print("Downloading YOLOv5 models...")
    
    # Setup YOLOv5 environment
    yolov5_dir = setup_yolov5_environment()
    if not yolov5_dir:
        print("Failed to setup YOLOv5 environment")
        return {}
    
    models = {}
    
    # Download model weights
    for model_name, url in YOLOV5_MODEL_URLS.items():
        model_path = os.path.join(OUTPUT_DIR, f"{model_name}.pt")
        
        if os.path.exists(model_path):
            print(f"{model_name} model already exists at {model_path}")
        else:
            if download_file(url, model_path):
                print(f"Downloaded {model_name} model")
            else:
                print(f"Failed to download {model_name} model")
                continue
        
        models[model_name] = model_path
    
    MODELS_DOWNLOADED = True
    YOLOV5_MODELS = models
    
    print(f"Downloaded {len(models)} models: {list(models.keys())}")
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
    global DATASET_PREPARED, filtered_coco_data
    
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
    
    if len(available_files) == 0:
        print("No images found. Checking additional directories...")
        # Check data directory and img directory like the original code
        for check_dir in [DATA_DIR, os.path.join(os.getcwd(), "img")]:
            if os.path.exists(check_dir):
                print(f"Checking for images in {check_dir}...")
                for root, dirs, files in os.walk(check_dir):
                    for file in files:
                        if file.endswith(('.jpg', '.jpeg', '.png')):
                            source_path = os.path.join(root, file)
                            dest_path = os.path.join(IMAGE_DIR, file)
                            print(f"Copying {source_path} to {dest_path}")
                            try:
                                shutil.copy2(source_path, dest_path)
                                available_files.add(file)
                            except Exception as e:
                                print(f"Error copying image: {str(e)}")
    
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

def convert_coco_to_yolo_format(coco_json_path, output_dir):
    """Convert COCO format annotations to YOLO format for YOLOv5"""
    print(f"\nConverting COCO annotations to YOLO format: {coco_json_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        print(f"Loaded COCO data with {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
    except Exception as e:
        print(f"Error loading COCO annotations: {str(e)}")
        return None
    
    # Create category mapping (always use single class for simplicity)
    print("Simplifying dataset to single class (all objects mapped to class 0)")
    
    # Get image dimensions
    image_info = {}
    for img in coco_data['images']:
        image_info[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }
    
    # Group annotations by image
    annotations_by_image = {}
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(annotation)
    
    print(f"\nFound annotations for {len(annotations_by_image)} images")
    
    label_files_created = 0
    label_entries_added = 0
    
    # Convert each annotation
    for image_id, annotations in annotations_by_image.items():
        if image_id not in image_info:
            continue
        
        img_data = image_info[image_id]
        img_width = img_data['width']
        img_height = img_data['height']
        img_filename = img_data['file_name']
        img_basename = os.path.splitext(img_filename)[0]
        
        # Create YOLO format label file
        label_file = os.path.join(output_dir, img_basename + '.txt')
        
        has_valid_annotations = False
        
        with open(label_file, 'w') as f:
            for annotation in annotations:
                try:
                    # Always use class 0 for single class mode
                    category_id = 0
                    
                    # Convert bbox from [x,y,width,height] to YOLO format
                    x, y, width, height = annotation['bbox']
                    
                    if width <= 0 or height <= 0:
                        continue
                    
                    # Normalize coordinates
                    x_center = (x + width / 2) / img_width
                    y_center = (y + height / 2) / img_height
                    norm_width = width / img_width
                    norm_height = height / img_height
                    
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                            0 < norm_width <= 1 and 0 < norm_height <= 1):
                        continue
                    
                    f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                    has_valid_annotations = True
                    label_entries_added += 1
                    
                except Exception as e:
                    print(f"Error processing annotation for image {img_filename}: {str(e)}")
                    continue
        
        if not has_valid_annotations:
            try:
                os.remove(label_file)
            except:
                pass
        else:
            label_files_created += 1
    
    print(f"Created {label_files_created} label files with {label_entries_added} bounding boxes")
    return label_files_created > 0

def test_yolov5(model_size='s'):
    """Test YOLOv5 model with specified size variant"""
    assert model_size in ['s', 'm'], "Only 's' and 'm' variants are supported"
    
    model_name = f"yolov5{model_size}"
    yolo_output = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(yolo_output, exist_ok=True)
    
    # Setup YOLOv5 environment
    yolov5_dir = setup_yolov5_environment()
    if not yolov5_dir:
        return {"mAP50": 0, "mAP50-95": 0, "Error": "Failed to setup YOLOv5 environment"}
    
    # Download models
    models = download_yolov5_models()
    if model_name not in models:
        return {"mAP50": 0, "mAP50-95": 0, "Error": f"Failed to download {model_name} model"}
    
    # Create dataset structure
    dataset_root = os.path.join(yolo_output, "dataset")
    os.makedirs(dataset_root, exist_ok=True)
    
    # Create train and val directories
    for split in ['train', 'val']:
        split_images = os.path.join(dataset_root, split, 'images')
        split_labels = os.path.join(dataset_root, split, 'labels')
        os.makedirs(split_images, exist_ok=True)
        os.makedirs(split_labels, exist_ok=True)
    
    # Process annotations for train and val splits
    print("\nProcessing annotations...")
    for split in ['train', 'val']:
        annot_file = os.path.join(OUTPUT_DIR, f"{split}_annotations.json")
        
        if not os.path.exists(annot_file):
            print(f"Creating {split} annotations file...")
            # Create train/val split if not exists
            from sklearn.model_selection import train_test_split
            
            image_ids = [img['id'] for img in coco_data['images']]
            if len(image_ids) == 0:
                print("No images found in dataset")
                continue
            
            train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
            
            if split == 'train':
                split_ids = train_ids
            else:
                split_ids = val_ids
            
            split_data = {
                'images': [img for img in coco_data['images'] if img['id'] in split_ids],
                'annotations': [ann for ann in coco_data['annotations'] if ann['image_id'] in split_ids],
                'categories': coco_data['categories']
            }
            
            with open(annot_file, 'w') as f:
                json.dump(split_data, f)
        
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
            
        except Exception as e:
            print(f"Error processing {split} annotations: {str(e)}")
            return {"mAP50": 0, "mAP50-95": 0, "Error": f"Annotation processing error: {str(e)}"}
    
    # Create YAML configuration for YOLOv5
    yaml_content = f"""# YOLOv5 dataset configuration
path: {os.path.abspath(dataset_root)}
train: train/images
val: val/images
test: # optional

# Classes
nc: 1  # number of classes
names: ['object']  # class names
"""
    
    yaml_path = os.path.join(yolo_output, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nCreated dataset configuration at {yaml_path}")
    
    # Change to YOLOv5 directory for training
    original_cwd = os.getcwd()
    os.chdir(yolov5_dir)
    
    try:
        print("\nStarting YOLOv5 training...")
        start_time = time.time()
        
        # Import sys at the function level to ensure it's available
        import sys
        
        # YOLOv5 training command
        train_cmd = [
            sys.executable, "train.py",
            "--data", yaml_path,
            "--weights", models[model_name],
            "--epochs", "30",
            "--batch-size", "16",
            "--img", "640",
            "--device", "0" if torch.cuda.is_available() else "cpu",
            "--project", yolo_output,
            "--name", "train",
            "--exist-ok",
            "--single-cls",
            "--cache",
            "--save-period", "5",
            "--patience", "5",
            # Hyperparameters
            "--hyp", "data/hyps/hyp.scratch-low.yaml",
        ]
        
        print(f"Running command: {' '.join(train_cmd)}")
        
        # Run training
        result = subprocess.run(train_cmd, capture_output=True, text=True, cwd=yolov5_dir)
        
        training_time = (time.time() - start_time) / 60.0
        
        print(f"Training completed in {training_time:.2f} minutes")
        print("Training stdout:", result.stdout[-1000:])  # Last 1000 chars
        if result.stderr:
            print("Training stderr:", result.stderr[-1000:])
        
        # Parse results
        metrics = {}
        
        # Look for results.csv or results.txt
        results_dir = os.path.join(yolo_output, "train")
        results_file = os.path.join(results_dir, "results.csv")
        
        if os.path.exists(results_file):
            try:
                results_df = pd.read_csv(results_file)
                # Get the last row (final epoch results)
                last_row = results_df.iloc[-1]
                
                # Extract metrics (YOLOv5 CSV format)
                metrics["mAP50"] = float(last_row.get('val/mAP@0.5', 0))
                metrics["mAP50-95"] = float(last_row.get('val/mAP@0.5:0.95', 0))
                metrics["Precision"] = float(last_row.get('val/precision', 0))
                metrics["Recall"] = float(last_row.get('val/recall', 0))
                metrics["Box Loss"] = float(last_row.get('val/box_loss', 0))
                
                print(f"Extracted metrics from results.csv:")
                print(f"mAP50: {metrics['mAP50']:.4f}")
                print(f"mAP50-95: {metrics['mAP50-95']:.4f}")
                
            except Exception as e:
                print(f"Error reading results.csv: {str(e)}")
                metrics = {
                    "mAP50": 0,
                    "mAP50-95": 0,
                    "Precision": 0,
                    "Recall": 0,
                    "Box Loss": 0,
                    "Error": f"Results parsing error: {str(e)}"
                }
        else:
            print(f"Results file not found at {results_file}")
            metrics = {
                "mAP50": 0,
                "mAP50-95": 0,
                "Precision": 0,
                "Recall": 0,
                "Box Loss": 0,
                "Error": "Results file not found"
            }
        
        # Always set training time in metrics
        metrics["Training Time (min)"] = training_time
        
        # Calculate model parameters and GFLOPs
        try:
            # Load the trained model to get parameters
            best_model_path = os.path.join(results_dir, "weights", "best.pt")
            if os.path.exists(best_model_path):
                print(f"Loading model from {best_model_path} for parameter analysis")
                
                # Load model with safe_load option
                try:
                    # Use the global sys module
                    sys.path.append(yolov5_dir)  # Ensure YOLOv5 modules are in path
                    
                    # Try loading the model using YOLOv5's methods if possible
                    try:
                        from models.experimental import attempt_load
                        model = attempt_load(best_model_path, map_location='cpu')
                        print("Loaded model using YOLOv5's attempt_load")
                    except Exception as e:
                        print(f"Falling back to standard torch.load: {str(e)}")
                        model_state = torch.load(best_model_path, map_location='cpu', pickle_module=pickle)
                        model = model_state['model'].float() if 'model' in model_state else None
                        print("Loaded model from state dict")
                        
                    if model is not None:
                        # Count total parameters
                        total_params = sum(p.numel() for p in model.parameters())
                        metrics["Parameters (M)"] = total_params / 1e6
                        
                        # Count trainable parameters
                        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        metrics["Trainable Parameters (M)"] = trainable_params / 1e6
                        
                        # Calculate percentage of trainable parameters
                        percentage = (trainable_params / total_params * 100) if total_params > 0 else 0
                        print(f"\nParameter count: {total_params:,} total, {trainable_params:,} trainable ({percentage:.2f}%)")
                        
                        # Calculate GFLOPs with more robust error handling
                        try:
                            # Install thop if needed
                            try:
                                from thop import profile
                            except ImportError:
                                print("Installing thop for FLOPs calculation...")
                                subprocess.run([sys.executable, "-m", "pip", "install", "thop"], check=True)
                                from thop import profile
                            
                            # Create proper dummy input
                            model.eval()
                            img_size = 640
                            dummy_input = torch.zeros((1, 3, img_size, img_size), device='cpu')
                            
                            # Profile with error handling
                            with torch.no_grad():
                                try:
                                    macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
                                    gflops = macs * 2 / 1e9
                                    metrics["GFLOPs"] = gflops
                                    print(f"Model GFLOPs: {gflops:.2f}")
                                except Exception as e:
                                    print(f"Error in profiling: {str(e)}")
                            
                            # Alternative method using a forward pass
                            try:
                                print("Trying alternative GFLOPs calculation...")
                                start = time.time()
                                _ = model(dummy_input)
                                end = time.time()
                                inference_time = (end - start) * 1000  # ms
                                
                                # Use a simplified GFLOPs estimation based on model size
                                est_gflops = total_params * 2 / 1e9  # Rough estimate
                                metrics["GFLOPs"] = est_gflops
                                print(f"Estimated GFLOPs: ~{est_gflops:.2f} (based on param count)")
                            except Exception as inner_e:
                                print(f"Alternative method failed: {str(inner_e)}")
                                metrics["GFLOPs"] = "N/A"
                        except Exception as e:
                            print(f"GFLOPs calculation completely failed: {str(e)}")
                            metrics["GFLOPs"] = "N/A"
                    else:
                        print("Model could not be properly loaded")
                        metrics["Parameters (M)"] = "N/A"
                        metrics["Trainable Parameters (M)"] = "N/A"
                        metrics["GFLOPs"] = "N/A"
                except Exception as e:
                    print(f"Error loading model: {str(e)}")
                    metrics["Parameters (M)"] = "N/A"
                    metrics["Trainable Parameters (M)"] = "N/A"
                    metrics["GFLOPs"] = "N/A"
            else:
                print(f"Model file not found at {best_model_path}")
                metrics["Parameters (M)"] = "N/A"
                metrics["Trainable Parameters (M)"] = "N/A"
                metrics["GFLOPs"] = "N/A"
        except Exception as e:
            print(f"Error calculating parameters: {str(e)}")
            traceback.print_exc()
            metrics["Parameters (M)"] = "N/A"
            metrics["Trainable Parameters (M)"] = "N/A"
            metrics["GFLOPs"] = "N/A"
        
        # Calculate inference time
        if torch.cuda.is_available():
            try:
                # Load model for inference testing
                import torch.nn as nn
                sys.path.append(yolov5_dir)
                
                best_model_path = os.path.join(results_dir, "weights", "best.pt")
                if os.path.exists(best_model_path):
                    # Create dummy input
                    dummy_input = torch.randn(1, 3, 640, 640).cuda()
                    
                    # Load model
                    model = torch.load(best_model_path, map_location='cuda')['model'].float()
                    model.eval()
                    
                    # Warmup
                    for _ in range(10):
                        with torch.no_grad():
                            _ = model(dummy_input)
                    
                    torch.cuda.synchronize()
                    
                    # Measure inference time
                    t0 = time.time()
                    iterations = 50
                    with torch.no_grad():
                        for _ in range(iterations):
                            _ = model(dummy_input)
                    torch.cuda.synchronize()
                    
                    inference_time = (time.time() - t0) * 1000 / iterations
                    metrics["Inference Time (ms)"] = inference_time
                    metrics["GPU Memory (GB)"] = torch.cuda.max_memory_allocated() / 1e9
                else:
                    metrics["Inference Time (ms)"] = "N/A"
                    metrics["GPU Memory (GB)"] = "N/A"
            except Exception as e:
                print(f"Error measuring inference time: {str(e)}")
                metrics["Inference Time (ms)"] = "N/A"
                metrics["GPU Memory (GB)"] = "N/A"
        else:
            metrics["Inference Time (ms)"] = "N/A"
            metrics["GPU Memory (GB)"] = "N/A"
        
        # Save metrics to CSV
        csv_path = os.path.join(OUTPUT_DIR, 'yolov5.csv')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        metrics_row = {
            'Timestamp': timestamp,
            'Model': model_name,
            'mAP50': metrics['mAP50'],
            'mAP50-95': metrics['mAP50-95'],
            'Parameters (M)': metrics['Parameters (M)'],
            'Trainable Parameters (M)': metrics.get('Trainable Parameters (M)', metrics['Parameters (M)']),
            'GFLOPs': metrics['GFLOPs'],
            'Inference Time (ms)': metrics['Inference Time (ms)'],
            'Training Time (min)': metrics['Training Time (min)'],
            'GPU Memory (GB)': metrics['GPU Memory (GB)'],
            'Error': metrics.get('Error', '')
        }
        
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            df = pd.DataFrame(columns=metrics_row.keys())
        
        df = pd.concat([df, pd.DataFrame([metrics_row])], ignore_index=True)
        df.to_csv(csv_path, index=False)
        
        print(f"\nMetrics saved to {csv_path}")
        
        # Print summary
        print("\nTraining Summary:")
        print(f"Model: {model_name}")
        print(f"mAP50: {metrics['mAP50']:.4f}")
        print(f"mAP50-95: {metrics['mAP50-95']:.4f}")
        print(f"Parameters (M): {metrics['Parameters (M)']}")
        print(f"Trainable Parameters (M): {metrics.get('Trainable Parameters (M)', metrics['Parameters (M)'])}")
        
        if metrics['GFLOPs'] != "N/A":
            print(f"GFLOPs: {metrics['GFLOPs']:.2f}")
        else:
            print(f"GFLOPs: {metrics['GFLOPs']}")
            
        print(f"Training Time: {metrics['Training Time (min)']:.2f} minutes")
        print(f"Inference Time: {metrics['Inference Time (ms)']} ms")
        print(f"GPU Memory: {metrics['GPU Memory (GB)']} GB")
        
        return metrics
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        traceback.print_exc()
        return {"mAP50": 0, "mAP50-95": 0, "Error": f"Training error: {str(e)}"}
    
    finally:
        # Always change back to original directory
        os.chdir(original_cwd)

def run_yolov5_tests():
    """Run YOLOv5 model tests"""
    results = {}
    
    # Test YOLOv5s
    try:
        print("\n--- Testing YOLOv5-s ---")
        results['YOLOv5-s'] = test_yolov5('s')
    except Exception as e:
        print(f"Error testing YOLOv5-s: {e}")
        results['YOLOv5-s'] = {"mAP50": 0, "mAP50-95": 0, "Error": str(e)}
    
    # Test YOLOv5m
    try:
        print("\n--- Testing YOLOv5-m ---")
        results['YOLOv5-m'] = test_yolov5('m')
    except Exception as e:
        print(f"Error testing YOLOv5-m: {e}")
        results['YOLOv5-m'] = {"mAP50": 0, "mAP50-95": 0, "Error": str(e)}
    
    return results

def visualize_yolov5_results(results):
    """Create visualization of YOLOv5 model performance"""
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'mAP50': [results[model].get('mAP50', 0) for model in results],
        'mAP50-95': [results[model].get('mAP50-95', 0) for model in results],
        'Parameters (M)': [results[model].get('Parameters (M)', 0) for model in results],
        'Trainable Parameters (M)': [results[model].get('Trainable Parameters (M)', results[model].get('Parameters (M)', 0)) for model in results],
        'GFLOPs': [results[model].get('GFLOPs', 'N/A') for model in results],
        'Inference Time (ms)': [results[model].get('Inference Time (ms)', 0) for model in results],
        'Training Time (min)': [results[model].get('Training Time (min)', 'N/A') for model in results],
        'GPU Memory (GB)': [results[model].get('GPU Memory (GB)', 'N/A') for model in results],
        'Error': [results[model].get('Error', '') for model in results]
    })
    
    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, 'yolov5_comparison.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"YOLOv5 results saved to {csv_path}")
    
    # Create visualizations
    if metrics_df.empty or metrics_df['mAP50'].max() == 0:
        print("No valid performance metrics to visualize.")
        return metrics_df
    
    try:
        plt.figure(figsize=(15, 12))
        
        # 1. Accuracy comparison
        plt.subplot(3, 2, 1)
        bars = plt.bar(metrics_df['Model'], metrics_df['mAP50'], color='green')
        plt.title('YOLOv5 mAP50 Comparison', fontsize=14)
        plt.ylabel('mAP50')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        # 2. Model Size vs Accuracy
        plt.subplot(3, 2, 2)
        valid_params = [p for p in metrics_df['Parameters (M)'] if p != 'N/A']
        if valid_params:
            plt.scatter(metrics_df['Parameters (M)'], metrics_df['mAP50'], s=100, alpha=0.7)
            for i, model in enumerate(metrics_df['Model']):
                plt.annotate(model, (metrics_df['Parameters (M)'].iloc[i], metrics_df['mAP50'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points')
        plt.title('Model Size vs Accuracy', fontsize=14)
        plt.xlabel('Parameters (M)')
        plt.ylabel('mAP50')
        plt.grid(True, alpha=0.7)
        
        # 3. Training Time Comparison
        plt.subplot(3, 2, 3)
        bars = plt.bar(metrics_df['Model'], metrics_df['Training Time (min)'], color='orange')
        plt.title('Training Time Comparison', fontsize=14)
        plt.ylabel('Training Time (minutes)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom')
        
        # 4. mAP50 vs mAP50-95
        plt.subplot(3, 2, 4)
        plt.scatter(metrics_df['mAP50'], metrics_df['mAP50-95'], s=100, alpha=0.7)
        for i, model in enumerate(metrics_df['Model']):
            plt.annotate(model, (metrics_df['mAP50'].iloc[i], metrics_df['mAP50-95'].iloc[i]),
                       xytext=(5, 5), textcoords='offset points')
        plt.title('mAP50 vs mAP50-95', fontsize=14)
        plt.xlabel('mAP50')
        plt.ylabel('mAP50-95')
        plt.grid(True, alpha=0.7)
        
        # 5. Total vs Trainable Parameters
        plt.subplot(3, 2, 5)
        x = metrics_df['Parameters (M)']
        y = metrics_df['Trainable Parameters (M)']
        
        # Create bar chart comparing total vs trainable parameters
        models = metrics_df['Model']
        x_pos = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots()
        ax.bar(x_pos - width/2, x, width, label='Total Parameters')
        ax.bar(x_pos + width/2, y, width, label='Trainable Parameters')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Parameters (M)')
        ax.set_title('Total vs Trainable Parameters')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models)
        ax.legend()
        
        # Save this plot separately
        plt.savefig(os.path.join(OUTPUT_DIR, 'yolov5_parameters_comparison.png'))
        plt.close()
        
        # 6. GFLOPs Comparison
        plt.subplot(3, 2, 6)
        if 'GFLOPs' in metrics_df.columns:
            # Filter numeric values for plotting
            gflop_data = metrics_df.copy()
            gflop_data = gflop_data[gflop_data['GFLOPs'] != 'N/A']
            
            if not gflop_data.empty:
                # Convert to numeric if needed
                if isinstance(gflop_data['GFLOPs'].iloc[0], str):
                    gflop_data['GFLOPs'] = pd.to_numeric(gflop_data['GFLOPs'], errors='coerce')
                
                bars = plt.bar(gflop_data['Model'], gflop_data['GFLOPs'], color='purple')
                plt.title('GFLOPs Comparison', fontsize=14)
                plt.ylabel('GFLOPs')
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}',
                            ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'yolov5_comparison_charts.png'))
        plt.close()
        
        print("YOLOv5 comparison charts saved")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        traceback.print_exc()
    
    return metrics_df

def main():
    """Main function to run YOLOv5 thermal vision model tests"""
    global DATASET_PREPARED, MODELS_DOWNLOADED
    
    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Prepare dataset
    if not DATASET_PREPARED:
        print("\n--- Preparing dataset ---")
        filtered_coco_data, filtered_annot_file = prepare_available_images_dataset()
        DATASET_PREPARED = True
    
    # Run YOLOv5 tests
    print("\n--- Running YOLOv5 Tests ---")
    results = run_yolov5_tests()
    
    # Visualize results
    if results:
        try:
            visualize_yolov5_results(results)
        except Exception as e:
            print(f"Error visualizing results: {str(e)}")
    
    # Print summary
    print("\n" + "="*50)
    print("YOLOv5 TESTING COMPLETE")
    print("="*50)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  mAP50: {metrics.get('mAP50', 'N/A')}")
        print(f"  mAP50-95: {metrics.get('mAP50-95', 'N/A')}")
        print(f"  Parameters (M): {metrics.get('Parameters (M)', 'N/A')}")
        print(f"  Trainable Parameters (M): {metrics.get('Trainable Parameters (M)', metrics.get('Parameters (M)', 'N/A'))}")
        print(f"  GFLOPs: {metrics.get('GFLOPs', 'N/A')}")
        print(f"  Training Time (min): {metrics.get('Training Time (min)', 'N/A')}")
        print(f"  Inference Time (ms): {metrics.get('Inference Time (ms)', 'N/A')}")
        if metrics.get('Error'):
            print(f"  Error: {metrics['Error']}")
    
    return results

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user")
    except Exception as e:
        print(f"Error in main script: {str(e)}")
        traceback.print_exc()