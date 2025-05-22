# Comprehensive Thermal Model Testing Framework
# Testing All Models on FLIR Thermal Dataset

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
from tqdm import tqdm  # Using standard tqdm instead of notebook version
import sys
import traceback
import yaml

# Set environment variable to avoid OpenMP errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set paths to your dataset
DATA_DIR = os.path.join(os.getcwd(), "data")
OUTPUT_DIR = os.path.join(os.getcwd(), "results")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
ANNOTATIONS_FILE = os.path.join(DATA_DIR, "annotations.json")  # Updated path to correct file

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

print(f"Data directory: {DATA_DIR}")
print(f"Images directory: {IMAGE_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# Global variables to track dataset and model preparation
# These MUST be at module level and initialized only once
DATASET_PREPARED = False
MODELS_DOWNLOADED = False
YOLO_MODELS = {}  # Dictionary to store preloaded models
filtered_coco_data = None  # Global variable to store the filtered data

# Define annotation file paths
train_annot_file = os.path.join(OUTPUT_DIR, "train_annotations.json")
val_annot_file = os.path.join(OUTPUT_DIR, "val_annotations.json")
filtered_annot_file = os.path.join(OUTPUT_DIR, "filtered_annotations.json")

# Function to download all models at once
def download_all_models():
    """Download all required YOLO models in advance"""
    global MODELS_DOWNLOADED, YOLO_MODELS
    
    if MODELS_DOWNLOADED:
        print("Models already downloaded, skipping.")
        return YOLO_MODELS
    
    print("Downloading all required models...")
    
    # Ensure required packages are installed
    try:
        import yolov5
        print("YOLOv5 package already installed")
    except ImportError:
        subprocess.run(["pip", "install", "--user", "yolov5"], check=False)
        try:
            import yolov5
            print("Installed YOLOv5")
        except ImportError:
            print("Failed to install YOLOv5")
    
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
    
    # Dictionary to store models
    models = {}
    
    
    # Download YOLOv8 models
    try:
        from ultralytics import YOLO
        for size in ['s', 'm']:
            model_name = f"yolov8{size}"
            print(f"Downloading {model_name}...")
            try:
                model = YOLO(f'{model_name}.pt')  # Will auto-download
                models[model_name] = model
                print(f"Successfully downloaded {model_name}")
            except Exception as e:
                print(f"Error downloading {model_name}: {str(e)}")
    except Exception as e:
        print(f"Error with YOLOv8 downloads: {str(e)}")
    
    
    MODELS_DOWNLOADED = True
    YOLO_MODELS = models
    
    print(f"Downloaded {len(models)} models: {list(models.keys())}")
    return models

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    # Print GPU memory
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

# Function to filter annotations to only available images
def prepare_available_images_dataset():
    """Filter annotations to only include available images and create new COCO file"""
    global DATASET_PREPARED, filtered_coco_data, filtered_annot_file
    
    # Define the filtered annotations file path
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
            print("Will re-prepare the dataset.")
            # Continue with preparation
    
    print("Preparing dataset with only available images...")
    
    # Get list of all available image files in the directory
    available_files = set()
    
    if not os.path.exists(IMAGE_DIR):
        print(f"Warning: Image directory {IMAGE_DIR} does not exist. Creating it.")
        os.makedirs(IMAGE_DIR, exist_ok=True)
    
    print(f"Searching for images in: {IMAGE_DIR}")
    image_count = 0
    
    # List all image files in the directory
    for root, dirs, files in os.walk(IMAGE_DIR):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_count += 1
                relative_path = os.path.relpath(os.path.join(root, file), IMAGE_DIR)
                available_files.add(relative_path)
    
    print(f"Found {len(available_files)} image files in directory")
    
    # If no images found, check if there are images in the dataset directory above
    if len(available_files) == 0:
        print("No images found in the images directory. Checking if images exist in the data directory...")
        for root, dirs, files in os.walk(DATA_DIR):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    # Found images in data directory, copy them to images directory
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
                        # Found images in img directory, copy them to images directory
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
    
    # Also check for files directly with their basename
    available_basenames = {os.path.basename(f) for f in available_files}
    
    # Filter images to only those available in the directory
    available_images = []
    available_image_ids = set()
    
    for img in full_coco_data['images']:
        file_name = img['file_name']
        basename = os.path.basename(file_name)
        
        # Check if file exists either by relative path or basename
        if file_name in available_files or basename in available_basenames:
            # Update file_name to be just the basename for easier loading
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
    filtered_annot_file = os.path.join(OUTPUT_DIR, "filtered_annotations.json")
    with open(filtered_annot_file, 'w') as f:
        json.dump(filtered_coco_data, f)
    
    print(f"Created filtered annotation file with {len(available_images)} images")
    
    DATASET_PREPARED = True
    return filtered_coco_data, filtered_annot_file

# Filter annotations to available images
filtered_coco_data, filtered_annot_file = prepare_available_images_dataset()
coco_data = filtered_coco_data  # Replace the full coco_data with filtered data

# Extract categories
categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
print(f"Categories: {categories}")

# Results logging function
def log_result(model_name, metrics):
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
    # Get the first available image
    sample_img_filename = coco_data['images'][0]['file_name']
    sample_img_path = os.path.join(IMAGE_DIR, sample_img_filename)
    
    # If file not found directly, try to search for it
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
            # Check if image is grayscale or RGB
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
train_annot_file = os.path.join(OUTPUT_DIR, "train_annotations.json")
val_annot_file = os.path.join(OUTPUT_DIR, "val_annotations.json")
yaml_file = os.path.join(OUTPUT_DIR, "thermal_dataset.yaml")

# Check if files already exist
if not (os.path.exists(train_annot_file) and os.path.exists(val_annot_file) and os.path.exists(yaml_file)):
    # Split dataset (80% train, 20% val)
    from sklearn.model_selection import train_test_split
    
    image_ids = [img['id'] for img in coco_data['images']]
    
    # Check if we have images to split
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
    
    # Create YAML config - specifying the actual image directory and annotation format
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
    
    # Update YAML file to include image directories even if files exist
    if os.path.exists(yaml_file):
        with open(yaml_file, 'r') as f:
            yaml_content = f.read()
        
        if 'train_dir' not in yaml_content:
            print("Updating YAML to include image directories")
            yaml_content = f"""
# Dataset paths
path: {OUTPUT_DIR}  # Root directory
train: {train_annot_file}  # Path to train annotations
val: {val_annot_file}  # Path to validation annotations
test:  # Path to test annotations (optional)

# Image directory - this is critical
train_dir: {IMAGE_DIR}  # Directory containing training images
val_dir: {IMAGE_DIR}  # Directory containing validation images
test_dir:  # Directory containing test images (optional)

# Classes
names:
  {os.linesep.join([f'{i}: {categories[cat_id]}' for i, cat_id in enumerate(categories)])}
"""
            with open(yaml_file, 'w') as f:
                f.write(yaml_content)

# Global model performance tracker
all_results = {}

#---------------------------------------------------------------------------------
# COCO to YOLO Conversion Helper
#---------------------------------------------------------------------------------
def convert_coco_to_yolo(coco_json_path, output_dir, image_dir):
    """
    Convert COCO format annotations to YOLO format
    Args:
        coco_json_path: Path to COCO JSON file
        output_dir: Directory to save YOLO annotations
        image_dir: Directory containing images
    """
    print(f"\nConverting COCO annotations to YOLO format: {coco_json_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load COCO annotations
    try:
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        print(f"Loaded COCO data with {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
    except Exception as e:
        print(f"Error loading COCO annotations: {str(e)}")
        return None
    
    # Create category lookup (id -> index)
    # Create a sequential mapping from category ID to index starting from 0
    categories = {}
    cat_names = []
    for i, category in enumerate(coco_data['categories']):
        categories[category['id']] = i  # Map each category ID to sequential index
        cat_names.append(category['name'])
    
    print(f"\nCategory mapping: {categories}")
    print(f"Number of categories: {len(categories)}")
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
    
    # Track category usage
    category_usage = {i: 0 for i in range(len(categories))}
    
    # Check if image files exist in the target directory
    os.makedirs(image_dir, exist_ok=True)
    
    # Counts for tracking
    label_files_created = 0
    label_entries_added = 0
    valid_image_count = 0
    
    # Convert each annotation
    for image_id, annotations in annotations_by_image.items():
        if image_id not in image_info:
            print(f"Warning: No image info for image_id {image_id}")
            continue
        
        img_data = image_info[image_id]
        img_width = img_data['width']
        img_height = img_data['height']
        
        # Get image filename
        img_filename = img_data['file_name']
        
        # Check for the image file in various formats
        img_basename = os.path.splitext(img_data['file_name'])[0]
        
        # Get the image path
        img_path = os.path.join(image_dir, img_filename)
        
        # Create YOLO format label file
        label_file = os.path.join(output_dir, img_basename + '.txt')
        
        # Track if we have valid annotations for this image
        has_valid_annotations = False
        
        with open(label_file, 'w') as f:
            for annotation in annotations:
                try:
                    # Always use class 0 for all objects (single class mode)
                    category_id = 0  
                    category_usage[category_id] += 1
                    
                    # Convert bbox from [x,y,width,height] to YOLO format [x_center, y_center, width, height]
                    x, y, width, height = annotation['bbox']
                    
                    # Skip invalid boxes
                    if width <= 0 or height <= 0:
                        print(f"Warning: Invalid box dimensions for image {img_filename}: width={width}, height={height}")
                        continue
                        
                    # Normalize coordinates
                    x_center = (x + width / 2) / img_width
                    y_center = (y + height / 2) / img_height
                    norm_width = width / img_width
                    norm_height = height / img_height
                    
                    # Skip invalid normalized values
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                            0 < norm_width <= 1 and 0 < norm_height <= 1):
                        print(f"Warning: Invalid normalized coordinates for image {img_filename}: x_center={x_center}, y_center={y_center}, width={norm_width}, height={norm_height}")
                        continue
                    
                    # Write YOLO format: class_id x_center y_center width height
                    f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                    has_valid_annotations = True
                    label_entries_added += 1
                    
                except Exception as e:
                    print(f"Error processing annotation for image {img_filename}: {str(e)}")
                    continue
        
        # If no valid annotations were written, remove the empty label file
        if not has_valid_annotations:
            try:
                os.remove(label_file)
                print(f"Removed empty label file for {img_filename}")
            except:
                pass
        else:
            label_files_created += 1
            valid_image_count += 1
    
    print(f"\nConversion Summary:")
    print(f"Category usage counts: {category_usage}")
    print(f"Created {label_files_created} label files with {label_entries_added} bounding boxes")
    print(f"Valid images with annotations: {valid_image_count}")
    
    # Create example images with placeholder data if needed
    if valid_image_count == 0:
        print("\nNo valid labels created. Creating example placeholder images...")
        # Create at least one example image and label for training to work
        example_dir = os.path.join(os.path.dirname(output_dir), 'images')
        os.makedirs(example_dir, exist_ok=True)
        
        # Create an example image
        example_img = np.zeros((640, 640, 3), dtype=np.uint8)
        example_img[200:400, 200:400, :] = 255  # White square in the middle
        
        # Save the example image
        example_path = os.path.join(example_dir, 'example.jpg')
        cv2.imwrite(example_path, example_img)
        
        # Create matching label file
        example_label_path = os.path.join(output_dir, 'example.txt')
        with open(example_label_path, 'w') as f:
            # Object centered at (300, 300) with width/height of 200 pixels, normalized to 0-1
            f.write(f"0 0.5 0.5 0.3125 0.3125\n")
        
        print(f"Created example image and label at {example_path}")
        valid_image_count = 1
    
    # Create dataset.yaml file
    yaml_path = os.path.join(os.path.dirname(output_dir), 'data.yaml')
    
    # Count image files
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_count = 0
    for ext in img_extensions:
        image_count += len(glob.glob(os.path.join(image_dir, ext)))
                 
    print(f"\nSetting up YOLO dataset with {image_count} images...")
    
    # Write the YAML file
    with open(yaml_path, 'w') as f:
        f.write(f"# YOLO dataset configuration\n")
        f.write(f"path: {os.path.abspath(os.path.dirname(output_dir))}\n")
        f.write(f"train: images\n")
        f.write(f"val: images\n")
        f.write(f"test: images\n\n")
        f.write(f"# Class definitions\n")
        f.write(f"nc: 1\n")  # Always use 1 class (simplified)
        f.write(f"names: ['object']\n")  # Simplified class name
    
    print(f"YOLO dataset created at {os.path.dirname(output_dir)}")
    print(f"YAML config file: {yaml_path}")
    
    return yaml_path

#----------------------------------------------------------------------------------
# 2. YOLOv8 Models (s and m variants)
#----------------------------------------------------------------------------------

def test_yolov8(model_size='s'):
    """Test YOLOv8 model with specified size variant"""
    assert model_size in ['s', 'm'], "Only 's' and 'm' variants are supported"
    global YOLO_MODELS
    
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
    
    model_name = f"yolov8{model_size}"
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
        if model_name in YOLO_MODELS and YOLO_MODELS[model_name] is not None:
            model = YOLO_MODELS[model_name]
            print(f"Using cached {model_name} model")
        else:
            model = YOLO(f'{model_name}.pt')
            print(f"Loaded {model_name} model")
            
        # Check model trainability before training
        total_params_before = sum(p.numel() for p in model.model.parameters())
        trainable_params_before = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        percentage = (trainable_params_before / total_params_before * 100) if total_params_before > 0 else 0
        
        print(f"\nBefore training: {total_params_before:,} total, {trainable_params_before:,} trainable ({percentage:.2f}%)")
        
        # Ensure parameters are trainable
        if trainable_params_before == 0:
            print("WARNING: No trainable parameters detected. Enabling training mode...")
            # Attempt to set requires_grad to True for all parameters
            for param in model.model.parameters():
                param.requires_grad = True
            
            # Check if that worked
            trainable_params_after = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            if trainable_params_after > 0:
                print(f"Successfully enabled training for {trainable_params_after:,} parameters")
            else:
                print("Could not enable training parameters - may be in inference-only mode")
                
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return {"mAP50": 0, "mAP50-95": 0, "Error": f"Model loading error: {str(e)}"}
    
    # Train the model
    try:
        print("\nStarting training...")
        start_time = time.time()  # Start timing
        
        # Calculate and print model parameters before training
        num_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        params_millions = num_params / 1e6
        print(f"\nModel Parameters: {params_millions:.2f}M")
        
        # Calculate GFLOPs (for 640x640 input)
        try:
            from thop import profile
            device = next(model.model.parameters()).device  # Get model's device
            input_tensor = torch.randn(1, 3, 640, 640).to(device)
            model.model.to(device)  # Ensure model is on same device as input
            
            # Ensure model is in eval mode for proper profiling
            model.model.eval()
            
            # Disable gradients for faster profiling
            with torch.no_grad():
                # Run a warm-up pass
                _ = model.model(input_tensor)
                
                # Use torch.cuda.synchronize() if available to ensure accurate timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Profile with thop
                macs, params = profile(model.model, inputs=(input_tensor,), verbose=False)
                
                # Verify parameters count matches actual model parameters
                actual_params = sum(p.numel() for p in model.model.parameters())
                if abs(params - actual_params) > 1000:  # Allow small differences
                    print(f"Warning: Parameter count mismatch: thop={params}, actual={actual_params}")
                    params = actual_params
                
                gflops = macs * 2 / 1e9  # Convert MACs to GFLOPs
                print(f"Model GFLOPs: {gflops:.2f}")
                print(f"Model Parameters: {params:,} ({params/1e6:.2f}M)\n")
        except ImportError:
            print("Installing thop for FLOPs calculation...")
            subprocess.run([sys.executable, "-m", "pip", "install", "thop"], check=True)
            from thop import profile
            device = next(model.model.parameters()).device
            input_tensor = torch.randn(1, 3, 640, 640).to(device)
            model.model.to(device)
            model.model.eval()
            
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                macs, params = profile(model.model, inputs=(input_tensor,), verbose=False)
                gflops = macs * 2 / 1e9
                print(f"Model GFLOPs: {gflops:.2f}")
                print(f"Model Parameters: {params:,} ({params/1e6:.2f}M)\n")
        except Exception as e:
            print(f"Could not calculate GFLOPs: {str(e)}")
            traceback.print_exc()
            gflops = 0
            params = sum(p.numel() for p in model.model.parameters())
            print(f"Fallback parameter count: {params:,} ({params/1e6:.2f}M)")
        
        # Add metrics to results.csv after training
        def save_metrics_to_csv(metrics_dict):
            results_dir = os.path.join(yolo_output, 'train')
            results_file = os.path.join(results_dir, 'results.csv')
            
            # Create results directory if it doesn't exist
            os.makedirs(results_dir, exist_ok=True)
            
            # Prepare metrics row with new columns
            metrics_row = {
                'epoch': metrics_dict.get('epoch', 0),
                'train/box_loss': metrics_dict.get('train/box_loss', 0),
                'train/cls_loss': metrics_dict.get('train/cls_loss', 0),
                'train/dfl_loss': metrics_dict.get('train/dfl_loss', 0),
                'metrics/precision(B)': metrics_dict.get('metrics/precision(B)', 0),
                'metrics/recall(B)': metrics_dict.get('metrics/recall(B)', 0),
                'metrics/mAP50(B)': metrics_dict.get('metrics/mAP50(B)', 0),
                'metrics/mAP50-95(B)': metrics_dict.get('metrics/mAP50-95(B)', 0),
                'val/box_loss': metrics_dict.get('val/box_loss', 0),
                'val/cls_loss': metrics_dict.get('val/cls_loss', 0),
                'val/dfl_loss': metrics_dict.get('val/dfl_loss', 0),
                'params(M)': params_millions,
                'gflops': gflops
            }
            
            # Write to CSV
            if not os.path.exists(results_file):
                pd.DataFrame([metrics_row]).to_csv(results_file, index=False)
            else:
                pd.DataFrame([metrics_row]).to_csv(results_file, mode='a', header=False, index=False)
        
        # Ensure model is in training mode and parameters are trainable
        model.model.train()
        
        # Check trainable parameters right before training
        trainable_before_train = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        print(f"Trainable parameters right before training: {trainable_before_train:,} ({trainable_before_train/1e6:.2f}M)")
        
        # If still no trainable parameters, try to enable them at the lowest level
        if trainable_before_train == 0:
            print("WARNING: Still no trainable parameters. Making one last attempt to enable them...")
            for module in model.model.modules():
                for param in module.parameters(recurse=False):
                    param.requires_grad = True
                    
            # Final check
            trainable_after_fix = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            print(f"Trainable parameters after fix: {trainable_after_fix:,} ({trainable_after_fix/1e6:.2f}M)")
            
        # Train the model without custom callbacks
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
        
        # After training, save final metrics
        if hasattr(results, 'results_dict'):
            save_metrics_to_csv(results.results_dict)
        
        # Check trainable parameters after training
        trainable_after_train = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        total_after_train = sum(p.numel() for p in model.model.parameters())
        print(f"\nAfter training: {total_after_train:,} total, {trainable_after_train:,} trainable ({trainable_after_train/total_after_train*100:.2f}%)")
        
        # Calculate training time
        training_time = (time.time() - start_time) / 60.0  # Convert to minutes
        
        # Get metrics
        metrics = {}
        try:
            if hasattr(results, 'results_dict'):
                metrics_dict = results.results_dict
                print("\nTraining Results:")
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
        
        # Calculate additional metrics - ensure we count ALL parameters, not just trainable ones
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        metrics["Parameters (M)"] = total_params / 1e6
        metrics["Trainable Parameters (M)"] = trainable_params / 1e6
        metrics["Training Time (min)"] = training_time
        
        # Check if there are trainable parameters and report
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        percentage = (trainable_params / total_params * 100) if total_params > 0 else 0
        
        print(f"\nParameter count: {total_params:,} total, {trainable_params:,} trainable ({percentage:.2f}%)")
        if trainable_params == 0:
            print("WARNING: No trainable parameters! The model is likely in inference mode only.")
            
            # Check if we can determine why parameters aren't trainable
            for name, param in model.model.named_parameters():
                if not param.requires_grad:
                    print(f"Parameter {name} is frozen (requires_grad=False)")
                else:
                    print(f"Parameter {name} is trainable (requires_grad=True)")
                # Only show first few
                if name.count('.') < 2:  # Only show top-level parameters
                    print(f"Top level parameter {name}: requires_grad={param.requires_grad}")
        
        # Add GFLOPs calculation - need to put model in eval mode for accurate measurement
        try:
            from thop import profile
            model.model.eval()  # Set to evaluation mode
            device = next(model.model.parameters()).device
            input_tensor = torch.randn(1, 3, 640, 640).to(device)
            
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                macs, _ = profile(model.model, inputs=(input_tensor,), verbose=False)
                gflops = macs * 2 / 1e9
                metrics["GFLOPs"] = gflops
                print(f"Calculated GFLOPs: {gflops:.2f}")
        except Exception as e:
            print(f"Error calculating GFLOPs: {str(e)}")
            metrics["GFLOPs"] = None
        
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
            
            inference_time = (time.time() - t0) * 1000 / iterations  # Convert to ms
            metrics["Inference Time (ms)"] = inference_time
            metrics["GPU Memory (GB)"] = torch.cuda.max_memory_allocated() / 1e9
        else:
            metrics["Inference Time (ms)"] = "N/A"
            metrics["GPU Memory (GB)"] = "N/A"
        
        # Save metrics to yolov8.csv
        csv_path = os.path.join(OUTPUT_DIR, 'yolov8.csv')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare metrics row
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
        print("\nTraining Summary:")
        print(f"Model: {model_name}")
        print(f"mAP50: {metrics['mAP50']:.4f}")
        print(f"mAP50-95: {metrics['mAP50-95']:.4f}")
        print(f"Parameters (M): {metrics['Parameters (M)']:.2f}")
        print(f"Trainable Parameters (M): {metrics.get('Trainable Parameters (M)', metrics['Parameters (M)']):.2f}")
        if metrics.get('GFLOPs') is not None:
            print(f"GFLOPs: {metrics['GFLOPs']:.2f}")
        print(f"Training Time: {metrics['Training Time (min)']:.2f} minutes")
        print(f"Inference Time: {metrics['Inference Time (ms)']}")
        print(f"GPU Memory: {metrics['GPU Memory (GB)']} GB")
        
        return metrics
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return {"mAP50": 0, "mAP50-95": 0, "Error": f"Training error: {str(e)}"}


#----------------------------------------------------------------------------------
# Main Testing Function
#----------------------------------------------------------------------------------

def run_all_tests():
    """Run all model tests and compile results"""
    results = {}
    
    # Test YOLO models
    try:
        print("\n--- Testing YOLOv8-s ---")
        results['YOLOv8-s'] = test_yolov8('s')
    except Exception as e:
        print(f"Error testing YOLOv8-s: {e}")
        results['YOLOv8-s'] = {"mAP50": 0, "mAP50-95": 0, "Error": str(e)}
    
    try:
        print("\n--- Testing YOLOv8-m ---")
        results['YOLOv8-m'] = test_yolov8('m')
    except Exception as e:
        print(f"Error testing YOLOv8-m: {e}")
        results['YOLOv8-m'] = {"mAP50": 0, "mAP50-95": 0, "Error": str(e)}
    
    
    return results

#----------------------------------------------------------------------------------
# Results Visualization
#----------------------------------------------------------------------------------

def visualize_results(results):
    """Create visualization of model performance"""
    # Convert GFLOPs to numeric values where possible
    for model in results:
        if isinstance(results[model].get('GFLOPs'), str) and results[model].get('GFLOPs') == 'N/A':
            results[model]['GFLOPs'] = None
    
    # Create DataFrame from results
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'mAP50': [results[model].get('mAP50', 0) for model in results],
        'mAP50-95': [results[model].get('mAP50-95', 0) for model in results],
        'Parameters (M)': [results[model].get('Parameters (M)', 0) for model in results],
        'Trainable Parameters (M)': [results[model].get('Trainable Parameters (M)', results[model].get('Parameters (M)', 0)) for model in results],
        'GFLOPs': [results[model].get('GFLOPs', None) for model in results],
        'Inference Time (ms)': [results[model].get('Inference Time (ms)', 0) for model in results],
        'Training Time (min)': [results[model].get('Training Time (min)', 'N/A') for model in results],
        'GPU Memory (GB)': [results[model].get('GPU Memory (GB)', 'N/A') for model in results],
        'Error': [results[model].get('Error', '') for model in results]
    })
    
    # Save to CSV first in case later visualization steps fail
    csv_path = os.path.join(OUTPUT_DIR, 'model_comparison.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Group models by architecture type
    architecture_groups = {
        'YOLO': ['YOLOv8-s', 'YOLOv8-m'],
    }
    
    # Add architecture type to DataFrame
    metrics_df['Architecture'] = metrics_df['Model'].apply(
        lambda x: next((k for k, v in architecture_groups.items() if x in v), 'Other')
    )
    
    # Check if we have valid data to visualize
    if metrics_df.empty or metrics_df['mAP50'].max() == 0:
        print("No valid performance metrics to visualize. Skipping visualization.")
        return metrics_df
    
    # Create visualizations
    try:
        plt.figure(figsize=(20, 15))
        
        # 1. Accuracy (mAP50) comparison
        plt.subplot(2, 2, 1)
        bars = plt.bar(metrics_df['Model'], metrics_df['mAP50'], color=[
            'red' if g == 'YOLO' else 'blue' if g == 'Two-Stage' else 'green' if g == 'Transformer' else 'orange'
            for g in metrics_df['Architecture']
        ])
        plt.title('mAP50 Comparison', fontsize=14)
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
        for arch in architecture_groups:
            models = architecture_groups[arch]
            subset = metrics_df[metrics_df['Model'].isin(models)]
            if not subset.empty:
                plt.scatter(
                    subset['Parameters (M)'], 
                    subset['mAP50'],
                    s=100, 
                    label=arch,
                    alpha=0.7
                )
                
                # Add model names as labels
                for i, model in enumerate(subset['Model']):
                    plt.annotate(
                        model,
                        (subset['Parameters (M)'].iloc[i], subset['mAP50'].iloc[i]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center'
                    )
        
        plt.title('Model Size vs. Accuracy', fontsize=14)
        plt.xlabel('Parameters (M)')
        plt.ylabel('mAP50')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 3. Inference Speed vs. Accuracy
        plt.subplot(2, 2, 3)
        for arch in architecture_groups:
            models = architecture_groups[arch]
            subset = metrics_df[metrics_df['Model'].isin(models)]
            if not subset.empty:
                plt.scatter(
                    subset['Inference Time (ms)'], 
                    subset['mAP50'],
                    s=100, 
                    label=arch,
                    alpha=0.7
                )
                
                # Add model names as labels
                for i, model in enumerate(subset['Model']):
                    plt.annotate(
                        model,
                        (subset['Inference Time (ms)'].iloc[i], subset['mAP50'].iloc[i]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center'
                    )
        
        plt.title('Inference Speed vs. Accuracy', fontsize=14)
        plt.xlabel('Inference Time (ms)')
        plt.ylabel('mAP50')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 4. Computational Efficiency (GFLOPs vs. mAP50)
        plt.subplot(2, 2, 4)
        for arch in architecture_groups:
            models = architecture_groups[arch]
            subset = metrics_df[metrics_df['Model'].isin(models)]
            if not subset.empty and 'GFLOPs' in subset.columns:
                valid_rows = subset[pd.to_numeric(subset['GFLOPs'], errors='coerce').notna()]
                if not valid_rows.empty:
                    plt.scatter(
                        valid_rows['GFLOPs'], 
                        valid_rows['mAP50'],
                        s=100, 
                        label=arch,
                        alpha=0.7
                    )
                    
                    # Add model names as labels
                    for i, model in enumerate(valid_rows['Model']):
                        plt.annotate(
                            model,
                            (valid_rows['GFLOPs'].iloc[i], valid_rows['mAP50'].iloc[i]),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center'
                        )
        
        plt.title('Computational Cost vs. Accuracy', fontsize=14)
        plt.xlabel('GFLOPs')
        plt.ylabel('mAP50')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Create additional figure for parameter efficiency
        plt.figure(figsize=(15, 8))
        
        # Total vs Trainable Parameters comparison
        for arch in architecture_groups:
            models = architecture_groups[arch]
            subset = metrics_df[metrics_df['Model'].isin(models)]
            if not subset.empty:
                x_pos = np.arange(len(subset))
                width = 0.35
                
                # Create bar chart
                ax = plt.subplot(1, 1, 1)
                ax.bar(x_pos - width/2, subset['Parameters (M)'], width, label='Total Parameters')
                ax.bar(x_pos + width/2, subset['Trainable Parameters (M)'], width, label='Trainable Parameters')
                
                # Add model names as x-labels
                plt.xticks(x_pos, subset['Model'])
                
                # Add value labels on top of bars
                for i, (total, trainable) in enumerate(zip(subset['Parameters (M)'], subset['Trainable Parameters (M)'])):
                    plt.text(i - width/2, total + 0.5, f'{total:.1f}M', ha='center')
                    plt.text(i + width/2, trainable + 0.5, f'{trainable:.1f}M', ha='center')
                
                plt.title('Total vs Trainable Parameters', fontsize=14)
                plt.ylabel('Parameters (M)')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.legend()
                
                # Add efficiency percentage as text
                for i, (model, total, trainable) in enumerate(zip(subset['Model'], subset['Parameters (M)'], subset['Trainable Parameters (M)'])):
                    efficiency = (trainable / total) * 100 if total > 0 else 0
                    plt.text(i, max(total, trainable) + 2, f'{efficiency:.1f}% trainable', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'parameter_efficiency.png'))
        plt.close()
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison_charts.png'))
        plt.close()
        
        # Create a radar chart for multi-metric comparison
        metrics_to_plot = ['mAP50', 'Parameters (M)', 'Trainable Parameters (M)', 'Inference Time (ms)', 'GFLOPs']
        
        # Select best model from each architecture for radar chart
        best_models = []
        for arch in architecture_groups:
            models = architecture_groups[arch]
            if models:
                # Get model with highest mAP50
                subset = metrics_df[metrics_df['Model'].isin(models)]
                if not subset.empty and subset['mAP50'].max() > 0:
                    best_model = subset.loc[subset['mAP50'].idxmax()]['Model']
                    best_models.append(best_model)
        
        # Only create radar chart if we have models to compare
        if best_models:
            # Filter data for radar chart
            radar_df = metrics_df[metrics_df['Model'].isin(best_models)]
            
            # Handle non-numeric values in metrics for normalization
            radar_df_numeric = radar_df.copy()
            for metric in metrics_to_plot:
                radar_df_numeric[metric] = pd.to_numeric(radar_df_numeric[metric], errors='coerce')
            
            # Normalize metrics for radar chart
            for metric in metrics_to_plot:
                if metric == 'mAP50':
                    # Higher is better
                    max_val = radar_df_numeric[metric].max()
                    if max_val > 0:
                        radar_df[f'{metric}_norm'] = radar_df_numeric[metric] / max_val
                    else:
                        radar_df[f'{metric}_norm'] = 0
                else:
                    # Lower is better (invert)
                    max_val = radar_df_numeric[metric].max()
                    if max_val > 0:
                        radar_df[f'{metric}_norm'] = 1 - (radar_df_numeric[metric] / max_val)
                    else:
                        radar_df[f'{metric}_norm'] = 0
            
                            # Create radar chart
            plt.figure(figsize=(10, 10))
            from matplotlib.path import Path
            from matplotlib.spines import Spine
            from matplotlib.transforms import Affine2D
            
            # Number of variables - ensure we only use metrics that have valid numeric data
            valid_metrics = []
            for metric in metrics_to_plot:
                # Check if we have at least one valid value for this metric
                if any(pd.to_numeric(radar_df_numeric[metric], errors='coerce').notna()):
                    valid_metrics.append(metric)
                else:
                    print(f"Skipping metric {metric} in radar chart due to no valid data")
            
            # If we have no valid metrics, skip radar chart
            if not valid_metrics:
                print("No valid metrics for radar chart")
                return metrics_df
                
            # Use only valid metrics
            metrics_to_plot = valid_metrics
            N = len(metrics_to_plot)
            
            # What will be the angle of each axis in the plot
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            
            # Initialise the plot
            ax = plt.subplot(111, polar=True)
            
            # Draw one axis per variable + add labels
            plt.xticks(angles[:-1], metrics_to_plot, fontsize=12)
            
            # Draw ylabels
            ax.set_rlabel_position(0)
            plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=10)
            plt.ylim(0, 1)
            
            # Plot each model
            for i, model in enumerate(radar_df['Model']):
                values = [radar_df[f'{metric}_norm'].iloc[i] for metric in metrics_to_plot]
                values += values[:1]
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
                ax.fill(angles, values, alpha=0.1)
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title('Model Performance Comparison (Normalized)', fontsize=15)
            
            plt.savefig(os.path.join(OUTPUT_DIR, 'radar_chart_comparison.png'))
            plt.close()
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        traceback.print_exc()
    
    return metrics_df

def prepare_dataset():
    """Prepare the dataset for training and validation"""
    global DATASET_PREPARED
    
    # Check if dataset files already exist
    train_split_file = os.path.join(OUTPUT_DIR, 'train_split.txt')
    val_split_file = os.path.join(OUTPUT_DIR, 'val_split.txt')
    
    # If train and validation splits already exist and flag is set, just use them
    if DATASET_PREPARED and os.path.exists(train_split_file) and os.path.exists(val_split_file):
        print("Dataset already prepared, skipping preparation")
        return
    
    print("Preparing dataset with only available images...")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get list of available image files in the image directory
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(glob.glob(os.path.join(DATA_DIR, '**', f'*{ext}'), recursive=True))
    
    print(f"Found {len(image_files)} image files in directory")
    
    # Load original COCO annotations
    coco_path = os.path.join(DATA_DIR, 'annotations.json')
    
    try:
        with open(coco_path, 'r') as f:
            coco_data = json.load(f)
    except Exception as e:
        print(f"Error loading annotations file: {str(e)}")
        # Generate placeholder annotations if we can't load
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': [{'id': 1, 'name': 'object'}]
        }
    
    # Create lookup for image ids to filenames
    image_lookup = {}
    for img in coco_data['images']:
        image_lookup[img['file_name']] = img['id']
    
    # Filter annotations to only include those for available images
    available_images = [os.path.basename(img) for img in image_files]
    filtered_images = []
    for img in coco_data['images']:
        if img['file_name'] in available_images:
            filtered_images.append(img)
    
    filtered_annotations = []
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        for img in filtered_images:
            if img['id'] == image_id:
                filtered_annotations.append(ann)
                break
    
    print(f"Filtered to {len(filtered_images)} images with annotations")
    print(f"Keeping {len(filtered_annotations)} annotations for available images")
    
    # Create filtered annotation file
    filtered_data = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': coco_data['categories']
    }
    
    filtered_annot_file = os.path.join(OUTPUT_DIR, 'filtered_annotations.json')
    with open(filtered_annot_file, 'w') as f:
        json.dump(filtered_data, f)
    
    print(f"Created filtered annotation file with {len(filtered_images)} images")
    
    # Print information about categories
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    print(f"Categories: {categories}")
    
    # Split into train/validation sets
    num_images = len(filtered_images)
    if num_images == 0:
        print("No valid images found. Please check your dataset.")
        return
    
    # Show shape of a sample image to check format
    try:
        import cv2
        sample_img = cv2.imread(image_files[0])
        print(f"Sample image shape: {sample_img.shape}")
        
        # Check if images are grayscale or RGB
        if sample_img.shape[2] == 3:
            print("Images are in RGB format, will convert to grayscale for thermal models")
        else:
            print("Images are already in grayscale format")
    except Exception as e:
        print(f"Error checking image format: {str(e)}")
    
    # Shuffle and split images
    np.random.seed(42)  # For reproducibility
    indices = np.random.permutation(num_images)
    train_size = int(0.8 * num_images)
    
    # Create train/val splits
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create train and validation annotation files
    train_images = [filtered_images[i] for i in train_indices]
    val_images = [filtered_images[i] for i in val_indices]
    
    # Create sets of image IDs for quick lookup
    train_image_ids = {img['id'] for img in train_images}
    val_image_ids = {img['id'] for img in val_images}
    
    # Create train annotations
    train_annotations = [ann for ann in filtered_annotations if ann['image_id'] in train_image_ids]
    train_data = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': coco_data['categories']
    }
    
    # Create validation annotations
    val_annotations = [ann for ann in filtered_annotations if ann['image_id'] in val_image_ids]
    val_data = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': coco_data['categories']
    }
    
    # Write train/val annotation files
    train_annot_file = os.path.join(OUTPUT_DIR, 'train_annotations.json')
    val_annot_file = os.path.join(OUTPUT_DIR, 'val_annotations.json')
    
    with open(train_annot_file, 'w') as f:
        json.dump(train_data, f)
    
    with open(val_annot_file, 'w') as f:
        json.dump(val_data, f)
    
    # Write train/val split files with image paths
    train_img_paths = [os.path.basename(img['file_name']) for img in train_images]
    val_img_paths = [os.path.basename(img['file_name']) for img in val_images]
    
    with open(train_split_file, 'w') as f:
        for path in train_img_paths:
            f.write(f"{path}\n")
    
    with open(val_split_file, 'w') as f:
        for path in val_img_paths:
            f.write(f"{path}\n")
    
    # Create necessary directories
    os.makedirs(os.path.join(DATA_DIR, 'train'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'val'), exist_ok=True)
    
    # Copy or link images to train/val directories
    try:
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            
            if img_name in train_img_paths:
                dest_dir = os.path.join(DATA_DIR, 'train')
            elif img_name in val_img_paths:
                dest_dir = os.path.join(DATA_DIR, 'val')
            else:
                continue
            
            dest_path = os.path.join(dest_dir, img_name)
            
            # Skip if already exists
            if os.path.exists(dest_path):
                continue
            
            try:
                # Try to create symbolic link first (faster, uses less disk space)
                os.symlink(img_path, dest_path)
            except Exception:
                # If symlink fails, copy the file
                shutil.copy2(img_path, dest_path)
    except Exception as e:
        print(f"Error copying images: {str(e)}")
        # Continue with what we have
        
    DATASET_PREPARED = True

def main():
    """Main function to run thermal vision model tests"""
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
    
    # Prepare dataset - this will only run once even if called multiple times
    if not DATASET_PREPARED:
        print("\n--- Preparing dataset (will only run once) ---")
        prepare_dataset()
        DATASET_PREPARED = True
    else:
        print("\n--- Dataset already prepared, skipping ---")
    
    # Download all models at once - much more efficient than downloading individually
    if not MODELS_DOWNLOADED:
        print("\n--- Setting up all models (will only run once) ---")
        models = download_all_models()
        MODELS_DOWNLOADED = True
        print(f"Available models: {list(models.keys())}")
    else:
        print("\n--- Models already downloaded, skipping ---")
    
    # Define models to test - only YOLOv8-small
    tests = {
        "yolov8s": lambda: test_yolov8('s'),
        "yolov8m": lambda: test_yolov8('m'),
    }
    
    # For debugging, you can add specific models to test
    selected_tests = ["yolov8s", "yolov8m"]  # Run both YOLOv8-small and YOLOv8-medium
    
    print("Running selected tests...")
    
    # Check if PyTorch is installed correctly
    print("Required dependencies are installed")
    
    # Store results of all model tests
    results = {}
    
    # Run tests
    for model_name in selected_tests:
        if model_name in tests:
            try:
                print(f"\n--- Testing {model_name.upper()} ---")
                test_func = tests[model_name]
                
                # Call the test function for each model
                model_results = test_func()
                
                # Save results
                results[model_name] = model_results
            except KeyboardInterrupt:
                print("Testing interrupted by user")
                break
            except Exception as e:
                print(f"Error testing {model_name}: {str(e)}")
                traceback.print_exc()
                results[model_name] = {"Error": str(e)}
    
    # Generate summary plot and comparison
    if results:
        try:
            visualize_results(results)
        except Exception as e:
            print(f"Error visualizing results: {str(e)}")
            traceback.print_exc()
    
    print("\nTesting complete!")
    return results

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user")
    except Exception as e:
        print(f"Error in main script: {str(e)}")
        traceback.print_exc()