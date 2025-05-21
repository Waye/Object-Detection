# Comprehensive Thermal Model Testing Framework
# Testing RT-DETR (Real-Time Detection Transformer) on FLIR Thermal Dataset

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

# Set environment variable to avoid OpenMP errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set paths to your dataset
DATA_DIR = os.path.join(os.getcwd(), "data")
OUTPUT_DIR = os.path.join(os.getcwd(), "results")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
ANNOTATIONS_FILE = os.path.join(DATA_DIR, "annotations.json")  # Define annotation file path

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

print(f"Data directory: {DATA_DIR}")
print(f"Images directory: {IMAGE_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# Global variables to track dataset preparation
# These MUST be at module level and initialized only once
DATASET_PREPARED = False
MODEL_DOWNLOADED = False
RT_DETR_MODEL = None  # Global variable to store the model
filtered_coco_data = None  # Global variable to store the filtered data

# Define annotation file paths
train_annot_file = os.path.join(OUTPUT_DIR, "train_annotations.json")
val_annot_file = os.path.join(OUTPUT_DIR, "val_annotations.json")
filtered_annot_file = os.path.join(OUTPUT_DIR, "filtered_annotations.json")

# Function to download and initialize RT-DETR model
def download_rt_detr():
    """Download and initialize RT-DETR model"""
    global MODEL_DOWNLOADED, RT_DETR_MODEL
    
    if MODEL_DOWNLOADED and RT_DETR_MODEL is not None:
        print("RT-DETR model already downloaded, skipping.")
        return RT_DETR_MODEL
    
    print("Downloading and initializing RT-DETR model...")
    
    # Install required packages
    try:
        import torchvision
        print("Torchvision already installed")
    except ImportError:
        subprocess.run(["pip", "install", "--user", "torchvision"], check=False)
        try:
            import torchvision
            print("Installed torchvision")
        except ImportError:
            print("Failed to install torchvision")
            return None
    
    # Install metrics package
    try:
        import torchmetrics
        print("Torchmetrics already installed")
    except ImportError:
        subprocess.run(["pip", "install", "--user", "torchmetrics"], check=False)
        try:
            import torchmetrics
            print("Installed torchmetrics")
        except ImportError:
            print("Failed to install torchmetrics")
            return None
    
    # Install timm package for transformer models
    try:
        import timm
        print("Timm already installed")
    except ImportError:
        subprocess.run(["pip", "install", "--user", "timm"], check=False)
        try:
            import timm
            print("Installed timm")
        except ImportError:
            print("Failed to install timm")
            return None
    
    # Initialize RT-DETR model
    try:
        # We'll create a custom RT-DETR model using PyTorch components
        # Starting with DETR-like architecture but optimized for real-time performance
        from torch import nn
        from torchvision.models import resnet50
        import torch.nn.functional as F
        
        class RTDETRModel(nn.Module):
            def __init__(self, num_classes=2, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
                super().__init__()
                
                # Use ResNet50 as the backbone
                backbone = resnet50(pretrained=True)
                self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove avg pool and FC
                
                # Project the backbone features to the transformer dimension
                self.conv = nn.Conv2d(2048, hidden_dim, 1)
                
                # RT-DETR specific: Use efficient attention mechanism
                self.transformer = nn.Transformer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    num_encoder_layers=num_encoder_layers,
                    num_decoder_layers=num_decoder_layers,
                    dim_feedforward=hidden_dim*4,
                    dropout=0.1,
                    activation="relu",
                    batch_first=True
                )
                
                # Use learnable positional embeddings
                self.position_embedding = nn.Parameter(torch.zeros((100, hidden_dim)))
                
                # Object query embeddings for the decoder
                self.query_embed = nn.Embedding(100, hidden_dim)
                
                # Output heads for classification and bounding box regression
                self.class_embed = nn.Linear(hidden_dim, num_classes)
                self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
                
                # RT-DETR specific: Add a feature pyramid network for better performance
                self.fpn = FeaturePyramidNetwork(
                    in_channels_list=[256, 512, 1024, 2048],
                    out_channels=hidden_dim
                )
            
            def forward(self, x):
                # Extract backbone features
                features = self.backbone(x)
                
                # Apply FPN to create multi-scale features
                feature_maps = self.fpn([
                    x,                         # Original input
                    self.backbone[:5](x),      # After layer1
                    self.backbone[:6](x),      # After layer2
                    self.backbone[:7](x),      # After layer3
                    features                   # After layer4
                ])
                
                # Process the final feature map
                h = self.conv(features)
                
                # Flatten spatial dimensions and transpose to sequence format
                bs, c, h, w = h.shape
                h = h.flatten(2).permute(0, 2, 1)
                
                # Generate positional embeddings
                pos_embed = self.position_embedding[:h.size(1), :].unsqueeze(0).repeat(bs, 1, 1)
                
                # Transformer encoder-decoder
                query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
                tgt = torch.zeros_like(query_embed)
                
                # Run the transformer
                h = self.transformer(
                    src=h + pos_embed,
                    tgt=tgt,
                    query_pos=query_embed
                )
                
                # Predict class and bounding box
                outputs_class = self.class_embed(h)
                outputs_coord = self.bbox_embed(h).sigmoid()
                
                return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        
        # MLP definition for bounding box regression
        class MLP(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
                super().__init__()
                self.num_layers = num_layers
                h = [hidden_dim] * (num_layers - 1)
                self.layers = nn.ModuleList(
                    nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
                )
                
            def forward(self, x):
                for i, layer in enumerate(self.layers):
                    x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
                return x
        
        # Feature Pyramid Network implementation
        class FeaturePyramidNetwork(nn.Module):
            def __init__(self, in_channels_list, out_channels):
                super().__init__()
                self.inner_blocks = nn.ModuleList(
                    [nn.Conv2d(in_channels, out_channels, 1) for in_channels in in_channels_list]
                )
                self.layer_blocks = nn.ModuleList(
                    [nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list]
                )
                
            def forward(self, x):
                results = []
                last_inner = self.inner_blocks[-1](x[-1])
                results.append(self.layer_blocks[-1](last_inner))
                
                for i in range(len(x) - 2, -1, -1):
                    inner_lateral = self.inner_blocks[i](x[i])
                    inner_top_down = F.interpolate(last_inner, size=inner_lateral.shape[-2:], mode="nearest")
                    last_inner = inner_lateral + inner_top_down
                    results.insert(0, self.layer_blocks[i](last_inner))
                
                return results
        
        # Create RT-DETR instance
        model = RTDETRModel(num_classes=2)  # Background + object
        RT_DETR_MODEL = model
        MODEL_DOWNLOADED = True
        print("Successfully initialized RT-DETR model")
        return model
    except Exception as e:
        print(f"Error initializing RT-DETR: {str(e)}")
        traceback.print_exc()
        return None

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
                print("Images are in RGB format, original thermal data may need conversion")
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

# Check if files already exist
if not (os.path.exists(train_annot_file) and os.path.exists(val_annot_file)):
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
    
    print(f"Dataset prepared: {len(train_ids)} training images, {len(val_ids)} validation images")
else:
    print("Dataset split files already exist, skipping preparation")

#----------------------------------------------------------------------------------
# Dataset class for RT-DETR
#----------------------------------------------------------------------------------

# Create COCO dataset class for RT-DETR
class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transform=None):
        self.root = root
        self.transform = transform
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create image lookup
        self.images = self.coco_data['images']
        self.image_ids = [img['id'] for img in self.images]
        self.image_lookup = {img['id']: img for img in self.images}
        
        # Group annotations by image
        self.annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
        
        # Valid image ids (those with annotations)
        self.valid_ids = [img_id for img_id in self.image_ids if img_id in self.annotations]
        
        print(f"Loaded {len(self.valid_ids)} valid images with annotations")
    
    def __len__(self):
        return len(self.valid_ids)
    
    def __getitem__(self, idx):
        # Get image ID
        img_id = self.valid_ids[idx]
        
        # Get image info
        img_info = self.image_lookup[img_id]
        img_filename = os.path.basename(img_info['file_name'])
        
        # Find image path
        img_path = os.path.join(self.root, img_filename)
        
        # If file not found, try to search for it
        if not os.path.exists(img_path):
            for root, dirs, files in os.walk(self.root):
                for file in files:
                    if file == img_filename:
                        img_path = os.path.join(root, file)
                        break
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not load image {img_path}")
            # Create a dummy image to avoid crashing
            image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations for this image
        anns = self.annotations.get(img_id, [])
        
        # Extract bounding boxes and labels
        boxes = []
        labels = []
        
        for ann in anns:
            # Skip annotations with invalid boxes
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            
            # Convert COCO format [x, y, w, h] to normalized format [x1, y1, x2, y2]
            x1 = x / image.shape[1]
            y1 = y / image.shape[0]
            x2 = (x + w) / image.shape[1]
            y2 = (y + h) / image.shape[0]
            boxes.append([x1, y1, x2, y2])
            
            # Use category_id for class label - RT-DETR uses 0-indexed classes
            cat_id = ann.get('category_id', 1)
            labels.append(cat_id)
        
        # Handle empty annotations
        if len(boxes) == 0:
            # Create a dummy box to avoid errors
            boxes = [[0.0, 0.0, 0.1, 0.1]]
            labels = [0]  # Background class
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Convert numpy array to tensor
            image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        
        return image, target

#----------------------------------------------------------------------------------
# RT-DETR Implementation
#----------------------------------------------------------------------------------

def test_rt_detr():
    """Test RT-DETR model on thermal dataset"""
    rt_detr_output = os.path.join(OUTPUT_DIR, "rt_detr")
    os.makedirs(rt_detr_output, exist_ok=True)
    
    # Ensure required packages are installed
    try:
        import torchvision
        print("Torchvision already installed")
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "--user", "torchvision"], check=True)
        try:
            import torchvision
            print("Installed torchvision")
        except ImportError:
            print("Failed to install torchvision, skipping this test")
            return {"mAP50": 0, "mAP50-95": 0, "Error": "Installation failed"}
    
    # Install metrics package if needed
    try:
        import torchmetrics
        print("Torchmetrics already installed")
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "--user", "torchmetrics"], check=True)
        try:
            import torchmetrics
            print("Installed torchmetrics")
        except ImportError:
            print("Failed to install torchmetrics, skipping this test")
            return {"mAP50": 0, "mAP50-95": 0, "Error": "Installation failed"}
    
    # Collate function for the data loader
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    # Create data transforms with augmentations appropriate for RT-DETR
    from torchvision import transforms
    
    # RT-DETR specific augmentations
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop((512, 512), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    try:
        train_dataset = COCODataset(
            root=IMAGE_DIR,
            annotation_file=train_annot_file,
            transform=transform_train
        )
        
        val_dataset = COCODataset(
            root=IMAGE_DIR,
            annotation_file=val_annot_file,
            transform=transform_val
        )
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        traceback.print_exc()
        return {"mAP50": 0, "mAP50-95": 0, "Error": f"Dataset loading failed: {str(e)}"}
    
    # Check if we have valid data
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        error_msg = "No valid images with annotations found. Skipping test."
        print(error_msg)
        return {"mAP50": 0, "mAP50-95": 0, "Error": error_msg}
    
    # Create data loaders
    batch_size = 8  # RT-DETR can handle larger batch sizes than Cascade R-CNN
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_mem < 8:  # Less than 8GB VRAM
            batch_size = 4
            print(f"Reducing batch size to {batch_size} due to limited GPU memory")
    else:
        batch_size = 2
        print("No GPU found, using minimal batch size")
    
    # Set num_workers=0 to avoid multiprocessing issues on Windows
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,  # Always use batch size 1 for validation
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images")
    
    # Initialize RT-DETR model
    try:
        # Get RT-DETR model
        model = download_rt_detr()
        if model is None:
            raise ValueError("Failed to initialize RT-DETR model")
        
        # Move model to device
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        traceback.print_exc()
        return {"mAP50": 0, "mAP50-95": 0, "Error": f"Model initialization failed: {str(e)}"}
    
    # Define the matcher between predictions and ground truth
    class HungarianMatcher(torch.nn.Module):
        def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
            super().__init__()
            self.cost_class = cost_class
            self.cost_bbox = cost_bbox
            self.cost_giou = cost_giou
            
        def forward(self, outputs, targets):
            bs, num_queries = outputs["pred_logits"].shape[:2]
            
            # Compute the classification cost
            out_prob = outputs["pred_logits"].softmax(-1)
            out_bbox = outputs["pred_boxes"]
            
            indices = []
            for b in range(bs):
                if len(targets[b]["boxes"]) == 0:
                    indices.append(([], []))
                    continue
                    
                tgt_ids = targets[b]["labels"]
                tgt_bbox = targets[b]["boxes"]
                
                # Classification cost: -log prob of target class
                cost_class = -out_prob[b, :, tgt_ids]
                
                # Bbox cost: L1 loss
                cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)
                
                # GIoU loss
                from torchvision.ops.boxes import box_iou
                iou, _ = box_iou(out_bbox[b], tgt_bbox)
                cost_giou = -iou
                
                # Final cost matrix
                C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
                
                # Hungarian matching
                from scipy.optimize import linear_sum_assignment
                C_cpu = C.detach().cpu().numpy()
                indices_b = linear_sum_assignment(C_cpu)
                indices.append(indices_b)
                
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    
    # Define the RT-DETR loss
    class RTDETRLoss(torch.nn.Module):
        def __init__(self, matcher):
            super().__init__()
            self.matcher = matcher
            self.weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
            
            # Define losses
            self.ce_loss = torch.nn.CrossEntropyLoss()
            self.l1_loss = torch.nn.L1Loss()
            
        def loss_labels(self, outputs, targets, indices):
            pred_logits = outputs['pred_logits']
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            target_classes = torch.full(pred_logits.shape[:2], 0, dtype=torch.int64, device=pred_logits.device)
            target_classes[idx] = target_classes_o
            
            loss_ce = self.ce_loss(pred_logits.transpose(1, 2), target_classes)
            return loss_ce
            
        def loss_boxes(self, outputs, targets, indices):
            idx = self._get_src_permutation_idx(indices)
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            
            # L1 loss
            loss_bbox = self.l1_loss(src_boxes, target_boxes)
            
            # GIoU loss
            from torchvision.ops.boxes import generalized_box_iou
            giou = generalized_box_iou(
                # Convert to [x1, y1, x2, y2] format for GIoU calculation
                src_boxes,
                target_boxes
            )
            loss_giou = 1 - torch.diag(giou)
            return loss_bbox, loss_giou.mean()
            
        def _get_src_permutation_idx(self, indices):
            # Permute predictions following indices
            batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
            src_idx = torch.cat([src for (src, _) in indices])
            return batch_idx, src_idx
            
        def forward(self, outputs, targets):
            # Match predictions to targets
            indices = self.matcher(outputs, targets)
            
            # Compute losses
            loss_ce = self.loss_labels(outputs, targets, indices)
            loss_bbox, loss_giou = self.loss_boxes(outputs, targets, indices)
            
            losses = {
                'loss_ce': loss_ce,
                'loss_bbox': loss_bbox,
                'loss_giou': loss_giou
            }
            
            # Weight and sum losses
            return sum(losses[k] * self.weight_dict[k] for k in losses.keys())
    
    # Initialize matcher and loss
    matcher = HungarianMatcher()
    criterion = RTDETRLoss(matcher)
    
    # Training settings optimized for RT-DETR
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.0001,  # Lower learning rate for transformer-based models
        weight_decay=1e-4
    )
    
    # Learning rate scheduler with warmup for transformer models
    def get_lr_scheduler(optimizer, warmup_epochs=5, max_epochs=50):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch) / float(max(1, warmup_epochs))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs))))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    import math
    lr_scheduler = get_lr_scheduler(optimizer, warmup_epochs=5, max_epochs=20)
    
    # Start timing for training
    print(f"Starting RT-DETR training...")
    start_time = time.time()
    
    # Number of training epochs
    num_epochs = 20  # RT-DETR may need more epochs for convergence
    
    # Training loop
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        
        # Track metrics
        epoch_loss = 0
        iter_count = 0
        
        # Progress bar
        print(f"Epoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(train_loader, desc=f"Training", leave=False)
        
        for images, targets in progress_bar:
            try:
                # Move images and targets to device
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(torch.stack(images))
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                
                # Gradient clipping (important for transformer models)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                
                optimizer.step()
                
                # Track losses
                epoch_loss += loss.item()
                iter_count += 1
                
                # Update progress bar
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            except Exception as e:
                print(f"Error in training iteration: {str(e)}")
                traceback.print_exc()
                continue
        
        # Update learning rate
        lr_scheduler.step()
        
        # Print epoch stats
        if iter_count > 0:
            avg_loss = epoch_loss / iter_count
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, No valid iterations")
        
        # Validation loop every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            model.eval()
            val_loss = 0
            val_iter = 0
            
            print("Running validation...")
            with torch.no_grad():
                for images, targets in tqdm(val_loader, desc="Validating"):
                    try:
                        # Move images and targets to device
                        images = [image.to(device) for image in images]
                        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                        
                        # Forward pass
                        outputs = model(torch.stack(images))
                        loss = criterion(outputs, targets)
                        
                        val_loss += loss.item()
                        val_iter += 1
                    except Exception as e:
                        print(f"Error in validation: {str(e)}")
                        continue
            
            if val_iter > 0:
                avg_val_loss = val_loss / val_iter
                print(f"Validation Loss: {avg_val_loss:.6f}")
    
    # Calculate training time
    training_time = (time.time() - start_time) / 60.0  # minutes
    print(f"Training completed in {training_time:.2f} minutes")
    
    # Evaluation
    print("Evaluating model...")
    model.eval()
    
    # Convert RT-DETR outputs to detection format for evaluation
    def rtdetr_to_detection_format(outputs, threshold=0.5):
        # Get the predictions
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        # Convert to probabilities and get the highest scoring class
        prob = torch.nn.functional.softmax(pred_logits, dim=-1)
        scores, labels = prob.max(-1)
        
        # Filter out background predictions (class 0) and low confidence
        keep = (labels != 0) & (scores > threshold)
        
        # Convert to list of dictionaries
        result = {
            'boxes': pred_boxes[0][keep],
            'labels': labels[0][keep],
            'scores': scores[0][keep]
        }
        
        return result
    
    # Initialize metrics
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    
    # Evaluate on validation dataset
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            try:
                # Move images to device
                images = [image.to(device) for image in images]
                
                # Get predictions
                outputs = model(torch.stack(images))
                predictions = [rtdetr_to_detection_format(outputs)]
                
                # Prepare targets for evaluation
                target_list = []
                for t in targets:
                    target_dict = {
                        "boxes": t["boxes"].cpu(),
                        "labels": t["labels"].cpu()
                    }
                    target_list.append(target_dict)
                
                # Update metric
                metric.update(predictions, target_list)
            except Exception as e:
                print(f"Error in evaluation: {str(e)}")
                traceback.print_exc()
                continue
    
    # Compute metrics
    metrics_dict = metric.compute()
    
    # Extract relevant metrics
    metrics = {}
    metrics["mAP50"] = metrics_dict["map_50"].item() if "map_50" in metrics_dict else 0
    metrics["mAP50-95"] = metrics_dict["map"].item() if "map" in metrics_dict else 0
    
    # Model parameters
    metrics["Parameters (M)"] = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    metrics["Training Time (min)"] = training_time
    
    # For GFLOPs, we would need a separate profiling tool
    metrics["GFLOPs"] = "N/A"
    
    # Estimate inference time
    if torch.cuda.is_available():
        # Create sample image for timing
        dummy_input = torch.rand(1, 3, 512, 512).to(device)
        
        # Warm up
        for _ in range(5):
            _ = model(dummy_input)
        
        # Measure
        torch.cuda.synchronize()
        t0 = time.time()
        iterations = 20
        for _ in range(iterations):
            _ = model(dummy_input)
        torch.cuda.synchronize()
        
        inference_time = (time.time() - t0) * 1000 / iterations  # ms
        metrics["Inference Time (ms)"] = inference_time
        metrics["GPU Memory (GB)"] = torch.cuda.max_memory_allocated() / 1e9
    else:
        # Use CPU timing
        dummy_input = torch.rand(1, 3, 512, 512)
        
        # Warm up
        for _ in range(2):
            _ = model(dummy_input)
        
        # Measure
        t0 = time.time()
        iterations = 5  # Fewer iterations for CPU
        for _ in range(iterations):
            _ = model(dummy_input)
        
        inference_time = (time.time() - t0) * 1000 / iterations  # ms
        metrics["Inference Time (ms)"] = inference_time
        metrics["GPU Memory (GB)"] = "N/A"
    
    # Save the trained model
    model_save_path = os.path.join(rt_detr_output, "rt_detr_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Log results
    log_result("RT_DETR", metrics)
    
    return metrics

# Visualize results
def visualize_results(results):
    """Create visualization of model performance"""
    # Create DataFrame from results
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'mAP50': [results[model].get('mAP50', 0) for model in results],
        'mAP50-95': [results[model].get('mAP50-95', 0) for model in results],
        'Parameters (M)': [results[model].get('Parameters (M)', 0) for model in results],
        'Inference Time (ms)': [results[model].get('Inference Time (ms)', 0) for model in results],
        'Training Time (min)': [results[model].get('Training Time (min)', 'N/A') for model in results],
        'GPU Memory (GB)': [results[model].get('GPU Memory (GB)', 'N/A') for model in results],
        'Error': [results[model].get('Error', '') for model in results]
    })
    
    # Save to CSV first in case later visualization steps fail
    csv_path = os.path.join(OUTPUT_DIR, 'rt_detr_results.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Check if we have valid data to visualize
    if metrics_df.empty or metrics_df['mAP50'].max() == 0:
        print("No valid performance metrics to visualize. Skipping visualization.")
        return metrics_df
    
    # Create visualizations
    try:
        plt.figure(figsize=(12, 10))
        
        # 1. Accuracy (mAP50) bar chart
        plt.subplot(2, 2, 1)
        bars = plt.bar(metrics_df['Model'], metrics_df['mAP50'], color='blue')
        plt.title('mAP50 Performance', fontsize=14)
        plt.ylabel('mAP50')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', rotation=0)
        
        # 2. Inference speed
        plt.subplot(2, 2, 2)
        bars = plt.bar(metrics_df['Model'], metrics_df['Inference Time (ms)'], color='green')
        plt.title('Inference Time', fontsize=14)
        plt.ylabel('Time (ms)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', rotation=0)
        
        # 3. Training time
        plt.subplot(2, 2, 3)
        
        # Convert training time to numeric, handling 'N/A'
        metrics_df['Training Time (Numeric)'] = pd.to_numeric(metrics_df['Training Time (min)'], errors='coerce')
        
        # Plot only valid numeric values
        valid_rows = metrics_df.dropna(subset=['Training Time (Numeric)'])
        if not valid_rows.empty:
            bars = plt.bar(valid_rows['Model'], valid_rows['Training Time (Numeric)'], color='orange')
            plt.title('Training Time', fontsize=14)
            plt.ylabel('Time (minutes)')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', rotation=0)
        
        # 4. Model parameters
        plt.subplot(2, 2, 4)
        bars = plt.bar(metrics_df['Model'], metrics_df['Parameters (M)'], color='purple')
        plt.title('Model Size', fontsize=14)
        plt.ylabel('Parameters (M)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'rt_detr_performance.png'))
        plt.close()
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        traceback.print_exc()
    
    return metrics_df

def main():
    """Main function to run RT-DETR thermal model testing"""
    global DATASET_PREPARED, MODEL_DOWNLOADED
    
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
        prepare_available_images_dataset()
        DATASET_PREPARED = True
    else:
        print("\n--- Dataset already prepared, skipping ---")
    
    # Download model
    if not MODEL_DOWNLOADED:
        print("\n--- Setting up RT-DETR model (will only run once) ---")
        download_rt_detr()
    else:
        print("\n--- Model already initialized, skipping ---")
    
    # Store results
    results = {}
    
    # Run RT-DETR test
    try:
        print("\n--- Testing RT-DETR ---")
        model_results = test_rt_detr()
        results["RT-DETR"] = model_results
    except KeyboardInterrupt:
        print("Testing interrupted by user")
    except Exception as e:
        print(f"Error testing RT-DETR: {str(e)}")
        traceback.print_exc()
        results["RT-DETR"] = {"Error": str(e)}
    
    # Generate visualization
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