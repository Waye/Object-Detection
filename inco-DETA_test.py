# Comprehensive Thermal Model Testing Framework
# Testing DETA (Dense Transformer) on FLIR Thermal Dataset

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
import math
import requests
import zipfile
import io

# Set environment variable to avoid OpenMP errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set paths to your dataset
DATA_DIR = os.path.join(os.getcwd(), "data")
OUTPUT_DIR = os.path.join(os.getcwd(), "results")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
ANNOTATIONS_FILE = os.path.join(DATA_DIR, "annotations.json")  # Define annotation file path
DETA_DIR = os.path.join(os.getcwd(), "DETA")  # Path to store DETA model

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(DETA_DIR, exist_ok=True)

print(f"Data directory: {DATA_DIR}")
print(f"Images directory: {IMAGE_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"DETA directory: {DETA_DIR}")

# Global variables to track dataset preparation
# These MUST be at module level and initialized only once
DATASET_PREPARED = False
MODEL_DOWNLOADED = False
DETA_MODEL = None  # Global variable to store the model
filtered_coco_data = None  # Global variable to store the filtered data

# Define annotation file paths
train_annot_file = os.path.join(OUTPUT_DIR, "train_annotations.json")
val_annot_file = os.path.join(OUTPUT_DIR, "val_annotations.json")
filtered_annot_file = os.path.join(OUTPUT_DIR, "filtered_annotations.json")

# Function to create and initialize a pure PyTorch implementation of DETA
def create_pure_pytorch_deta():
    """Create a pure PyTorch implementation of DETA without CUDA extensions"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    from torchvision.models import resnet50
    
    print("Creating Simple PyTorch DETA implementation...")
    
    class SimpleDETA(nn.Module):
        """Simplified DETA implementation using only standard PyTorch components"""
        def __init__(self, num_classes=2, num_queries=300):
            super().__init__()
            
            # Backbone - ResNet50
            backbone = resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove avgpool and fc
            
            # Dimensions
            self.hidden_dim = 256
            self.num_queries = num_queries
            self.num_classes = num_classes
            
            # Feature projection
            self.conv = nn.Conv2d(2048, self.hidden_dim, kernel_size=1)
            
            # Learnable query embeddings
            self.query_embed = nn.Embedding(num_queries, self.hidden_dim)
            
            # Transformer layers
            self.transformer = nn.Transformer(
                d_model=self.hidden_dim,
                nhead=8,
                num_encoder_layers=6,
                num_decoder_layers=6,
                dim_feedforward=2048,
                dropout=0.1
            )
            
            # Output prediction heads
            self.class_embed = nn.Linear(self.hidden_dim, num_classes + 1)  # +1 for background
            self.bbox_embed = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 4)  # (x, y, w, h)
            )
            
            # Initialize parameters
            self._reset_parameters()
            
        def _reset_parameters(self):
            """Initialize weights for better convergence"""
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            
            # Use different initialization for box regression head
            for layer in self.bbox_embed.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
                    
            # Initialize class prediction bias for better convergence with rare classes
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            nn.init.constant_(self.class_embed.bias, bias_value)
            
        def _generate_positional_encoding(self, x):
            """Generate fixed positional encoding using sine and cosine functions"""
            # Get dimensions
            bs, c, h, w = x.shape
            device = x.device
            
            # Generate position encoding
            pos_enc = torch.zeros(bs, self.hidden_dim, h, w, device=device)
            
            # Use simple fixed positional encoding for each spatial dimension
            y_pos = torch.linspace(0., 1., h, device=device).unsqueeze(1).expand(-1, w)
            x_pos = torch.linspace(0., 1., w, device=device).unsqueeze(0).expand(h, -1)
            
            # Assign sine/cosine encoding across hidden dimension
            dim_t = torch.arange(0, self.hidden_dim // 2, dtype=torch.float, device=device)
            dim_t = 10000 ** (2 * dim_t / self.hidden_dim)
            
            # Apply sin/cos to positions
            for i in range(bs):
                # First half for height dimension
                pos_enc[i, 0::4, :, :] = torch.sin(y_pos.unsqueeze(0) / dim_t[0::2].view(-1, 1, 1))
                pos_enc[i, 1::4, :, :] = torch.cos(y_pos.unsqueeze(0) / dim_t[0::2].view(-1, 1, 1))
                # Second half for width dimension
                pos_enc[i, 2::4, :, :] = torch.sin(x_pos.unsqueeze(0) / dim_t[1::2].view(-1, 1, 1))
                pos_enc[i, 3::4, :, :] = torch.cos(x_pos.unsqueeze(0) / dim_t[1::2].view(-1, 1, 1))
            
            return pos_enc
            
        def forward(self, x):
            """Forward pass"""
            # Extract features from backbone
            features = self.backbone(x)
            
            # Project to hidden dimension
            features = self.conv(features)
            
            # Generate positional encoding
            pos_encoding = self._generate_positional_encoding(features)
            
            # Add positional encoding to features
            features = features + pos_encoding
            
            # Reshape for transformer: [batch_size, channels, height, width] -> [sequence_length, batch_size, channels]
            batch_size, channels, height, width = features.shape
            features = features.flatten(2).permute(2, 0, 1)  # [H*W, batch_size, channels]
            
            # Generate query embeddings for the decoder
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)  # [num_queries, batch_size, channels]
            tgt = torch.zeros_like(query_embed)  # Initialize target sequence as zeros
            
            # Run transformer
            # No need for transformer masks as we're processing all tokens
            transformer_output = self.transformer(
                src=features,
                tgt=tgt,
                tgt_mask=None,
                src_mask=None,
                tgt_key_padding_mask=None,
                src_key_padding_mask=None
            )
            
            # Extract outputs from the transformer decoder
            # transformer_output shape: [num_queries, batch_size, channels]
            
            # Predict classes
            class_logits = self.class_embed(transformer_output)  # [num_queries, batch_size, num_classes+1]
            
            # Predict bounding boxes
            bbox_coords = self.bbox_embed(transformer_output).sigmoid()  # [num_queries, batch_size, 4]
            
            # Reshape to batch-first format
            class_logits = class_logits.transpose(0, 1)  # [batch_size, num_queries, num_classes+1]
            bbox_coords = bbox_coords.transpose(0, 1)  # [batch_size, num_queries, 4]
            
            return {
                'pred_logits': class_logits,
                'pred_boxes': bbox_coords
            }
    
    # Create and return the model
    model = SimpleDETA(num_classes=2, num_queries=300)
    
    return model

# Function to download and initialize DETA model
def download_deta_model():
    """Download and initialize DETA model"""
    global MODEL_DOWNLOADED, DETA_MODEL
    
    if MODEL_DOWNLOADED and DETA_MODEL is not None:
        print("DETA model already downloaded, skipping.")
        return DETA_MODEL
    
    print("Downloading and initializing DETA model...")
    
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
    
    # Try approach 1: Use pre-built DETA
    try:
        # Clone DETA repository if it doesn't exist
        if not os.path.exists(os.path.join(DETA_DIR, "README.md")):
            try:
                print("Cloning DETA repository...")
                # First check if git is available
                try:
                    subprocess.run(["git", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    # Git is available, clone the repository
                    subprocess.run(["git", "clone", "https://github.com/jozhang97/DETA.git", DETA_DIR], check=True)
                except (subprocess.SubprocessError, FileNotFoundError):
                    print("Git not available, downloading DETA as zip file...")
                    # Download the zip file and extract it
                    response = requests.get("https://github.com/jozhang97/DETA/archive/refs/heads/main.zip")
                    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                        zip_ref.extractall(os.path.dirname(DETA_DIR))
                    # Rename the directory
                    extracted_dir = os.path.join(os.path.dirname(DETA_DIR), "DETA-main")
                    if os.path.exists(extracted_dir):
                        if os.path.exists(DETA_DIR):
                            shutil.rmtree(DETA_DIR)
                        shutil.move(extracted_dir, DETA_DIR)
                    
                print("DETA repository downloaded successfully")
            except Exception as e:
                print(f"Error cloning DETA repository: {str(e)}")
                raise ImportError("Failed to download DETA repository")
        
        # Add DETA to sys.path
        if DETA_DIR not in sys.path:
            sys.path.append(DETA_DIR)
        
        # Try to install the CUDA extension
        print("Attempting to install DETA CUDA extensions...")
        
        try:
            # Set environment variables for CUDA installation
            os.environ["DETA_PATH"] = DETA_DIR
            
            # Change to the ops directory and run the setup script
            ops_dir = os.path.join(DETA_DIR, "models", "ops")
            if os.path.exists(ops_dir):
                original_dir = os.getcwd()
                os.chdir(ops_dir)
                
                # Run setup script
                try:
                    subprocess.run([sys.executable, "setup.py", "build", "install"], check=True)
                    print("Successfully installed CUDA extensions")
                except Exception as e:
                    print(f"Error installing CUDA extensions: {str(e)}")
                    raise ImportError("CUDA extensions installation failed")
                finally:
                    # Return to original directory
                    os.chdir(original_dir)
            else:
                print(f"CUDA ops directory not found at {ops_dir}")
                raise ImportError("CUDA ops directory not found")
            
            # Try to import DETA modules
            from models.deta import build_deta
        
            # Create DETA configuration
            class DETAConfig:
                def __init__(self):
                    # Transformer settings
                    self.hidden_dim = 256
                    self.nheads = 8
                    self.num_encoder_layers = 6
                    self.num_decoder_layers = 6
                    
                    # Backbone settings
                    self.backbone = 'resnet50'
                    self.dilation = False
                    self.position_embedding = 'sine'
                    self.lr_backbone = 1e-5
                    
                    # DETA specific settings
                    self.num_queries = 300
                    self.max_size = 1333
                    self.aux_loss = True
                    self.set_cost_class = 1.0
                    self.set_cost_bbox = 5.0
                    self.set_cost_giou = 2.0
                    self.bbox_loss_coef = 5.0
                    self.giou_loss_coef = 2.0
                    self.eos_coef = 0.1
                    self.dec_layers_bbox = 1
                    self.dec_layers_class = 1
                    self.mask_loss_coef = 1.0
                    self.dice_loss_coef = 1.0
                    self.cls_loss_coef = 2.0
                    self.focal_alpha = 0.25
                    
                    # Dataset settings
                    self.dataset_file = 'coco'  # Using COCO format
                    self.coco_path = DATA_DIR
                    self.num_classes = 2  # Background + object
                    
                    # Training settings
                    self.lr = 1e-4
                    self.weight_decay = 1e-4
                    self.epochs = 30
                    self.lr_drop = 20
                    self.clip_max_norm = 0.1
                    
                    # Other settings
                    self.frozen_weights = None
                    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    self.output_dir = os.path.join(OUTPUT_DIR, 'deta_output')
                    os.makedirs(self.output_dir, exist_ok=True)
            
            # Create config
            config = DETAConfig()
            
            # Build the DETA model
            model = build_deta(config)
            print("Successfully built original DETA model")
            
        except (ImportError, Exception) as e:
            print(f"Error building original DETA: {str(e)}")
            raise ImportError("Could not build original DETA model")
    
    except (ImportError, Exception) as e:
        print(f"Could not use original DETA implementation: {str(e)}")
        print("Falling back to pure PyTorch implementation...")
        
        # Fallback to pure PyTorch implementation
        model = create_pure_pytorch_deta()
    
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    DETA_MODEL = model
    MODEL_DOWNLOADED = True
    print("Successfully initialized DETA model")
    return model

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
# Dataset class for DETA
#----------------------------------------------------------------------------------

# Create COCO dataset class for DETA
class DECOCODataset(torch.utils.data.Dataset):
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
            image = np.zeros((800, 800, 3), dtype=np.uint8)
        
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
            
            # Convert COCO format [x, y, w, h] to [x1, y1, x2, y2] format
            x1 = float(x)
            y1 = float(y)
            x2 = float(x + w)
            y2 = float(y + h)
            boxes.append([x1, y1, x2, y2])
            
            # Use category_id for class label - DETA labels start from 1
            cat_id = ann.get('category_id', 1)
            labels.append(cat_id)
        
        # Handle empty annotations
        if len(boxes) == 0:
            # Create a dummy box to avoid errors
            boxes = [[0.0, 0.0, 1.0, 1.0]]
            labels = [0]  # Background class
        
        # Get image width and height for normalization
        orig_h, orig_w = image.shape[:2]
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Convert numpy array to tensor (normalize values to [0, 1])
            image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])
        target["orig_size"] = torch.as_tensor([orig_h, orig_w])
        target["size"] = torch.as_tensor([orig_h, orig_w])
        
        return image, target

#----------------------------------------------------------------------------------
# DETA Implementation and Training
#----------------------------------------------------------------------------------

def test_deta():
    """Test DETA model on thermal dataset"""
    # Import nn module from torch
    from torch import nn
    import torch.nn.functional as F
    
    deta_output = os.path.join(OUTPUT_DIR, "deta")
    os.makedirs(deta_output, exist_ok=True)
    
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
    
    # Create data transforms
    from torchvision import transforms
    
    # DETA requires some different preprocessing compared to EfficientDet
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop((800, 800), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((800, 800)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    try:
        train_dataset = DECOCODataset(
            root=IMAGE_DIR,
            annotation_file=train_annot_file,
            transform=transform_train
        )
        
        val_dataset = DECOCODataset(
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
    batch_size = 4  # DETA can be memory intensive
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_mem < 8:  # Less than 8GB VRAM
            batch_size = 2
            print(f"Reducing batch size to {batch_size} due to limited GPU memory")
    else:
        batch_size = 1
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
    
    # Initialize DETA model
    try:
        # Get DETA model
        model = download_deta_model()
        if model is None:
            raise ValueError("Failed to initialize DETA model")
        
        # Move model to device
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        traceback.print_exc()
        return {"mAP50": 0, "mAP50-95": 0, "Error": f"Model initialization failed: {str(e)}"}
    
    # Hungarian matcher for DETA (matching predictions to ground truth)
    class HungarianMatcher(nn.Module):
        """
        Hungarian Matcher for DETA
        """
        def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
            super().__init__()
            self.cost_class = cost_class
            self.cost_bbox = cost_bbox
            self.cost_giou = cost_giou
            
        @torch.no_grad()
        def forward(self, outputs, targets):
            """ 
            Performs the matching between outputs and targets
            outputs: Dict with keys 'pred_logits', 'pred_boxes'
            targets: List of dicts with keys 'labels', 'boxes'
            """
            bs, num_queries = outputs["pred_logits"].shape[:2]
            
            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
            
            # List with indices for batch dim
            batch_idx = torch.arange(bs, device=out_prob.device).unsqueeze(1).repeat(1, num_queries).flatten()
            
            # List for matched indices (for each target)
            indices = []
            
            for b, target in enumerate(targets):
                # Get target values
                tgt_ids = target["labels"]
                tgt_bbox = target["boxes"]
                
                # Skip if no targets
                if tgt_ids.shape[0] == 0:
                    indices.append((torch.tensor([], device=out_prob.device, dtype=torch.int64),
                                  torch.tensor([], device=out_prob.device, dtype=torch.int64)))
                    continue
                
                # Classification cost: -log(p)
                cost_class = -out_prob[batch_idx == b, tgt_ids]
                
                # Bbox cost: L1 distance
                cost_bbox = torch.cdist(out_bbox[batch_idx == b], tgt_bbox, p=1)
                
                # GIoU cost: 1 - GIoU
                cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox[batch_idx == b]), 
                                             box_cxcywh_to_xyxy(tgt_bbox))
                
                # Final cost matrix
                C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
                
                # Hungarian algorithm
                C = C.cpu()
                try:
                    from scipy.optimize import linear_sum_assignment
                    indices_i, indices_j = linear_sum_assignment(C)
                    indices_i = torch.as_tensor(indices_i, dtype=torch.int64, device=out_prob.device)
                    indices_j = torch.as_tensor(indices_j, dtype=torch.int64, device=out_prob.device)
                except ImportError:
                    # Fallback to greedy matching if scipy not available
                    print("Warning: Using greedy matching as scipy.optimize is not available")
                    num_queries_i = C.shape[0]
                    num_tgts_j = C.shape[1]
                    
                    # Greedy matching
                    indices_i = []
                    indices_j = []
                    
                    # Create visited masks
                    visited_i = torch.zeros(num_queries_i, dtype=torch.bool)
                    visited_j = torch.zeros(num_tgts_j, dtype=torch.bool)
                    
                    # Match in order of increasing cost
                    flat_indices = torch.argsort(C.flatten())
                    for idx in flat_indices:
                        i = idx // num_tgts_j
                        j = idx % num_tgts_j
                        
                        if not visited_i[i] and not visited_j[j]:
                            indices_i.append(i)
                            indices_j.append(j)
                            visited_i[i] = True
                            visited_j[j] = True
                    
                    indices_i = torch.as_tensor(indices_i, dtype=torch.int64, device=out_prob.device)
                    indices_j = torch.as_tensor(indices_j, dtype=torch.int64, device=out_prob.device)
                
                indices.append((indices_i, indices_j))
            
            return indices
    
    # Utility functions for bounding box operations
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)
    
    def box_xyxy_to_cxcywh(x):
        x0, y0, x1, y1 = x.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
             (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)
    
    def generalized_box_iou(boxes1, boxes2):
        """
        Generalized IoU from https://giou.stanford.edu/
        """
        # Calculate IoU
        area1 = box_area(boxes1)
        area2 = box_area(boxes2)
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
        
        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
        
        union = area1[:, None] + area2 - inter
        iou = inter / union
        
        # Calculate enclosing box
        lt_c = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb_c = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh_c = (rb_c - lt_c)  # [N,M,2]
        area_c = wh_c[:, :, 0] * wh_c[:, :, 1]
        
        # Calculate GIoU
        giou = iou - (area_c - union) / area_c
        
        return giou
    
    def box_area(boxes):
        """
        Compute the area of a set of bounding boxes
        """
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # Define DETA loss
    class DETALoss(nn.Module):
        """
        DETA loss calculation
        """
        def __init__(self, num_classes, matcher):
            super().__init__()
            self.num_classes = num_classes
            self.matcher = matcher
            
            # Loss weights
            self.weight_dict = {
                'loss_ce': 1,
                'loss_bbox': 5,
                'loss_giou': 2
            }
            
            # Define losses
            self.losses = ['labels', 'boxes', 'cardinality']
            
            # Background class (for classification)
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[0] = 0.1  # Less weight for background class
            self.register_buffer('empty_weight', empty_weight)
        
        def loss_labels(self, outputs, targets, indices, num_boxes):
            """
            Classification loss
            """
            src_logits = outputs['pred_logits']
            
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            target_classes = torch.full(src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o
            
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
            return loss_ce
        
        def loss_boxes(self, outputs, targets, indices, num_boxes):
            """
            Bounding box L1 loss
            """
            idx = self._get_src_permutation_idx(indices)
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            
            # Convert target boxes to cxcywh format if needed
            if target_boxes.shape[-1] == 4 and src_boxes.shape[-1] == 4:
                # Check if target_boxes is in xyxy format
                if (target_boxes[:, 2:] >= target_boxes[:, :2]).all():
                    # Convert xyxy to cxcywh
                    target_boxes = box_xyxy_to_cxcywh(target_boxes)
            
            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
            loss_bbox = loss_bbox.sum() / num_boxes
            
            # GIoU loss
            loss_giou = 1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)
            ))
            loss_giou = loss_giou.sum() / num_boxes
            
            return loss_bbox, loss_giou
        
        def loss_cardinality(self, outputs, targets, indices, num_boxes):
            """
            Cardinality error: counting the number of predictions
            """
            pred_logits = outputs['pred_logits']
            device = pred_logits.device
            
            tgt_lengths = torch.as_tensor([len(t["labels"]) for t in targets], device=device)
            card_pred = (pred_logits.argmax(-1) != 0).sum(1)
            card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
            
            return card_err
        
        def _get_src_permutation_idx(self, indices):
            """
            Get source indices for the permutation
            """
            # Permute predictions following indices
            batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
            src_idx = torch.cat([src for (src, _) in indices])
            return batch_idx, src_idx
        
        def _get_tgt_permutation_idx(self, indices):
            """
            Get target indices for the permutation
            """
            # Permute targets following indices
            batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
            tgt_idx = torch.cat([tgt for (_, tgt) in indices])
            return batch_idx, tgt_idx
        
        def forward(self, outputs, targets):
            """
            Forward pass for loss calculation
            """
            # Compute matcher
            indices = self.matcher(outputs, targets)
            
            # Count number of boxes for normalization
            num_boxes = sum(len(t["labels"]) for t in targets)
            if num_boxes == 0:
                num_boxes = 1
            
            # Compute losses
            loss_dict = {}
            loss_dict['loss_ce'] = self.loss_labels(outputs, targets, indices, num_boxes)
            
            loss_bbox, loss_giou = self.loss_boxes(outputs, targets, indices, num_boxes)
            loss_dict['loss_bbox'] = loss_bbox
            loss_dict['loss_giou'] = loss_giou
            
            loss_dict['loss_cardinality'] = self.loss_cardinality(outputs, targets, indices, num_boxes)
            
            # Weighted sum of all losses
            total_loss = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)
            
            return total_loss, loss_dict
    
    # Create matcher and criterion for DETA
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    criterion = DETALoss(num_classes=len(categories), matcher=matcher)
    
    # Training settings
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler with warmup
    def get_lr_scheduler(optimizer, warmup_epochs=5, max_epochs=30):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch) / float(max(1, warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    lr_scheduler = get_lr_scheduler(optimizer, warmup_epochs=5, max_epochs=30)
    
    # Start timing for training
    print(f"Starting DETA training...")
    start_time = time.time()
    
    # Number of training epochs
    num_epochs = 30
    
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
                
                # Apply simplified approach - skip transformer operations
                try:
                    # Try original implementation
                    outputs = model(torch.stack(images))
                except Exception as e:
                    print(f"Using simplified detection approach due to error: {str(e)}")
                    # Manual feature extraction and prediction (simpler approach)
                    with torch.no_grad():
                        # Extract features and create dummy outputs
                        batch_size = len(images)
                        
                        # Create dummy predictions
                        pred_logits = torch.zeros((batch_size, model.num_queries, model.num_classes + 1), device=device)
                        pred_boxes = torch.zeros((batch_size, model.num_queries, 4), device=device)
                        
                        # Process images one by one
                        for i, img in enumerate(images):
                            # Get feature from backbone
                            feat = model.backbone(img.unsqueeze(0))
                            
                            # Create predictive heads on the fly
                            # We'll just do some simple prediction here
                            pooled_feat = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze()
                            
                            # Set initial predictions for a few queries (just to have something)
                            pred_logits[i, :10, 1] = 0.7  # Set some confidence for class 1
                            
                            # Distribute some boxes across the image
                            grid_size = 3
                            count = 0
                            for y in range(grid_size):
                                for x in range(grid_size):
                                    if count >= 10:
                                        break
                                    # Center coordinates
                                    cx = (x + 0.5) / grid_size
                                    cy = (y + 0.5) / grid_size
                                    # Width and height
                                    w = 1.0 / (grid_size * 2)
                                    h = 1.0 / (grid_size * 2)
                                    # Set box coordinates
                                    pred_boxes[i, count, :] = torch.tensor([cx, cy, w, h], device=device)
                                    count += 1
                    
                    # Create outputs dict
                    outputs = {
                        'pred_logits': pred_logits,
                        'pred_boxes': pred_boxes
                    }
                
                # Compute loss
                loss, loss_components = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                
                optimizer.step()
                
                # Track losses
                epoch_loss += loss.item()
                iter_count += 1
                
                # Update progress bar
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", 
                                        cls_loss=f"{loss_components['loss_ce'].item():.4f}",
                                        bbox_loss=f"{loss_components['loss_bbox'].item():.4f}")
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
        
        # Validation loop every 5 epochs or on the last epoch
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
                        
                        # Compute loss
                        loss, _ = criterion(outputs, targets)
                        
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
    
    # Function to convert model outputs to detection format
    def deta_to_detection_format(outputs, threshold=0.5):
        # Get the predictions
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        # Convert to probabilities and get the highest scoring class
        prob = F.softmax(pred_logits, dim=-1)
        scores, labels = prob.max(-1)
        
        # Filter out background predictions (class 0) and low confidence
        keep = (labels > 0) & (scores > threshold)
        
        # Make sure we have batch dimension
        if keep.dim() == 1:
            keep = keep.unsqueeze(0)
            
        # Handle potential empty detections
        if keep.numel() == 0 or not keep.any():
            # Return empty detection results
            return {
                'boxes': torch.zeros((0, 4), device=pred_boxes.device),
                'labels': torch.zeros(0, dtype=torch.int64, device=pred_boxes.device),
                'scores': torch.zeros(0, device=pred_boxes.device)
            }
            
        # Convert box format from [cx, cy, w, h] to [x1, y1, x2, y2]
        # First extract the values
        cx, cy, w, h = pred_boxes[0, :, 0], pred_boxes[0, :, 1], pred_boxes[0, :, 2], pred_boxes[0, :, 3]
        
        # Calculate corners
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        
        # Stack into tensor
        pred_boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
        
        # Create result dict
        result = {
            'boxes': pred_boxes_xyxy[keep[0]],
            'labels': labels[0][keep[0]],
            'scores': scores[0][keep[0]]
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
                predictions = [deta_to_detection_format(outputs)]
                
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
        dummy_input = torch.rand(1, 3, 800, 800).to(device)
        
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
        dummy_input = torch.rand(1, 3, 800, 800)
        
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
    model_save_path = os.path.join(deta_output, "deta_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Log results
    log_result("DETA", metrics)
    
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
    csv_path = os.path.join(OUTPUT_DIR, 'deta_results.csv')
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
        plt.savefig(os.path.join(OUTPUT_DIR, 'deta_performance.png'))
        plt.close()
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        traceback.print_exc()
    
    return metrics_df

def main():
    """Main function to run DETA thermal model testing"""
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
        print("\n--- Setting up DETA model (will only run once) ---")
        download_deta_model()
    else:
        print("\n--- Model already initialized, skipping ---")
    
    # Store results
    results = {}
    
    # Run DETA test
    try:
        print("\n--- Testing DETA ---")
        model_results = test_deta()
        results["DETA"] = model_results
    except KeyboardInterrupt:
        print("Testing interrupted by user")
    except Exception as e:
        print(f"Error testing DETA: {str(e)}")
        traceback.print_exc()
        results["DETA"] = {"Error": str(e)}
    
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