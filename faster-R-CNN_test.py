# Comprehensive Thermal Model Testing Framework
# Testing Faster R-CNN on FLIR Thermal Dataset

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
ANNOTATIONS_FILE = os.path.join(DATA_DIR, "annotations.json")  # Updated path to correct file

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
FASTER_RCNN_MODEL = None  # Global variable to store the model
filtered_coco_data = None  # Global variable to store the filtered data

# Define annotation file paths
train_annot_file = os.path.join(OUTPUT_DIR, "train_annotations.json")
val_annot_file = os.path.join(OUTPUT_DIR, "val_annotations.json")
filtered_annot_file = os.path.join(OUTPUT_DIR, "filtered_annotations.json")

# Function to download and initialize Faster R-CNN model
def download_faster_rcnn():
    """Download and initialize Faster R-CNN model"""
    global MODEL_DOWNLOADED, FASTER_RCNN_MODEL
    
    if MODEL_DOWNLOADED and FASTER_RCNN_MODEL is not None:
        print("Faster R-CNN model already downloaded, skipping.")
        return FASTER_RCNN_MODEL
    
    print("Downloading and initializing Faster R-CNN model...")
    
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
    
    # Initialize Faster R-CNN
    try:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
        print("Initializing Faster R-CNN ResNet50 FPN...")
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        FASTER_RCNN_MODEL = model
        MODEL_DOWNLOADED = True
        print("Successfully initialized Faster R-CNN ResNet50 FPN")
        return model
    except Exception as e:
        print(f"Error initializing Faster R-CNN: {str(e)}")
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
# Dataset class for Faster R-CNN
#----------------------------------------------------------------------------------

# Create COCO dataset class for Faster R-CNN
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
            
            # Convert COCO format [x, y, w, h] to PyTorch format [x1, y1, x2, y2]
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            boxes.append([x1, y1, x2, y2])
            
            # Use 1 for all classes (single class model)
            # 0 is reserved for background in Faster R-CNN
            labels.append(1)
        
        # Handle empty annotations
        if len(boxes) == 0:
            # Create a dummy box to avoid errors
            boxes = [[0, 0, 1, 1]]
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
# Faster R-CNN Implementation
#----------------------------------------------------------------------------------

def test_faster_rcnn():
    """Test Faster R-CNN model on thermal dataset"""
    frcnn_output = os.path.join(OUTPUT_DIR, "faster_rcnn")
    os.makedirs(frcnn_output, exist_ok=True)
    
    # Ensure required packages are installed
    try:
        import torchvision
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        print("Torchvision already installed")
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "--user", "torchvision"], check=True)
        try:
            import torchvision
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
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
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    
    # Load datasets
    try:
        train_dataset = COCODataset(
            root=IMAGE_DIR,
            annotation_file=os.path.join(OUTPUT_DIR, "train_annotations.json"),
            transform=transform
        )
        
        val_dataset = COCODataset(
            root=IMAGE_DIR,
            annotation_file=os.path.join(OUTPUT_DIR, "val_annotations.json"),
            transform=transform
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
    batch_size = 4  # Smaller batch size for Faster R-CNN due to memory usage
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
    
    # Initialize Faster R-CNN model
    try:
        # Import needed for weights parameter
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        
        # Load pre-trained model
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        
        # Replace the box predictor with a new one for our dataset (2 classes: background + object)
        num_classes = 2  # Background + 1 object class
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Move model to device
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        traceback.print_exc()
        return {"mAP50": 0, "mAP50-95": 0, "Error": f"Model initialization failed: {str(e)}"}
    
    # Training settings
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )
    
    # Start timing for training
    print(f"Starting Faster R-CNN training...")
    start_time = time.time()
    
    # Number of training epochs
    num_epochs = 10  # Reduced for testing
    
    # Create a DataFrame to store per-epoch metrics
    epoch_metrics_df = pd.DataFrame(columns=[
        'Epoch', 'Loss', 'mAP50', 'mAP50-95', 'Parameters (M)', 
        'Trainable Parameters (M)', 'GFLOPs', 'Training Time (min)', 
        'Inference Time (ms)'
    ])
    
    # Save metrics CSV path
    per_epoch_csv = os.path.join(OUTPUT_DIR, 'faster_rcnn_per_epoch_metrics.csv')
    
    # Calculate initial metrics for model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate GFLOPs (do this once before training)
    try:
        # Try to import thop
        try:
            from thop import profile
        except ImportError:
            print("Installing thop for FLOPs calculation...")
            subprocess.run([sys.executable, "-m", "pip", "install", "thop"], check=True)
            from thop import profile
        
        model.eval()
        dummy_img = [torch.zeros((3, 640, 640), device=device)]
        with torch.no_grad():
            try:
                macs, _ = profile(model, inputs=(dummy_img,), verbose=False)
                gflops = macs * 2 / 1e9
                print(f"Initial model GFLOPs: {gflops:.2f}")
            except Exception:
                gflops = total_params * 2 / 1e9  # Fallback to parameter-based estimate
                print(f"Estimated GFLOPs: ~{gflops:.2f} (based on param count)")
    except Exception:
        gflops = "N/A"
    
    # Initialize metric tracking
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    
    # Training loop
    epoch_start_time = time.time()
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        
        # Track metrics
        epoch_loss = 0
        iter_count = 0
        
        # Progress bar
        print(f"Epoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(train_loader, desc=f"Training", leave=False)
        
        # Track epoch start time
        start_time = time.time()
        
        for images, targets in progress_bar:
            try:
                # Move images and targets to device
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # Backward pass and optimize
                losses.backward()
                optimizer.step()
                
                # Track losses
                epoch_loss += losses.item()
                iter_count += 1
                
                # Update progress bar
                progress_bar.set_postfix(loss=f"{losses.item():.4f}")
            except Exception as e:
                print(f"Error in training iteration: {str(e)}")
                continue
        
        # Update learning rate
        lr_scheduler.step()
        
        # Calculate epoch time
        epoch_time = (time.time() - start_time) / 60.0  # minutes
        total_time = (time.time() - epoch_start_time) / 60.0  # total minutes so far
        
        # Print epoch stats
        if iter_count > 0:
            avg_loss = epoch_loss / iter_count
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Time: {epoch_time:.2f} min")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, No valid iterations")
            avg_loss = 0
        
        # Run evaluation at the end of each epoch
        print(f"Evaluating after epoch {epoch+1}...")
        model.eval()
        
        # Initialize metrics
        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        
        # Evaluate on validation dataset
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Evaluating", leave=False):
                try:
                    # Move images to device
                    images = [image.to(device) for image in images]
                    
                    # Get predictions
                    predictions = model(images)
                    
                    # Prepare targets in the format expected by the metric
                    for t in targets:
                        # Convert device tensors to CPU for the metric
                        t["boxes"] = t["boxes"].cpu()
                        t["labels"] = t["labels"].cpu()
                    
                    # Prepare predictions in the format expected by the metric
                    for p in predictions:
                        # Convert device tensors to CPU for the metric
                        p["boxes"] = p["boxes"].cpu()
                        p["labels"] = p["labels"].cpu()
                        p["scores"] = p["scores"].cpu()
                    
                    # Update metric
                    metric.update(predictions, targets)
                except Exception as e:
                    print(f"Error in evaluation: {str(e)}")
                    continue
        
        # Compute metrics
        metrics_dict = metric.compute()
        
        # Extract relevant metrics
        map50 = metrics_dict["map_50"].item() if "map_50" in metrics_dict else 0
        map50_95 = metrics_dict["map"].item() if "map" in metrics_dict else 0
        
        # Measure inference time
        if torch.cuda.is_available():
            dummy_input = [torch.rand(3, 640, 640).to(device)]
            
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
        else:
            # Use CPU timing
            dummy_input = [torch.rand(3, 640, 640)]
            
            # Warm up
            for _ in range(2):
                _ = model(dummy_input)
            
            # Measure
            t0 = time.time()
            iterations = 5
            for _ in range(iterations):
                _ = model(dummy_input)
            
            inference_time = (time.time() - t0) * 1000 / iterations  # ms
        
        # Store per-epoch metrics
        epoch_metrics = {
            'Epoch': epoch + 1,
            'Loss': avg_loss,
            'mAP50': map50,
            'mAP50-95': map50_95,
            'Parameters (M)': total_params / 1e6,
            'Trainable Parameters (M)': trainable_params / 1e6,
            'GFLOPs': gflops if isinstance(gflops, (int, float)) else "N/A",
            'Training Time (min)': total_time,
            'Inference Time (ms)': inference_time
        }
        
        # Add to DataFrame
        epoch_metrics_df = pd.concat([epoch_metrics_df, pd.DataFrame([epoch_metrics])], ignore_index=True)
        
        # Save to CSV after each epoch
        epoch_metrics_df.to_csv(per_epoch_csv, index=False)
        
        # Print metrics summary
        print(f"\nEpoch {epoch+1} Metrics:")
        print(f"  mAP50: {map50:.4f}")
        print(f"  mAP50-95: {map50_95:.4f}")
        print(f"  Parameters (M): {total_params / 1e6:.2f}")
        print(f"  Trainable Parameters (M): {trainable_params / 1e6:.2f}")
        print(f"  GFLOPs: {gflops if isinstance(gflops, (int, float)) else 'N/A'}")
        print(f"  Training Time (min): {total_time:.2f}")
        print(f"  Inference Time (ms): {inference_time:.2f}")
    
    # Calculate training time
    training_time = (time.time() - start_time) / 60.0  # minutes
    print(f"Training completed in {training_time:.2f} minutes")
    
    # Evaluation
    print("Evaluating model...")
    model.eval()
    
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
                predictions = model(images)
                
                # Prepare targets in the format expected by the metric
                for t in targets:
                    # Convert device tensors to CPU for the metric
                    t["boxes"] = t["boxes"].cpu()
                    t["labels"] = t["labels"].cpu()
                
                # Prepare predictions in the format expected by the metric
                for p in predictions:
                    # Convert device tensors to CPU for the metric
                    p["boxes"] = p["boxes"].cpu()
                    p["labels"] = p["labels"].cpu()
                    p["scores"] = p["scores"].cpu()
                
                # Update metric
                metric.update(predictions, targets)
            except Exception as e:
                print(f"Error in evaluation: {str(e)}")
                continue
    
    # Compute metrics
    metrics_dict = metric.compute()
    
    # Extract relevant metrics
    metrics = {}
    metrics["mAP50"] = metrics_dict["map_50"].item() if "map_50" in metrics_dict else 0
    metrics["mAP50-95"] = metrics_dict["map"].item() if "map" in metrics_dict else 0
    
    # Calculate both total and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percentage = (trainable_params / total_params * 100) if total_params > 0 else 0
    
    print(f"\nParameter count: {total_params:,} total, {trainable_params:,} trainable ({percentage:.2f}%)")
    
    metrics["Parameters (M)"] = total_params / 1e6
    metrics["Trainable Parameters (M)"] = trainable_params / 1e6
    metrics["Training Time (min)"] = training_time
    
    # Calculate GFLOPs
    try:
        # Install thop if needed
        try:
            from thop import profile
        except ImportError:
            print("Installing thop for FLOPs calculation...")
            subprocess.run([sys.executable, "-m", "pip", "install", "thop"], check=True)
            from thop import profile
        
        # Create dummy input
        model.eval()
        dummy_img = [torch.zeros((3, 640, 640), device=device)]
        
        # Profile with thop
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            try:
                # Method 1: Direct thop profiling
                macs, _ = profile(model, inputs=(dummy_img,), verbose=False)
                gflops = macs * 2 / 1e9
                metrics["GFLOPs"] = gflops
                print(f"Calculated GFLOPs: {gflops:.2f}")
            except Exception as e1:
                print(f"Primary GFLOPs calculation failed: {str(e1)}")
                
                try:
                    # Method 2: Alternative approach via timed inference
                    print("Trying alternative GFLOPs calculation method...")
                    
                    # Time the forward pass
                    iterations = 10
                    start = time.time()
                    for _ in range(iterations):
                        _ = model(dummy_img)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end = time.time()
                    
                    # Estimate GFLOPs based on parameter count (rough approximation)
                    est_gflops = total_params * 2 / 1e9  # Very rough estimate
                    
                    metrics["GFLOPs"] = est_gflops
                    print(f"Estimated GFLOPs: ~{est_gflops:.2f} (based on param count)")
                except Exception as e2:
                    print(f"Alternative GFLOPs calculation also failed: {str(e2)}")
                    metrics["GFLOPs"] = "N/A"
    except Exception as e:
        print(f"GFLOPs calculation failed completely: {str(e)}")
        metrics["GFLOPs"] = "N/A"
    
    # Estimate inference time
    if torch.cuda.is_available():
        # Create sample image for timing
        dummy_input = [torch.rand(3, 640, 640).to(device)]
        
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
        dummy_input = [torch.rand(3, 640, 640)]
        
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
    model_save_path = os.path.join(frcnn_output, "faster_rcnn_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Log results
    log_result("Faster_RCNN", metrics)
    
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
        'Trainable Parameters (M)': [results[model].get('Trainable Parameters (M)', results[model].get('Parameters (M)', 0)) for model in results],
        'GFLOPs': [results[model].get('GFLOPs', 'N/A') for model in results],
        'Inference Time (ms)': [results[model].get('Inference Time (ms)', 0) for model in results],
        'Training Time (min)': [results[model].get('Training Time (min)', 'N/A') for model in results],
        'GPU Memory (GB)': [results[model].get('GPU Memory (GB)', 'N/A') for model in results],
        'Error': [results[model].get('Error', '') for model in results]
    })
    
    # Save to CSV first in case later visualization steps fail
    csv_path = os.path.join(OUTPUT_DIR, 'faster_rcnn_results.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Check if we have valid data to visualize
    if metrics_df.empty or metrics_df['mAP50'].max() == 0:
        print("No valid performance metrics to visualize. Skipping visualization.")
        return metrics_df
    
    # Create visualizations
    try:
        plt.figure(figsize=(15, 15))
        
        # 1. Accuracy (mAP50) bar chart
        plt.subplot(3, 2, 1)
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
        plt.subplot(3, 2, 2)
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
        plt.subplot(3, 2, 3)
        
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
        
        # 4. Total vs Trainable Parameters
        plt.subplot(3, 2, 4)
        
        # Create a side-by-side bar chart
        x = np.arange(len(metrics_df['Model']))
        width = 0.35
        
        fig, ax = plt.subplots()
        ax.bar(x - width/2, metrics_df['Parameters (M)'], width, label='Total Parameters', color='purple')
        ax.bar(x + width/2, metrics_df['Trainable Parameters (M)'], width, label='Trainable Parameters', color='magenta')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Parameters (M)')
        ax.set_title('Total vs Trainable Parameters')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df['Model'])
        ax.legend()
        
        # Save this plot separately
        plt.savefig(os.path.join(OUTPUT_DIR, 'faster_rcnn_parameters_comparison.png'))
        plt.close()
        
        # 5. GFLOPs comparison
        plt.subplot(3, 2, 5)
        # Filter numeric GFLOPs
        gflop_data = metrics_df.copy()
        gflop_data = gflop_data[gflop_data['GFLOPs'] != 'N/A']
        
        if not gflop_data.empty:
            # Convert to numeric if needed
            if isinstance(gflop_data['GFLOPs'].iloc[0], str):
                gflop_data['GFLOPs'] = pd.to_numeric(gflop_data['GFLOPs'], errors='coerce')
            
            bars = plt.bar(gflop_data['Model'], gflop_data['GFLOPs'], color='teal')
            plt.title('GFLOPs Comparison', fontsize=14)
            plt.ylabel('GFLOPs')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')
        
        # 6. Per-epoch metrics visualization
        plt.subplot(3, 2, 6)
        epoch_csv = os.path.join(OUTPUT_DIR, 'faster_rcnn_per_epoch_metrics.csv')
        
        if os.path.exists(epoch_csv):
            try:
                epoch_df = pd.read_csv(epoch_csv)
                plt.plot(epoch_df['Epoch'], epoch_df['mAP50'], 'b-o', label='mAP50')
                plt.plot(epoch_df['Epoch'], epoch_df['mAP50-95'], 'r-o', label='mAP50-95')
                plt.title('Performance by Epoch', fontsize=14)
                plt.xlabel('Epoch')
                plt.ylabel('mAP')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
            except Exception as e:
                print(f"Error plotting epoch metrics: {e}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'faster_rcnn_performance.png'))
        plt.close()
        
        # Also create a separate chart for per-epoch metrics
        if os.path.exists(epoch_csv):
            try:
                epoch_df = pd.read_csv(epoch_csv)
                
                plt.figure(figsize=(15, 10))
                
                # Plot mAP metrics over epochs
                plt.subplot(2, 2, 1)
                plt.plot(epoch_df['Epoch'], epoch_df['mAP50'], 'b-o', label='mAP50')
                plt.plot(epoch_df['Epoch'], epoch_df['mAP50-95'], 'r-o', label='mAP50-95')
                plt.title('mAP over Epochs', fontsize=14)
                plt.xlabel('Epoch')
                plt.ylabel('mAP')
                plt.grid(True, alpha=0.7)
                plt.legend()
                
                # Plot loss over epochs
                plt.subplot(2, 2, 2)
                plt.plot(epoch_df['Epoch'], epoch_df['Loss'], 'g-o')
                plt.title('Loss over Epochs', fontsize=14)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.7)
                
                # Plot inference time over epochs
                plt.subplot(2, 2, 3)
                plt.plot(epoch_df['Epoch'], epoch_df['Inference Time (ms)'], 'm-o')
                plt.title('Inference Time over Epochs', fontsize=14)
                plt.xlabel('Epoch')
                plt.ylabel('Inference Time (ms)')
                plt.grid(True, alpha=0.7)
                
                # Plot training time over epochs
                plt.subplot(2, 2, 4)
                plt.plot(epoch_df['Epoch'], epoch_df['Training Time (min)'], 'c-o')
                plt.title('Cumulative Training Time', fontsize=14)
                plt.xlabel('Epoch')
                plt.ylabel('Time (min)')
                plt.grid(True, alpha=0.7)
                
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, 'faster_rcnn_epoch_metrics.png'))
                plt.close()
                
            except Exception as e:
                print(f"Error creating epoch metrics visualization: {e}")
                traceback.print_exc()
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        traceback.print_exc()
    
    return metrics_df

def main():
    """Main function to run Faster R-CNN thermal model testing"""
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
        print("\n--- Setting up Faster R-CNN model (will only run once) ---")
        download_faster_rcnn()
    else:
        print("\n--- Model already initialized, skipping ---")
    
    # Store results
    results = {}
    
    # Run Faster R-CNN test
    try:
        print("\n--- Testing Faster R-CNN ---")
        model_results = test_faster_rcnn()
        results["Faster R-CNN"] = model_results
    except KeyboardInterrupt:
        print("Testing interrupted by user")
    except Exception as e:
        print(f"Error testing Faster R-CNN: {str(e)}")
        traceback.print_exc()
        results["Faster R-CNN"] = {"Error": str(e)}
    
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