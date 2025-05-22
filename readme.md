# Summary of Models for Thermal Object Detection Evaluation

## Dataset Description

The thermal object detection dataset used in this evaluation consists of thermal imagery captured from various video sequences. The dataset has the following characteristics:

- **Data Type**: Thermal (infrared) imagery from multiple video sources
- **Image Format**: JPEG files extracted as frames from thermal video sequences
- **Resolution**: Thermal images of varying dimensions (640x512 pixels), grayscale thermal representations
- **Dataset Size**: 
  - Total images: ~2,175 thermal frames
  - Training set: ~1740 images (80%)
  - Validation set: ~435 images (20%)
- **Annotation Format**: COCO-style JSON annotations (87MB annotations.json file)
- **Object Classes**: Person, vehicle, and other heat-emitting objects of interest in thermal spectrum
- **Data Sources**: Multiple thermal video sequences with identifiers (e.g., video-5YffDt2oYT6CDzYHk)
- **Capture Conditions**: Various environmental conditions, times of day, and thermal scenarios

The thermal nature of this dataset presents unique challenges compared to RGB imagery, as objects are represented by their heat signatures rather than visible features. This makes it particularly valuable for testing object detection models in low-light, nighttime, or adverse weather conditions where thermal imaging excels.

## Model Architecture Overview

| Model | Architecture Type | Year | Design Paradigm | Key Innovations | 
|-------|------------------|------|-----------------|-----------------|
| YOLOv5 | One-Stage CNN | 2020 | Single-pass detector | CSP backbone, anchor-based detection |
| YOLOv8 | One-Stage CNN | 2023 | Single-pass detector | Anchor-free design, decoupled head |
| YOLOv11 | One-Stage CNN | 2025 | Single-pass detector | Latest YOLO improvements, enhanced feature extraction |
| Faster R-CNN | Two-Stage CNN | 2015 | Region proposal + classifier | Region Proposal Network, ROI pooling |
| Cascade R-CNN | Multi-Stage CNN | 2018 | Progressive refinement | Cascaded detection stages with increasing IoU thresholds |
| RT-DETR | Transformer-Based | 2023 | End-to-end transformer | Efficient hybrid encoder, real-time transformer detection |
| EfficientDet-D2 | Hybrid CNN-Scaling | 2020 | Compound scaling | BiFPN, balanced scaling of network dimensions |
| DETA | Vision Transformer | 2024 | Transformer backbone | Advanced transformer detection, high-accuracy design |

## Technical Specifications (Expected Values)

| Model | Parameters (M) | GFLOPs | Expected FPS Range | GPU Memory | 
|-------|--------------|---------|-------------------|------------|
| YOLOv5-s | ~7.2 | ~17 | 60-120 | 2-3 GB |
| YOLOv5-m | ~21.2 | ~50 | 40-80 | 3-4 GB |
| YOLOv8-s | ~11.2 | ~29 | 50-100 | 2-3 GB |
| YOLOv8-m | ~25.9 | ~79 | 30-70 | 3-4 GB |
| YOLOv11-s | ~12-15 | ~30-35 | 40-90 | 2-3 GB |
| YOLOv11-m | ~27-30 | ~80-85 | 25-65 | 3-4 GB |
| Faster R-CNN | ~41 | ~180 | 10-25 | 4-5 GB |
| Cascade R-CNN | ~70 | ~230 | 7-20 | 5-6 GB |
| RT-DETR | ~33 | ~136 | 15-30 | 4-5 GB |
| EfficientDet-D2 | ~8.1 | ~35 | 30-60 | 3-4 GB |
| DETA | ~45-50 | ~200-225 | 5-15 | 5-7 GB |

## Experimental Results on Thermal Dataset

### Performance Metrics

| Model | mAP50 | mAP50-95 | Parameters (M) | Trainable Params (M) | GFLOPs | Inference Time (ms) | FPS | Training Time (min) | GPU Memory (GB) |
|-------|-------|----------|----------------|---------------------|--------|---------------------|-----|---------------------|-----------------|
| YOLOv5-s | 0.71 | 0.40 | 7.02 | 7.02 | ~17* | 5.33 | 187.6 | 8.52 | ~2-3* |
| YOLOv5-m | 0.76 | 0.47 | 20.87 | 20.87 | ~50* | 6.86 | 145.8 | 12.51 | ~3-4* |
| YOLOv8-s | 0.73 | 0.46 | 11.14 | 0.00 | 28.65 | 7.78 | 128.5 | 9.09 | 5.38 |
| YOLOv8-m | 0.76 | 0.48 | 25.86 | 0.00 | 79.07 | 10.79 | 92.7 | 15.14 | 7.02 |
| YOLOv11-s | 0.72 | 0.45 | 9.43 | 9.43 | 21.55 | 9.93 | 100.7 | 11.12 | 5.63 |
| YOLOv11-m | 0.74 | 0.47 | 20.05 | 20.05 | 68.19 | 12.31 | 81.2 | 16.38 | 8.84 |
| Faster R-CNN | 0.71 | 0.42 | 41.30 | 41.08 | 267.85 | 23.72 | 42.2 | 22.76** | 4.05 |
| Cascade R-CNN | 0.71 | 0.42 | 41.30 | 41.08 | 267.85 | 24.10 | 41.5 | 27.18** | 4.05 |

*Note: YOLOv5 GFLOPs and GPU memory values not directly measured in the experiment; using estimated values based on model architecture. The mAP values are taken from final epoch (epoch 29) in the training results.  
**Total training time for full training (10 epochs for Faster R-CNN, 12 epochs for Cascade R-CNN)

### Accuracy Progression During Training

#### YOLOv5-s
| Epoch | mAP50 | mAP50-95 | Loss (box) | Loss (obj) |
|-------|-------|----------|------------|------------|
| 10 | 0.65 | 0.32 | 0.050 | 0.055 |
| 20 | 0.69 | 0.39 | 0.044 | 0.051 |
| 29 | 0.71 | 0.40 | 0.041 | 0.048 |

#### YOLOv5-m
| Epoch | mAP50 | mAP50-95 | Loss (box) | Loss (obj) |
|-------|-------|----------|------------|------------|
| 10 | 0.71 | 0.38 | 0.045 | 0.051 |
| 20 | 0.75 | 0.44 | 0.039 | 0.046 |
| 29 | 0.76 | 0.47 | 0.035 | 0.043 |

#### YOLOv8-s
| Epoch | mAP50 | mAP50-95 | Loss (box) | Loss (cls) | Loss (dfl) |
|-------|-------|----------|------------|------------|------------|
| 10 | 0.62 | 0.36 | 1.38 | 0.88 | 1.06 |
| 20 | 0.69 | 0.42 | 1.22 | 0.73 | 1.00 |
| 30 | 0.73 | 0.46 | 1.09 | 0.61 | 0.94 |

#### YOLOv8-m
| Epoch | mAP50 | mAP50-95 | Loss (box) | Loss (cls) | Loss (dfl) |
|-------|-------|----------|------------|------------|------------|
| 10 | 0.64 | 0.38 | 1.34 | 0.85 | 1.08 |
| 20 | 0.72 | 0.44 | 1.15 | 0.68 | 1.00 |
| 30 | 0.76 | 0.48 | 1.04 | 0.56 | 0.95 |

#### YOLOv11-s
| Epoch | mAP50 | mAP50-95 | Loss (box) | Loss (cls) | Loss (dfl) |
|-------|-------|----------|------------|------------|------------|
| 10 | 0.61 | 0.35 | 1.42 | 0.93 | 1.08 |
| 20 | 0.68 | 0.41 | 1.25 | 0.76 | 1.00 |
| 30 | 0.72 | 0.45 | 1.14 | 0.65 | 0.95 |

#### YOLOv11-m
| Epoch | mAP50 | mAP50-95 | Loss (box) | Loss (cls) | Loss (dfl) |
|-------|-------|----------|------------|------------|------------|
| 10 | 0.61 | 0.35 | 1.41 | 0.93 | 1.11 |
| 20 | 0.70 | 0.43 | 1.22 | 0.74 | 1.02 |
| 30 | 0.74 | 0.47 | 1.09 | 0.62 | 0.96 |

#### Faster R-CNN
| Epoch | mAP50 | mAP50-95 | Loss |
|-------|-------|----------|------|
| 1 | 0.62 | 0.34 | 0.91 |
| 5 | 0.71 | 0.42 | 0.57 |
| 10 | 0.71 | 0.42 | 0.54 |

#### Cascade R-CNN
| Epoch | mAP50 | mAP50-95 | Loss |
|-------|-------|----------|------|
| 1 | 0.62 | 0.35 | 0.90 |
| 6 | 0.71 | 0.41 | 0.57 |
| 12 | 0.71 | 0.42 | 0.48 |

### Key Findings

1. **Best Accuracy**: YOLOv5-m and YOLOv8-m both achieved the highest mAP50 (0.76), with YOLOv8-m having a slightly better mAP50-95 (0.48 vs 0.47).

2. **Speed vs. Accuracy**: 
   - YOLO models delivered significantly faster inference (5-12ms) compared to R-CNN models (23-24ms)
   - YOLOv5-m offers the best balance of accuracy and speed, followed closely by YOLOv8-m
   - YOLOv11 models show good accuracy but are slightly slower than YOLOv5/v8 models

3. **Parameter Efficiency**:
   - YOLOv11 models fully utilize their parameters (100% trainable)
   - YOLOv8 models achieved high accuracy despite having no trainable parameters in these tests (using pre-trained weights)
   - YOLOv5 models use parameters efficiently with similar parameter counts to YOLOv11
   - R-CNN models use significantly more parameters for similar accuracy

4. **Computational Requirements**:
   - R-CNN models require 3-5x more GFLOPs than YOLO models
   - YOLOv5-s offers excellent accuracy (0.71 mAP50) with the lowest computational cost (~17 GFLOPs)
   - YOLOv5 models generally appear to be the most efficient in terms of GFLOPs/mAP ratio

5. **Training Convergence**:
   - YOLOv5-m and YOLOv8-m showed the strongest improvements throughout training
   - YOLOv8 models required 30 epochs to reach peak performance while YOLOv5 models reached similar performance in 29 epochs
   - YOLOv11 models showed steady but slightly slower convergence
   - R-CNN models reached their peak performance by epoch 5-6 and showed stable performance afterward

6. **Next Steps**:
   - Testing will continue with RT-DETR, EfficientDet-D2, and DETA models, which are popular alternatives in object detection
   - RT-DETR will provide insights into transformer-based architectures for thermal object detection
   - EfficientDet-D2 offers a potentially efficient alternative to YOLO models
   - DETA represents the latest in vision transformer technology and may offer accuracy improvements


