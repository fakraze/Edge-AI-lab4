# Edge-AI Lab 4: YOLOv8 Model Pruning for Edge Deployment

This project implements YOLOv8 instance segmentation model pruning using PyTorch-based structured pruning techniques. The goal is to reduce model size and computational complexity while maintaining acceptable performance for edge AI deployment.

## 🎬 Demo & Results

- **📺 Demo Video**: [Watch on YouTube](https://www.youtube.com/watch?v=MsMMYl46mtQ)

## 🎯 Final Approach

> **Important**: The pruning results were not ideal for our deployment requirements. Therefore, our final implementation uses **YOLOv8n-seg + INT8 quantization** as the optimal approach for edge deployment.

> ** Documentation**: For detailed implementation analysis and results, please refer to the report PDF file.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Performance Results](#performance-results)
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## 🎯 Project Overview

This project demonstrates structured pruning of YOLOv8 instance segmentation models using torch-pruning library. The implementation includes:

- **Iterative structured pruning** with gradual sparsity increase
- **Fine-tuning** after each pruning step to recover performance
- **Performance monitoring** with mAP tracking
- **Model export** to ONNX format for deployment

## ✨ Features

- 🔥 **Structured Pruning**: Uses torch-pruning for channel-wise pruning
- 📊 **Performance Tracking**: Monitors mAP changes throughout pruning process
- 🎯 **Configurable**: Flexible pruning parameters via YAML configuration
- 💾 **Model Export**: Exports pruned models to ONNX format
- 📈 **Visualization**: Generates performance graphs and charts
- 🔄 **Iterative Process**: Gradual pruning with fine-tuning steps

## 📈 Performance Results

| Metric | Original | Pruned | Change |
|--------|----------|---------|--------|
| **Model Size** | 3.41M params | 2.13M params | -37.6% |
| **MACs** | 6.36G | 4.62G | -27.3% |
| **Box mAP@0.5:0.95** | 0.364 | 0.275 | -24.5% |
| **Mask mAP@0.5:0.95** | 0.306 | 0.229 | -25.2% |
| **Speed Improvement** | 1x | 1.38x | +37.5% |

*Pruning ratio: 24% with target rate: 30%*

## 🛠️ Environment Setup

### 1. Create Virtual Environment

```bash
python3 -m venv yolov8_2
source yolov8_2/bin/activate
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install pruning and other dependencies
pip install torch-pruning numpy pyyaml
```

### 3. Install Ultralytics (Legacy Version)

```bash
mkdir repo
cd repo
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
git checkout -b legacy-v8.0.114 tags/v8.0.114
python -m pip uninstall ultralytics -y
python -m pip install -e .
```

## 📁 Dataset Preparation

### COCO Dataset Setup

The project uses COCO 2017 dataset for training and validation.

#### Tutorial Reference
- 📖 [Ultralytics COCO Dataset Documentation](https://docs.ultralytics.com/datasets/detect/coco/)

#### Automatic Dataset Download

The COCO dataset will be automatically downloaded during the first training run:

```bash
# This command will train yolo11n.pt and automatically download COCO datasets
yolo detect train data=coco.yaml model=yolo11n.pt epochs=100 imgsz=640
```

> **Note**: The dataset download is approximately 20.1 GB and will be placed in the `../datasets/coco` directory relative to your workspace.

#### Dataset Structure

After download, the dataset structure will be:

```
datasets/
├── coco/                        # COCO 2017 dataset (20.1 GB)
│   ├── LICENSE                  # Dataset license
│   ├── README.txt              # Dataset information
│   ├── images/                 # Image files
│   │   ├── train2017/          # Training images (118,287 images)
│   │   ├── val2017/            # Validation images (5,000 images)
│   │   └── test2017/           # Test images (40,670 images)
│   ├── labels/                 # Label files for segmentation
│   │   ├── train2017/          # Training labels
│   │   └── val2017/            # Validation labels
│   ├── annotations/            # COCO annotation files
│   ├── train2017.txt          # Training image list
│   ├── val2017.txt            # Validation image list
│   └── test-dev2017.txt       # Test image list
├── coco128-seg/               # Smaller subset for testing (optional)
│   ├── LICENSE
│   ├── README.txt
│   ├── images/                # 128 sample images
│   └── labels/                # Corresponding labels
└── coco2017labels-segments.zip  # Segmentation labels archive
```

## 🚀 Usage

### 1. Model Pruning

Run the iterative pruning process:

```bash
source yolov8_2/bin/activate

python yolov8_pruning_origin.py \
    --model yolov8n-seg.pt \
    --cfg pruning_default.yaml \
    --iterative-steps 10 \
    --target-prune-rate 0.3 \
    --max-map-drop 1
```

> **Note**: Additional hyperparameters such as batch size, epochs, and dataset configuration are defined in `pruning_default.yaml`.


### 2. Fine-tuning

Fine-tune the pruned model:

```bash
python finetune_yolov8.py \
    --model runs/segment/step_7_finetune/weights/best.pt \
    --data coco.yaml \
    --epochs 30 \
    --train-batch 32 \
    --val-batch 8 \
    --imgsz 640 \
    --lr 1e-3 \
    --device 0,1 \
    --project finetune \
    --name step_8 \
    --save-period 1 \
    --workers 2
```

### 3. Model Export

Export pruned model to ONNX:

```bash
python output_onnx.py --model path/to/pruned_model.pt --output model_pruned.onnx
```

### Configuration Files

**pruning_default.yaml**: Main pruning configuration
```yaml
data: coco.yaml
batch: 16
val_batch: 8
imgsz: 640
epochs: 4
save_period: 1
name: yolov8n-seg-pruning
```

## 📂 Project Structure

```
lab4/
├── README.md                    # This file
├── yolov8_pruning_origin.py    # Main pruning implementation
├── yolov8_pruning.py           # Alternative pruning script
├── finetune_yolov8.py          # Fine-tuning script
├── output_onnx.py              # ONNX export utility
├── coco.yaml                   # Dataset configuration
├── pruning_default.yaml        # Pruning parameters
├── note.txt                    # Development notes
├── env.txt                     # Environment setup guide
├── yolov8n-seg.pt             # Pre-trained model
├── datasets/                   # Dataset directory
│   ├── coco/                  # COCO dataset
│   └── coco128-seg/           # Smaller test dataset
├── runs/                      # Training/validation results
│   └── segment/               # Segmentation results
├── finetune/                  # Fine-tuning results
├── repo/                      # Ultralytics repository
│   └── ultralytics/           # Modified ultralytics code
└── yolov8_2/                  # Virtual environment
```

## 📊 Monitoring and Evaluation

The project tracks:
- Model size reduction (parameters and MACs)
- Performance metrics (box mAP, mask mAP)
- Inference speed improvements
- Training convergence during fine-tuning

## 📚 References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Torch-Pruning](https://github.com/VainF/Torch-Pruning)

## 📝 License

This project is based on Ultralytics YOLO which uses AGPL-3.0 License.

