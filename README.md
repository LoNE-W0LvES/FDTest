# Scientific Image Forgery Detection

A deep learning solution for detecting and segmenting forged regions in scientific images. This implementation uses a two-stage approach:
1. **Classification**: Determine if an image is authentic or forged
2. **Segmentation**: If forged, identify the manipulated regions

## Model Architecture

The solution uses a **U-Net based architecture** with the following components:

- **Encoder**: Pre-trained ResNet50/EfficientNet backbone for feature extraction
- **Classification Head**: Global average pooling + fully connected layers for binary classification
- **Segmentation Decoder**: U-Net decoder for pixel-wise forgery localization
- **Combined Loss**: Weighted combination of classification loss (CrossEntropy) and segmentation loss (BCE + Dice)

## Features

✓ Two-stage detection (classify then segment)  
✓ Multiple backbone options (ResNet, EfficientNet)  
✓ Advanced data augmentation  
✓ Mixed precision training support  
✓ RLE encoding for competition submission  
✓ Visualization tools  
✓ Comprehensive metrics tracking  

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Dataset Structure

Your data directory should be organized as follows:

```
data/
├── train_images/
│   ├── authentic/
│   │   ├── 10.png
│   │   ├── 57.png
│   │   └── ...
│   └── forged/
│       ├── 90.png
│       └── ...
├── train_masks/
│   ├── 10.npy
│   ├── 90.npy
│   └── ...
└── test_images/
    ├── 45.png
    └── ...
```

**Notes:**
- Authentic images have no corresponding masks
- Forged images have masks in `.npy` format (shape: [1, H, W] or [H, W])
- Masks are binary (0 = authentic, 1 = forged region)

## Usage

### 1. Training

Train a model on your data:

```bash
python train.py \
    --data_dir /path/to/data \
    --output_dir checkpoints \
    --model_type unet_resnet50 \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --img_size 512 512 \
    --num_workers 4 \
    --val_split 0.15
```

**Training Arguments:**
- `--data_dir`: Path to your data directory
- `--output_dir`: Where to save model checkpoints
- `--model_type`: Model architecture (options: unet_resnet50, unet_resnet34, unet_efficientnet_b3, unet_efficientnet_b4)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size (reduce if GPU memory issues)
- `--lr`: Learning rate
- `--img_size`: Input image size (H W)
- `--num_workers`: Number of data loading workers
- `--val_split`: Validation split ratio

**Training Output:**
- `best_model.pth`: Model with best validation F1 score
- `last_model.pth`: Model from last epoch
- `training_history.png`: Training curves

### 2. Inference

Generate predictions on test images:

```bash
python inference.py \
    --model_path checkpoints/best_model.pth \
    --test_dir /path/to/test_images \
    --output_path submission.csv \
    --model_type unet_resnet50 \
    --batch_size 8 \
    --img_size 512 512 \
    --class_threshold 0.5 \
    --seg_threshold 0.5
```

**Inference Arguments:**
- `--model_path`: Path to trained model checkpoint
- `--test_dir`: Directory containing test images
- `--output_path`: Output CSV file path
- `--model_type`: Must match training model type
- `--batch_size`: Batch size for inference
- `--img_size`: Image size (must match training)
- `--class_threshold`: Threshold for classification (0-1)
- `--seg_threshold`: Threshold for segmentation (0-1)

**Output Format:**
```csv
case_id,annotation
45,authentic
90,1 150 5 300 ...
```

### 3. Visualization

Visualize predictions on sample images:

```python
from inference import visualize_predictions

image_paths = [
    '/path/to/image1.png',
    '/path/to/image2.png',
]

visualize_predictions(
    model_path='checkpoints/best_model.pth',
    image_paths=image_paths,
    output_dir='visualizations',
    model_type='unet_resnet50',
    img_size=(512, 512),
    classification_threshold=0.5,
    segmentation_threshold=0.5
)
```

This creates visualizations showing:
- Original image
- Predicted mask
- Overlay of mask on image

## Python API Usage

### Training from Python

```python
from train import train_model

model, history = train_model(
    data_dir='/path/to/data',
    output_dir='checkpoints',
    model_type='unet_resnet50',
    epochs=50,
    batch_size=8,
    learning_rate=1e-4,
    img_size=(512, 512),
    device='cuda',
    num_workers=4,
    val_split=0.15
)
```

### Making Predictions

```python
from inference import ForgeryPredictor, generate_submission
from model import create_model
import torch

# Load model
device = torch.device('cuda')
checkpoint = torch.load('checkpoints/best_model.pth')
model = create_model('unet_resnet50', pretrained=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

# Create predictor
predictor = ForgeryPredictor(
    model=model,
    device=device,
    classification_threshold=0.5,
    segmentation_threshold=0.5
)

# Predict single image
from dataset import get_valid_transforms
transform = get_valid_transforms((512, 512))
result = predictor.predict_image(
    image_path='/path/to/image.png',
    transform=transform
)

print(f"Is forged: {result['is_forged']}")
print(f"Confidence: {result['class_prob'][1]:.3f}")

# Or generate full submission
generate_submission(
    model_path='checkpoints/best_model.pth',
    test_dir='/path/to/test_images',
    output_path='submission.csv',
    model_type='unet_resnet50'
)
```

## Model Performance

The model tracks multiple metrics:

**Classification Metrics:**
- Accuracy
- Precision, Recall, F1 score
- Classification loss

**Segmentation Metrics:**
- Dice coefficient
- IoU (Intersection over Union)
- Segmentation loss (BCE + Dice)

## Tips for Better Results

### 1. Data Augmentation
The default augmentation pipeline includes:
- Horizontal/vertical flips
- Random rotations
- Gaussian noise/blur
- Brightness/contrast adjustments
- Shift/scale/rotate

Modify `get_train_transforms()` in `dataset.py` to customize.

### 2. Model Selection
- **ResNet50**: Good balance of speed and accuracy
- **ResNet34**: Faster, slightly lower accuracy
- **EfficientNet-B3**: Better accuracy, slower
- **EfficientNet-B4**: Best accuracy, slowest

### 3. Hyperparameter Tuning
Key parameters to tune:
- Learning rate: Try 1e-4 to 1e-3
- Batch size: Larger is better (if GPU allows)
- Image size: Larger captures more detail but slower
- Loss weights: Adjust `class_weight` and `seg_weight` in `CombinedLoss`

### 4. Threshold Tuning
After training, tune thresholds on validation set:
- `classification_threshold`: Higher = more conservative (fewer false positives)
- `segmentation_threshold`: Higher = smaller masks

### 5. Post-processing
Consider adding:
- Morphological operations (opening/closing)
- Connected component analysis
- Minimum region size filtering

## File Descriptions

- **`dataset.py`**: Data loading and augmentation
- **`model.py`**: Model architectures and loss functions
- **`train.py`**: Training loop and evaluation
- **`inference.py`**: Prediction and submission generation
- **`requirements.txt`**: Python dependencies

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size
python train.py --batch_size 4

# Or reduce image size
python train.py --img_size 384 384
```

### Low Accuracy
- Train for more epochs
- Try different model architecture
- Adjust data augmentation
- Check data quality and labels

### Slow Training
- Use smaller model (resnet34)
- Reduce image size
- Enable mixed precision training
- Use more workers for data loading

## Competition Submission

1. Train your model
2. Generate predictions on test set
3. Submit the CSV file

The submission format follows competition requirements:
- `case_id`: Image identifier
- `annotation`: Either "authentic" or RLE-encoded mask

## File Descriptions

- **`dataset.py`**: Data loading and augmentation
- **`model.py`**: Model architectures and loss functions
- **`train.py`**: Training loop and evaluation
- **`inference.py`**: Prediction and submission generation
- **`requirements.txt`**: Python dependencies

## Contact

For questions or issues, please open an issue on GitHub.
