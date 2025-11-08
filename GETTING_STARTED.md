# Getting Started with Scientific Image Forgery Detection

## Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Organize Your Data
Place your data in this structure:
```
your_data_folder/
├── train_images/
│   ├── authentic/    # Images without forgery
│   └── forged/       # Images with forgery
├── train_masks/      # .npy files (only for forged images)
└── test_images/      # Test images (optional)
```

### Step 3: Train the Model
```bash
# Easy way (interactive):
python run.py /path/to/your_data_folder

# Or with custom settings:
python train.py --data_dir /path/to/your_data_folder --epochs 30 --batch_size 8
```

That's it! The model will train and save to `checkpoints/best_model.pth`.

---

## What You Get

After training completes, you'll have:

1. **`checkpoints/best_model.pth`** - Your trained model
2. **`checkpoints/training_history.png`** - Training curves showing performance
3. **`checkpoints/last_model.pth`** - Latest checkpoint

---

## Making Predictions

### On Test Images
```bash
python inference.py \
    --model_path checkpoints/best_model.pth \
    --test_dir /path/to/test_images \
    --output_path submission.csv
```

This creates `submission.csv` with predictions in competition format.

---

## File Guide

| File | Purpose |
|------|---------|
| `run.py` | **START HERE** - Easy interactive training |
| `train.py` | Main training script |
| `inference.py` | Prediction and submission generation |
| `model.py` | Model architectures |
| `dataset.py` | Data loading and augmentation |
| `requirements.txt` | Python dependencies |
| `README.md` | Full documentation |

---

## Common Issues & Solutions

### Out of Memory Error
**Solution:** Reduce batch size or image size
```bash
python train.py --batch_size 4 --img_size 384 384
```

### "No module named 'segmentation_models_pytorch'"
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### CUDA not available
**Solution:** Training will use CPU (slower but works)

### Low accuracy
**Solutions:**
- Train for more epochs (--epochs 50)
- Try different model (--model_type unet_efficientnet_b3)
- Check your data labels are correct

---

## Example Workflow

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train (30 epochs, ~1-2 hours on GPU)
python run.py C:\Users\nafit\Downloads\Compressed\recodai-luc-scientific-image-forgery-detection

# 3. Generate predictions
python inference.py \
    --model_path checkpoints/best_model.pth \
    --test_dir C:\Users\nafit\Downloads\Compressed\recodai-luc-scientific-image-forgery-detection\test_images \
    --output_path submission.csv
```

Good luck! 🚀
