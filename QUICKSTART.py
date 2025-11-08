"""
QUICK START GUIDE
=================

Scientific Image Forgery Detection System
"""

# ============================================================================
# STEP 1: INSTALL DEPENDENCIES
# ============================================================================

# Open terminal/command prompt and run:
# pip install torch torchvision numpy pandas Pillow opencv-python scikit-learn tqdm matplotlib

# OR use requirements.txt:
# pip install -r requirements.txt


# ============================================================================
# STEP 2: PREPARE YOUR DATA
# ============================================================================

# Organize your data like this:
"""
your_data_folder/
├── train_images/
│   ├── authentic/
│   │   ├── 10.png
│   │   ├── 57.png
│   │   └── ...
│   └── forged/
│       ├── 90.png
│       ├── 156.png
│       └── ...
├── train_masks/
│   ├── 90.npy
│   ├── 156.npy
│   └── ...
└── test_images/
    ├── 45.png
    └── ...
"""


# ============================================================================
# STEP 3: TRAIN YOUR MODELS
# ============================================================================

# Option A: Using the demo script (easiest)
"""
python demo.py
# Then select option 1 from the menu
"""

# Option B: Direct Python code
"""
from forgery_detection import main

classification_model, segmentation_model = main(
    data_path='C:/Users/nafit/Downloads/Compressed/recodai-luc-scientific-image-forgery-detection',
    batch_size=16,
    num_epochs=30
)
"""

# Option C: Command line with demo script
"""
python demo.py train
"""


# ============================================================================
# STEP 4: MAKE PREDICTIONS
# ============================================================================

# Option A: Create submission for competition
"""
from inference import ForgeryDetector

detector = ForgeryDetector(
    classification_model_path='classification_model_best.pth',
    segmentation_model_path='segmentation_model_best.pth'
)

# Process all test images
predictions = detector.predict_batch(
    image_paths='test_images/',
    output_csv='submission.csv'
)
"""

# Option B: Predict single image with visualization
"""
result = detector.visualize_prediction(
    'test_images/45.png',
    save_path='visualization.png'
)
"""

# Option C: Using demo script
"""
python demo.py infer
"""


# ============================================================================
# COMPLETE EXAMPLE: FROM SCRATCH TO SUBMISSION
# ============================================================================

if __name__ == "__main__":
    
    # 1. Import libraries
    from forgery_detection import main
    from inference import ForgeryDetector
    
    # 2. Set your data path
    DATA_PATH = "C:/Users/nafit/Downloads/Compressed/recodai-luc-scientific-image-forgery-detection"
    
    # 3. Train models (ONLY NEEDED ONCE)
    print("Training models... (this takes 30-60 minutes)")
    classification_model, segmentation_model = main(
        data_path=DATA_PATH,
        batch_size=16,
        num_epochs=30
    )
    print("✓ Training complete!")
    
    # 4. Create detector
    print("\nInitializing detector...")
    detector = ForgeryDetector(
        classification_model_path='classification_model_best.pth',
        segmentation_model_path='segmentation_model_best.pth'
    )
    
    # 5. Make predictions
    print("\nMaking predictions on test set...")
    predictions = detector.predict_batch(
        image_paths=f'{DATA_PATH}/test_images',
        output_csv='submission.csv',
        classification_threshold=0.5,
        segmentation_threshold=0.5
    )
    
    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)
    print("Your submission file 'submission.csv' is ready!")
    print(f"Total predictions: {len(predictions)}")
    print(f"Forged images detected: {(predictions['annotation'] != 'authentic').sum()}")


# ============================================================================
# TIPS FOR BETTER PERFORMANCE
# ============================================================================

"""
1. INCREASE TRAINING EPOCHS
   - Change num_epochs=30 to num_epochs=50 or more
   - Models continue improving with more training

2. ADJUST BATCH SIZE
   - Reduce if you get out-of-memory errors: batch_size=8 or batch_size=4
   - Increase if you have more GPU memory: batch_size=32

3. TUNE THRESHOLDS
   - For classification: Try classification_threshold=0.4 or 0.6
   - For segmentation: Try segmentation_threshold=0.4 or 0.6
   - Optimize on validation set for best results

4. USE ENSEMBLE
   - Train multiple models with different seeds
   - Average their predictions
   - Usually gives 1-3% improvement

5. DATA AUGMENTATION
   - Add more augmentation in training (rotation, flip, color jitter)
   - Helps model generalize better

6. POST-PROCESSING
   - Clean masks with morphological operations
   - Remove small connected components
   - Fill small holes

7. TRAIN LONGER
   - Monitor training curves
   - Stop when validation loss stops improving
   - Use early stopping with patience=10
"""


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
ERROR: "CUDA out of memory"
SOLUTION: Reduce batch_size to 8 or 4

ERROR: "No module named 'torch'"
SOLUTION: Run: pip install torch torchvision

ERROR: "FileNotFoundError"
SOLUTION: Check your data path is correct and files exist

ERROR: Models not learning (loss not decreasing)
SOLUTION: 
  - Check your data is loaded correctly
  - Verify masks match images
  - Try lower learning rate (lr=1e-5)
  - Check for data preprocessing issues

SLOW TRAINING:
SOLUTION:
  - Use GPU if available
  - Reduce image size to 224x224
  - Use mixed precision training (amp)
  - Reduce batch size and use gradient accumulation
"""


# ============================================================================
# FILE DESCRIPTIONS
# ============================================================================

"""
forgery_detection.py
  - Main training script
  - Contains model architectures
  - Training loops for both models
  - Data loading utilities

inference.py
  - Inference and prediction
  - ForgeryDetector class for easy use
  - Batch prediction with CSV output
  - Visualization tools

demo.py
  - Interactive demo suite
  - Quick testing and validation
  - Multiple demo scenarios
  - Command line interface

requirements.txt
  - All required Python packages
  - Install with: pip install -r requirements.txt

README.md
  - Comprehensive documentation
  - Detailed usage examples
  - Advanced techniques
  - Troubleshooting guide
"""


# ============================================================================
# EXPECTED OUTPUT FILES
# ============================================================================

"""
After training:
  - classification_model_best.pth (70-200 MB)
  - segmentation_model_best.pth (100-300 MB)
  - training_curves.png (visualization)

After inference:
  - submission.csv (competition format)
  - visualization.png (if requested)

CSV format:
  case_id,annotation
  45,authentic
  90,1 2 10 4 20 8
  156,authentic
"""


# ============================================================================
# NEED HELP?
# ============================================================================

"""
1. Check README.md for detailed documentation
2. Run demo.py for interactive examples
3. Review training_curves.png to diagnose training issues
4. Adjust hyperparameters based on your data
5. For specific errors, check the Troubleshooting section above
"""

print(__doc__)
