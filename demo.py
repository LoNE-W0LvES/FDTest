"""
Demo Script - Quick Start Example

This script demonstrates how to use the forgery detection system.
Modify the paths to match your data location.
"""

import sys
from pathlib import Path

def demo_training():
    """Demo: Train models on your data"""
    print("\n" + "="*70)
    print("DEMO 1: TRAINING MODELS")
    print("="*70)
    
    from forgery_detection import main
    
    # Update this path to your data directory
    DATA_PATH = "C:/Users/nafit/Downloads/Compressed/recodai-luc-scientific-image-forgery-detection"
    
    print(f"\nData path: {DATA_PATH}")
    print("\nStarting training...")
    print("This will train both classification and segmentation models.")
    print("Training takes approximately 30-60 minutes depending on your hardware.\n")
    
    # Train models
    classification_model, segmentation_model = main(
        data_path=DATA_PATH,
        batch_size=16,  # Reduce to 8 or 4 if out of memory
        num_epochs=30   # Increase to 50+ for better results
    )
    
    print("\n✓ Training complete!")
    print("  - Models saved to current directory")
    print("  - Training curves saved to training_curves.png")


def demo_inference():
    """Demo: Run inference on test images"""
    print("\n" + "="*70)
    print("DEMO 2: INFERENCE ON TEST IMAGES")
    print("="*70)
    
    from inference import ForgeryDetector
    
    # Check if models exist
    if not Path('classification_model_best.pth').exists():
        print("\n⚠ Error: Models not found!")
        print("Please run demo_training() first to train the models.")
        return
    
    # Initialize detector
    print("\nInitializing detector...")
    detector = ForgeryDetector(
        classification_model_path='classification_model_best.pth',
        segmentation_model_path='segmentation_model_best.pth'
    )
    
    # Update this path to your test images directory
    TEST_PATH = "C:/Users/nafit/Downloads/Compressed/recodai-luc-scientific-image-forgery-detection/test_images"
    
    print(f"\nTest images path: {TEST_PATH}")
    
    # Check if test path exists
    if not Path(TEST_PATH).exists():
        print(f"\n⚠ Warning: Test path not found: {TEST_PATH}")
        print("Please update TEST_PATH in this script.")
        return
    
    # Run batch prediction
    print("\nRunning predictions on test set...")
    predictions = detector.predict_batch(
        image_paths=TEST_PATH,
        output_csv='submission.csv',
        classification_threshold=0.5,
        segmentation_threshold=0.5
    )
    
    print("\n✓ Inference complete!")
    print("  - Submission saved to submission.csv")
    print(f"  - Processed {len(predictions)} images")
    print(f"  - Detected {(predictions['annotation'] != 'authentic').sum()} forged images")


def demo_single_prediction():
    """Demo: Predict and visualize a single image"""
    print("\n" + "="*70)
    print("DEMO 3: SINGLE IMAGE PREDICTION WITH VISUALIZATION")
    print("="*70)
    
    from inference import ForgeryDetector
    
    # Check if models exist
    if not Path('classification_model_best.pth').exists():
        print("\n⚠ Error: Models not found!")
        print("Please run demo_training() first to train the models.")
        return
    
    # Initialize detector
    print("\nInitializing detector...")
    detector = ForgeryDetector(
        classification_model_path='classification_model_best.pth',
        segmentation_model_path='segmentation_model_best.pth'
    )
    
    # Example image path - update this to any image you want to test
    test_image = "C:/Users/nafit/Downloads/Compressed/recodai-luc-scientific-image-forgery-detection/train_images/forged/90.png"
    
    if not Path(test_image).exists():
        print(f"\n⚠ Warning: Test image not found: {test_image}")
        print("Please update test_image path in this script.")
        return
    
    print(f"\nAnalyzing image: {test_image}")
    
    # Run prediction with visualization
    result = detector.visualize_prediction(
        test_image,
        save_path='demo_visualization.png'
    )
    
    print("\n✓ Prediction complete!")
    print("  - Visualization saved to demo_visualization.png")
    
    # Display detailed results
    print("\nDetailed Results:")
    print(f"  - Is Forged: {result['is_forged']}")
    print(f"  - Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
    
    if result['is_forged']:
        print(f"  - Forged regions detected")
        print(f"  - Mask shape: {result['mask'].shape}")
        print(f"  - Affected pixels: {result['mask'].sum()} / {result['mask'].size}")
        print(f"  - RLE encoding length: {len(result['rle'])} characters")


def demo_batch_with_analysis():
    """Demo: Batch prediction with detailed analysis"""
    print("\n" + "="*70)
    print("DEMO 4: BATCH PREDICTION WITH ANALYSIS")
    print("="*70)
    
    import pandas as pd
    from inference import ForgeryDetector
    
    # Check if models exist
    if not Path('classification_model_best.pth').exists():
        print("\n⚠ Error: Models not found!")
        print("Please run demo_training() first to train the models.")
        return
    
    # Initialize detector
    detector = ForgeryDetector(
        classification_model_path='classification_model_best.pth',
        segmentation_model_path='segmentation_model_best.pth'
    )
    
    # Predict on test set
    TEST_PATH = "C:/Users/nafit/Downloads/Compressed/recodai-luc-scientific-image-forgery-detection/test_images"
    
    if not Path(TEST_PATH).exists():
        print(f"\n⚠ Warning: Test path not found: {TEST_PATH}")
        return
    
    print("\nRunning predictions...")
    predictions = detector.predict_batch(
        image_paths=TEST_PATH,
        output_csv='submission_analyzed.csv',
        classification_threshold=0.5,
        segmentation_threshold=0.5
    )
    
    # Detailed analysis
    print("\n" + "="*70)
    print("DETAILED ANALYSIS")
    print("="*70)
    
    total = len(predictions)
    forged = (predictions['annotation'] != 'authentic').sum()
    authentic = total - forged
    
    print(f"\nDataset Statistics:")
    print(f"  - Total images: {total}")
    print(f"  - Authentic: {authentic} ({authentic/total*100:.1f}%)")
    print(f"  - Forged: {forged} ({forged/total*100:.1f}%)")
    
    print(f"\nConfidence Statistics:")
    print(f"  - Mean confidence: {predictions['confidence'].mean():.4f}")
    print(f"  - Min confidence: {predictions['confidence'].min():.4f}")
    print(f"  - Max confidence: {predictions['confidence'].max():.4f}")
    print(f"  - Std confidence: {predictions['confidence'].std():.4f}")
    
    # High confidence predictions
    high_conf_forged = predictions[
        (predictions['annotation'] != 'authentic') & 
        (predictions['confidence'] > 0.9)
    ]
    print(f"\nHigh Confidence Forgeries (>90%):")
    print(f"  - Count: {len(high_conf_forged)}")
    if len(high_conf_forged) > 0:
        print(f"  - Examples: {high_conf_forged['case_id'].head().tolist()}")
    
    # Low confidence predictions (uncertain)
    uncertain = predictions[
        (predictions['confidence'] > 0.4) & 
        (predictions['confidence'] < 0.6)
    ]
    print(f"\nUncertain Predictions (40-60%):")
    print(f"  - Count: {len(uncertain)}")
    if len(uncertain) > 0:
        print(f"  - Examples: {uncertain['case_id'].head().tolist()}")
    
    print("\n✓ Analysis complete!")


def main_menu():
    """Interactive menu for demos"""
    print("\n" + "="*70)
    print("SCIENTIFIC IMAGE FORGERY DETECTION - DEMO SUITE")
    print("="*70)
    print("\nAvailable demos:")
    print("  1. Train models on your data")
    print("  2. Run inference on test images")
    print("  3. Single image prediction with visualization")
    print("  4. Batch prediction with detailed analysis")
    print("  5. Run all demos")
    print("  0. Exit")
    
    while True:
        try:
            choice = input("\nSelect demo (0-5): ").strip()
            
            if choice == '0':
                print("\nExiting...")
                break
            elif choice == '1':
                demo_training()
            elif choice == '2':
                demo_inference()
            elif choice == '3':
                demo_single_prediction()
            elif choice == '4':
                demo_batch_with_analysis()
            elif choice == '5':
                print("\nRunning all demos...")
                demo_training()
                demo_inference()
                demo_single_prediction()
                demo_batch_with_analysis()
            else:
                print("Invalid choice. Please select 0-5.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\n⚠ Error: {e}")
            print("Please check your paths and try again.")


if __name__ == "__main__":
    # Check if running directly or with command line argument
    if len(sys.argv) > 1:
        demo_name = sys.argv[1].lower()
        
        if demo_name == 'train':
            demo_training()
        elif demo_name == 'infer':
            demo_inference()
        elif demo_name == 'single':
            demo_single_prediction()
        elif demo_name == 'analyze':
            demo_batch_with_analysis()
        elif demo_name == 'all':
            demo_training()
            demo_inference()
            demo_single_prediction()
            demo_batch_with_analysis()
        else:
            print(f"Unknown demo: {demo_name}")
            print("Usage: python demo.py [train|infer|single|analyze|all]")
    else:
        # Interactive menu
        main_menu()
