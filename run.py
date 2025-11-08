"""
Quick Start Script for Scientific Image Forgery Detection

This script provides an easy way to train and evaluate the model.
"""
import os
import sys

def main():
    print("="*70)
    print("Scientific Image Forgery Detection - Quick Start")
    print("="*70)
    print()
    
    # Check if data directory is provided
    if len(sys.argv) < 2:
        print("Usage: python run.py <data_directory>")
        print()
        print("Your data directory should contain:")
        print("  - train_images/authentic/")
        print("  - train_images/forged/")
        print("  - train_masks/")
        print("  - test_images/ (optional)")
        print()
        print("Example:")
        print("  python run.py /path/to/data")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    
    # Verify data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Check for required subdirectories
    required_dirs = [
        os.path.join(data_dir, 'train_images', 'authentic'),
        os.path.join(data_dir, 'train_images', 'forged'),
        os.path.join(data_dir, 'train_masks')
    ]
    
    for req_dir in required_dirs:
        if not os.path.exists(req_dir):
            print(f"Warning: Required directory not found: {req_dir}")
    
    print(f"Data directory: {data_dir}")
    print()
    
    # Training configuration
    print("Training Configuration:")
    print("-" * 50)
    config = {
        'model_type': 'unet_resnet50',
        'epochs': 5,
        'batch_size': 8,
        'learning_rate': 1e-2,
        'img_size': (512, 512),
        'val_split': 0.15,
        'output_dir': 'checkpoints'
    }
    
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Ask for confirmation
    response = input("Start training with these settings? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        sys.exit(0)
    
    print()
    print("="*70)
    print("Starting Training...")
    print("="*70)
    print()
    
    # Import and run training
    from train import train_model
    
    try:
        model, history = train_model(
            data_dir=data_dir,
            output_dir=config['output_dir'],
            model_type=config['model_type'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            img_size=config['img_size'],
            device='cuda',
            num_workers=4,
            val_split=config['val_split']
        )
        
        print()
        print("="*70)
        print("Training Completed Successfully!")
        print("="*70)
        print()
        print(f"Best model saved to: {config['output_dir']}/best_model.pth")
        print(f"Training history plot: {config['output_dir']}/training_history.png")
        print()
        
        # Check if test directory exists
        test_dir = os.path.join(data_dir, 'test_images')
        if os.path.exists(test_dir):
            response = input("Generate predictions on test set? (y/n): ")
            if response.lower() == 'y':
                print()
                print("Generating predictions...")
                
                from inference import generate_submission
                
                submission = generate_submission(
                    model_path=os.path.join(config['output_dir'], 'best_model.pth'),
                    test_dir=test_dir,
                    output_path='submission.csv',
                    model_type=config['model_type'],
                    batch_size=config['batch_size'],
                    img_size=config['img_size']
                )
                
                print(f"Submission saved to: submission.csv")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
