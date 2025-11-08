"""
Inference script for forgery detection
Generates predictions and creates submission file
"""
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from PIL import Image
import json

from model import create_model
from dataset import create_test_loader, get_valid_transforms


def rle_encode(mask):
    """
    Run-length encoding for binary masks
    Args:
        mask: 2D numpy array with 0s and 1s
    Returns:
        RLE string
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_to_pixel_list(rle_string):
    """
    Convert RLE encoded string to list of pixel indices
    Args:
        rle_string: RLE encoded string (e.g., "1 100 50 200")
    Returns:
        List of pixel indices as integers
    """
    if rle_string == 'authentic' or pd.isna(rle_string):
        return []

    try:
        s = str(rle_string).split()
        if not s or len(s) < 2:
            return []

        pixel_list = []
        for i in range(0, len(s), 2):
            if i + 1 < len(s):
                start = int(s[i])
                length = int(s[i + 1])
                # Add pixel indices for this run (RLE is 1-indexed)
                for j in range(length):
                    pixel_list.append(start - 1 + j)

        return pixel_list
    except Exception as e:
        print(f"Error parsing RLE: {str(e)}")
        return []


def rle_decode(rle_string, shape):
    """
    Decode RLE string to binary mask
    Args:
        rle_string: RLE encoded string
        shape: (height, width) of output mask
    Returns:
        2D numpy array
    """
    if rle_string == 'authentic' or pd.isna(rle_string):
        return np.zeros(shape, dtype=np.uint8)
    
    s = rle_string.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1
    return mask.reshape(shape)


class ForgeryPredictor:
    """Wrapper class for making predictions"""
    
    def __init__(self, model, device, classification_threshold=0.5, segmentation_threshold=0.5):
        self.model = model
        self.device = device
        self.classification_threshold = classification_threshold
        self.segmentation_threshold = segmentation_threshold
        self.model.eval()
    
    @torch.no_grad()
    def predict_batch(self, images):
        """
        Predict on a batch of images
        Args:
            images: Tensor of shape (B, 3, H, W)
        Returns:
            Dictionary with predictions
        """
        images = images.to(self.device)
        
        # Forward pass
        outputs = self.model(images)
        
        # Classification
        class_probs = torch.softmax(outputs['class_logits'], dim=1)
        is_forged = class_probs[:, 1] > self.classification_threshold
        
        # Segmentation
        seg_probs = torch.sigmoid(outputs['seg_logits'])
        seg_masks = (seg_probs > self.segmentation_threshold).float()
        
        # Zero out masks for authentic predictions
        seg_masks = seg_masks * is_forged.view(-1, 1, 1, 1).float()
        
        return {
            'class_probs': class_probs.cpu().numpy(),
            'is_forged': is_forged.cpu().numpy(),
            'seg_masks': seg_masks.cpu().numpy(),
            'seg_probs': seg_probs.cpu().numpy()
        }
    
    def predict_image(self, image_path, transform=None):
        """
        Predict on a single image
        Args:
            image_path: Path to image
            transform: Transformation to apply
        Returns:
            Dictionary with predictions
        """
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]
        
        # Apply transform
        if transform:
            transformed = transform(image=image)
            image_tensor = transformed['image'].unsqueeze(0)
        else:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # Predict
        predictions = self.predict_batch(image_tensor)
        
        # Resize mask back to original size
        seg_mask = predictions['seg_masks'][0, 0]
        seg_mask_resized = cv2.resize(
            seg_mask.astype(np.uint8),
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        return {
            'is_forged': predictions['is_forged'][0],
            'class_prob': predictions['class_probs'][0],
            'seg_mask': seg_mask_resized,
            'original_size': original_size
        }


def generate_submission(
    model_path,
    test_dir,
    output_path='submission.csv',
    model_type='unet_resnet50',
    batch_size=8,
    img_size=(512, 512),
    device='cuda',
    classification_threshold=0.5,
    segmentation_threshold=0.5,
    num_workers=4
):
    """
    Generate submission file for the competition
    
    Args:
        model_path: Path to trained model checkpoint
        test_dir: Directory containing test images
        output_path: Path to save submission CSV
        model_type: Type of model
        batch_size: Batch size for inference
        img_size: Image size for model input
        device: Device to run inference on ('cuda' or 'cpu')
        classification_threshold: Threshold for classifying as forged
        segmentation_threshold: Threshold for segmentation mask
        num_workers: Number of data loader workers
    """
    
    # Set device
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        if device == 'cuda':
            print("⚠ CUDA requested but not available, using CPU instead")
        else:
            print("Using CPU")
    
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        model = create_model(model_type=model_type, pretrained=False)
        
        # Handle both DataParallel and regular model state dicts
        state_dict = checkpoint['model_state_dict']
        
        # Try loading directly first
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            # If it fails, try removing 'module.' prefix
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded successfully")
        print(f"Best validation F1: {checkpoint.get('val_f1', 'N/A')}")
        print(f"Best validation Dice: {checkpoint.get('val_dice', 'N/A')}")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
    
    # Create predictor
    predictor = ForgeryPredictor(
        model=model,
        device=device,
        classification_threshold=classification_threshold,
        segmentation_threshold=segmentation_threshold
    )
    
    # Create test loader
    print("Loading test data...")
    try:
        test_loader = create_test_loader(
            test_dir=test_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=img_size,
            device=device.type
        )
    except Exception as e:
        print(f"Error creating test loader: {str(e)}")
        raise
    
    # Generate predictions
    print("Generating predictions...")
    predictions = []
    
    try:
        for batch in tqdm(test_loader):
            images = batch['image']
            image_ids = batch['image_id']
            original_sizes = batch['original_size']
            
            # Predict
            batch_preds = predictor.predict_batch(images)
            
            # Process each image in batch
            for i in range(len(image_ids)):
                image_id = image_ids[i]
                is_forged = batch_preds['is_forged'][i]
                seg_mask = batch_preds['seg_masks'][i, 0]
                original_size = (original_sizes[0][i].item(), original_sizes[1][i].item())
                
                if is_forged:
                    # Resize mask to original size
                    seg_mask_resized = cv2.resize(
                        seg_mask.astype(np.uint8),
                        (original_size[1], original_size[0]),
                        interpolation=cv2.INTER_NEAREST
                    )

                    # Encode to RLE first
                    rle = rle_encode(seg_mask_resized)

                    # Convert RLE to pixel list and then to JSON
                    if rle and rle.strip() != '':
                        pixel_list = rle_to_pixel_list(rle)
                        annotation = json.dumps(pixel_list)
                    else:
                        annotation = json.dumps([])
                else:
                    annotation = json.dumps([])

                predictions.append({
                    'case_id': image_id,
                    'annotation': annotation
                })
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise
    
    # Create submission dataframe
    submission_df = pd.DataFrame(predictions)
    
    # Sort by case_id
    submission_df['case_id_num'] = submission_df['case_id'].astype(int)
    submission_df = submission_df.sort_values('case_id_num').drop('case_id_num', axis=1)
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    
    # Count statistics
    authentic_count = sum(1 for p in predictions if p['annotation'] == '[]')
    forged_count = len(predictions) - authentic_count

    print(f"\n{'='*50}")
    print(f"Submission saved to {output_path}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Authentic (empty array): {authentic_count}")
    print(f"Forged (with pixel coordinates): {forged_count}")
    print(f"{'='*50}\n")
    
    return submission_df


def visualize_predictions(
    model_path,
    image_paths,
    output_dir='visualizations',
    model_type='unet_resnet50',
    img_size=(512, 512),
    device='cuda',
    classification_threshold=0.5,
    segmentation_threshold=0.5
):
    """
    Visualize predictions on sample images
    
    Args:
        model_path: Path to trained model
        image_paths: List of image paths to visualize
        output_dir: Directory to save visualizations
        model_type: Type of model
        img_size: Image size
        device: Device to run on
        classification_threshold: Classification threshold
        segmentation_threshold: Segmentation threshold
    """
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = create_model(model_type=model_type, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Create predictor
    predictor = ForgeryPredictor(
        model=model,
        device=device,
        classification_threshold=classification_threshold,
        segmentation_threshold=segmentation_threshold
    )
    
    # Get transform
    transform = get_valid_transforms(img_size)
    
    # Process each image
    for img_path in tqdm(image_paths, desc='Visualizing'):
        # Predict
        result = predictor.predict_image(img_path, transform)
        
        # Load original image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Segmentation mask
        axes[1].imshow(result['seg_mask'], cmap='hot')
        axes[1].set_title(f"Predicted Mask\n{'FORGED' if result['is_forged'] else 'AUTHENTIC'}")
        axes[1].axis('off')
        
        # Overlay
        overlay = image.copy()
        mask_colored = np.zeros_like(image)
        mask_colored[:, :, 0] = result['seg_mask'] * 255  # Red channel
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay\nConfidence: {result['class_prob'][1]:.3f}")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save
        img_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"pred_{img_name}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate predictions')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Directory containing test images')
    parser.add_argument('--output_path', type=str, default='submission.csv',
                        help='Output submission file path')
    parser.add_argument('--model_type', type=str, default='unet_resnet50',
                        help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--img_size', type=int, nargs=2, default=[512, 512],
                        help='Image size (H W)')
    parser.add_argument('--class_threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--seg_threshold', type=float, default=0.5,
                        help='Segmentation threshold')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    generate_submission(
        model_path=args.model_path,
        test_dir=args.test_dir,
        output_path=args.output_path,
        model_type=args.model_type,
        batch_size=args.batch_size,
        img_size=tuple(args.img_size),
        device=args.device,
        classification_threshold=args.class_threshold,
        segmentation_threshold=args.seg_threshold,
        num_workers=args.num_workers
    )
