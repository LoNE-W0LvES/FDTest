"""
Training script for forgery detection model
"""
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
from datetime import datetime

from model import create_model, CombinedLoss
from dataset import create_data_loaders


class MetricTracker:
    """Track training metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_loss = 0
        self.class_loss = 0
        self.seg_loss = 0
        self.class_correct = 0
        self.total = 0
        self.all_preds = []
        self.all_labels = []
        self.dice_scores = []
        
    def update(self, loss_dict, class_preds, labels, seg_preds=None, masks=None):
        batch_size = labels.size(0)
        
        self.total_loss += loss_dict['total_loss'].item() * batch_size
        self.class_loss += loss_dict['class_loss'].item() * batch_size
        self.seg_loss += loss_dict['seg_loss'].item() * batch_size
        
        # Classification metrics
        _, predicted = torch.max(class_preds, 1)
        self.class_correct += (predicted == labels).sum().item()
        self.total += batch_size
        
        self.all_preds.extend(predicted.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
        
        # Segmentation metrics (Dice score for forged images)
        if seg_preds is not None and masks is not None:
            forged_indices = (labels == 1).cpu().numpy()
            if forged_indices.sum() > 0:
                seg_preds_np = (torch.sigmoid(seg_preds) > 0.5).cpu().numpy()
                masks_np = masks.cpu().numpy()
                
                for i in np.where(forged_indices)[0]:
                    pred = seg_preds_np[i]
                    mask = masks_np[i]
                    dice = self._dice_coefficient(pred, mask)
                    self.dice_scores.append(dice)
    
    def _dice_coefficient(self, pred, target, smooth=1e-6):
        pred = pred.flatten()
        target = target.flatten()
        intersection = (pred * target).sum()
        return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    def get_metrics(self):
        metrics = {
            'loss': self.total_loss / self.total,
            'class_loss': self.class_loss / self.total,
            'seg_loss': self.seg_loss / self.total,
            'accuracy': self.class_correct / self.total,
        }
        
        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.all_labels, self.all_preds, average='binary', zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
        # Dice score
        if len(self.dice_scores) > 0:
            metrics['dice'] = np.mean(self.dice_scores)
        else:
            metrics['dice'] = 0.0
        
        return metrics


def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None):
    """Train for one epoch"""
    model.train()
    metrics = MetricTracker()
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss_dict = criterion(outputs, labels, masks)
        loss = loss_dict['total_loss']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if scheduler is not None and isinstance(scheduler, OneCycleLR):
            scheduler.step()
        
        # Update metrics
        metrics.update(
            loss_dict, 
            outputs['class_logits'], 
            labels,
            outputs['seg_logits'],
            masks
        )
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{metrics.class_correct / metrics.total:.4f}"
        })
    
    return metrics.get_metrics()


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    metrics = MetricTracker()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss_dict = criterion(outputs, labels, masks)
            
            # Update metrics
            metrics.update(
                loss_dict,
                outputs['class_logits'],
                labels,
                outputs['seg_logits'],
                masks
            )
            
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'acc': f"{metrics.class_correct / metrics.total:.4f}"
            })
    
    return metrics.get_metrics()


def train_model(
    data_dir,
    output_dir='checkpoints',
    model_type='unet_resnet50',
    epochs=50,
    batch_size=8,
    learning_rate=1e-4,
    img_size=(512, 512),
    device='cuda',
    num_workers=4,
    val_split=0.15,
    save_best_only=True
):
    """
    Main training function
    
    Args:
        data_dir: Path to data directory containing train_images and train_masks
        output_dir: Directory to save checkpoints
        model_type: Type of model to train
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        img_size: Image size (H, W)
        device: Device to train on
        num_workers: Number of data loader workers
        val_split: Validation split ratio
        save_best_only: Only save best model
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device and check for GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        num_gpus = torch.cuda.device_count()
        print(f"✓ GPU available: Using {num_gpus} GPU(s)")
        for i in range(num_gpus):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("⚠ No GPU found - Using CPU (this will be slow)")
        print("  For faster training, install CUDA and PyTorch with GPU support:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    print(f"\nDevice: {device}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
        img_size=img_size,
        device=device.type  # Pass device type to adjust settings
    )
    
    # Create model
    print(f"Creating model: {model_type}")
    model = create_model(model_type=model_type, pretrained=True)
    
    # Use DataParallel for multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"✓ Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = CombinedLoss(
        class_weight=1.0,
        seg_weight=2.0,
        dice_weight=0.5,
        bce_weight=0.5
    )
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Training loop
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_f1': [],
        'val_f1': [],
        'train_dice': [],
        'val_dice': []
    }
    
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        if not isinstance(scheduler, OneCycleLR):
            scheduler.step()
        
        # Log metrics
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_dice'].append(train_metrics['dice'])
        history['val_dice'].append(val_metrics['dice'])
        
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}, "
              f"Dice: {train_metrics['dice']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, "
              f"Dice: {val_metrics['dice']:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}\n")
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_val_loss = val_metrics['loss']
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_f1': val_metrics['f1'],
                'val_dice': val_metrics['dice'],
                'model_type': model_type,
            }
            
            torch.save(checkpoint, os.path.join(output_dir, 'best_model.pth'))
            print(f"✓ Saved best model (F1: {best_val_f1:.4f})")
        
        # Save last model
        if not save_best_only:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_f1': val_metrics['f1'],
                'model_type': model_type,
            }
            torch.save(checkpoint, os.path.join(output_dir, 'last_model.pth'))
        
        print("-" * 50)
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*50 + "\n")
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    return model, history


def plot_training_history(history, output_dir):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 Score
    axes[1, 0].plot(history['train_f1'], label='Train')
    axes[1, 0].plot(history['val_f1'], label='Validation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Dice Score
    axes[1, 1].plot(history['train_dice'], label='Train')
    axes[1, 1].plot(history['val_dice'], label='Validation')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Dice Score')
    axes[1, 1].set_title('Dice Score (Segmentation)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plot saved to {output_dir}/training_history.png")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train forgery detection model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--model_type', type=str, default='unet_resnet50',
                        choices=['unet_resnet50', 'unet_resnet34', 
                                'unet_efficientnet_b3', 'unet_efficientnet_b4'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--img_size', type=int, nargs=2, default=[512, 512],
                        help='Image size (H W)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Validation split ratio')
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        img_size=tuple(args.img_size),
        num_workers=args.num_workers,
        val_split=args.val_split
    )
