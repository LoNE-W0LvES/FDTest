"""
Scientific Image Forgery Detection and Segmentation

This script provides:
1. Classification model to detect forged vs authentic images
2. Segmentation model to locate forged regions
3. Training pipeline for both models
4. Inference pipeline with ensemble predictions
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class ForgeryDataset(Dataset):
    """Dataset for loading authentic and forged images"""
    
    def __init__(self, image_paths, labels, masks=None, transform=None, mode='classification'):
        self.image_paths = image_paths
        self.labels = labels
        self.masks = masks
        self.transform = transform
        self.mode = mode
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        if self.mode == 'segmentation' and self.masks is not None:
            mask_path = self.masks[idx]
            if mask_path and os.path.exists(mask_path):
                # Load mask from .npy file
                mask = np.load(mask_path)
                if len(mask.shape) == 3:
                    mask = mask[0]  # Take first channel if multi-channel
                mask = torch.from_numpy(mask).float()
                # Resize mask to match image size
                mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), 
                                   size=(256, 256), 
                                   mode='nearest').squeeze()
            else:
                # Create empty mask for authentic images
                mask = torch.zeros(256, 256)
            return image, label, mask
        
        return image, label


class ClassificationModel(nn.Module):
    """Binary classification model: authentic vs forged"""
    
    def __init__(self, backbone='efficientnet_b0', pretrained=True):
        super(ClassificationModel, self).__init__()
        
        if backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class UNetSegmentation(nn.Module):
    """U-Net architecture for segmentation of forged regions"""
    
    def __init__(self, in_channels=3, out_channels=1):
        super(UNetSegmentation, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2, 2)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return torch.sigmoid(self.out(dec1))


def load_data(base_path):
    """Load training data from directory structure"""
    
    base_path = Path(base_path)
    train_images_path = base_path / 'train_images'
    train_masks_path = base_path / 'train_masks'
    
    image_paths = []
    labels = []
    mask_paths = []
    
    # Load authentic images
    authentic_path = train_images_path / 'authentic'
    if authentic_path.exists():
        for img_file in authentic_path.glob('*.png'):
            image_paths.append(str(img_file))
            labels.append(0)  # 0 for authentic
            mask_paths.append(None)
    
    # Load forged images
    forged_path = train_images_path / 'forged'
    if forged_path.exists():
        for img_file in forged_path.glob('*.png'):
            image_paths.append(str(img_file))
            labels.append(1)  # 1 for forged
            
            # Find corresponding mask
            mask_file = train_masks_path / f"{img_file.stem}.npy"
            if mask_file.exists():
                mask_paths.append(str(mask_file))
            else:
                mask_paths.append(None)
    
    print(f"Loaded {len(image_paths)} images:")
    print(f"  - Authentic: {labels.count(0)}")
    print(f"  - Forged: {labels.count(1)}")
    
    return image_paths, labels, mask_paths


def train_classification_model(train_loader, val_loader, num_epochs=30, lr=1e-4):
    """Train the classification model"""
    
    print("\n" + "="*60)
    print("TRAINING CLASSIFICATION MODEL")
    print("="*60)
    
    model = ClassificationModel(backbone='efficientnet_b0').to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                           factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({'loss': loss.item(), 'acc': train_correct/train_total})
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device)
                
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(avg_val_loss)
        
        # Calculate AUC
        auc = roc_auc_score(all_labels, all_preds)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, AUC: {auc:.4f}')
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, 'classification_model_best.pth')
            print(f'  ✓ Best model saved!')
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses


def train_segmentation_model(train_loader, val_loader, num_epochs=30, lr=1e-4):
    """Train the segmentation model"""
    
    print("\n" + "="*60)
    print("TRAINING SEGMENTATION MODEL")
    print("="*60)
    
    model = UNetSegmentation().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                           factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Only train on forged images (labels == 1)
            forged_idx = (labels == 1)
            if forged_idx.sum() == 0:
                continue
            
            images = images[forged_idx]
            masks = masks[forged_idx]
            
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_count = 0
        
        with torch.no_grad():
            for images, labels, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                # Only evaluate on forged images
                forged_idx = (labels == 1)
                if forged_idx.sum() == 0:
                    continue
                
                images = images[forged_idx]
                masks = masks[forged_idx]
                
                outputs = model(images).squeeze()
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_count += 1
        
        if val_count > 0:
            avg_val_loss = val_loss / val_count
        else:
            avg_val_loss = 0
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        
        # Learning rate scheduling
        if val_count > 0:
            scheduler.step(avg_val_loss)
        
        # Save best model
        if val_count > 0 and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, 'segmentation_model_best.pth')
            print(f'  ✓ Best model saved!')
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses


def rle_encode(mask):
    """Encode binary mask to RLE format"""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def predict(classification_model, segmentation_model, image_path, threshold=0.5):
    """
    Predict if image is forged and segment forged regions if applicable
    
    Returns:
        is_forged: bool
        mask: numpy array or None
        rle: str or None (RLE encoded mask)
    """
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Classification
    classification_model.eval()
    with torch.no_grad():
        is_forged_prob = classification_model(image_tensor).item()
        is_forged = is_forged_prob > threshold
    
    # Segmentation (only if forged)
    if is_forged:
        segmentation_model.eval()
        with torch.no_grad():
            mask_pred = segmentation_model(image_tensor).squeeze().cpu().numpy()
            mask_binary = (mask_pred > 0.5).astype(np.uint8)
            
            # Resize mask to original image size
            mask_resized = cv2.resize(mask_binary, original_size, 
                                     interpolation=cv2.INTER_NEAREST)
            
            # Encode to RLE
            rle = rle_encode(mask_resized)
            
            return True, mask_resized, rle
    else:
        return False, None, None


def main(data_path, batch_size=16, num_epochs=30):
    """Main training pipeline"""
    
    # Load data
    print("Loading data...")
    image_paths, labels, mask_paths = load_data(data_path)
    
    # Split data
    train_imgs, val_imgs, train_labels, val_labels, train_masks, val_masks = \
        train_test_split(image_paths, labels, mask_paths, 
                        test_size=0.2, random_state=42, stratify=labels)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # ===== TRAIN CLASSIFICATION MODEL =====
    print("\n" + "="*60)
    print("PHASE 1: CLASSIFICATION MODEL TRAINING")
    print("="*60)
    
    train_dataset_cls = ForgeryDataset(train_imgs, train_labels, 
                                       transform=train_transform, 
                                       mode='classification')
    val_dataset_cls = ForgeryDataset(val_imgs, val_labels, 
                                     transform=val_transform, 
                                     mode='classification')
    
    train_loader_cls = DataLoader(train_dataset_cls, batch_size=batch_size, 
                                  shuffle=True, num_workers=2)
    val_loader_cls = DataLoader(val_dataset_cls, batch_size=batch_size, 
                                shuffle=False, num_workers=2)
    
    classification_model, cls_train_loss, cls_val_loss = \
        train_classification_model(train_loader_cls, val_loader_cls, 
                                  num_epochs=num_epochs)
    
    # ===== TRAIN SEGMENTATION MODEL =====
    print("\n" + "="*60)
    print("PHASE 2: SEGMENTATION MODEL TRAINING")
    print("="*60)
    
    train_dataset_seg = ForgeryDataset(train_imgs, train_labels, train_masks,
                                       transform=train_transform, 
                                       mode='segmentation')
    val_dataset_seg = ForgeryDataset(val_imgs, val_labels, val_masks,
                                     transform=val_transform, 
                                     mode='segmentation')
    
    train_loader_seg = DataLoader(train_dataset_seg, batch_size=batch_size, 
                                  shuffle=True, num_workers=2)
    val_loader_seg = DataLoader(val_dataset_seg, batch_size=batch_size, 
                                shuffle=False, num_workers=2)
    
    segmentation_model, seg_train_loss, seg_val_loss = \
        train_segmentation_model(train_loader_seg, val_loader_seg, 
                                num_epochs=num_epochs)
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(cls_train_loss, label='Train Loss')
    axes[0].plot(cls_val_loss, label='Val Loss')
    axes[0].set_title('Classification Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(seg_train_loss, label='Train Loss')
    axes[1].plot(seg_val_loss, label='Val Loss')
    axes[1].set_title('Segmentation Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print(f"\n✓ Training curves saved to 'training_curves.png'")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"✓ Classification model saved: classification_model_best.pth")
    print(f"✓ Segmentation model saved: segmentation_model_best.pth")
    
    return classification_model, segmentation_model


if __name__ == "__main__":
    # Example usage
    DATA_PATH = "/path/to/your/data"  # Update this path
    
    # Train models
    cls_model, seg_model = main(DATA_PATH, batch_size=16, num_epochs=30)
    
    # Example prediction
    # is_forged, mask, rle = predict(cls_model, seg_model, "test_image.png")
    # print(f"Is forged: {is_forged}")
    # if is_forged:
    #     print(f"RLE mask: {rle}")
