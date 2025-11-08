"""
Dataset loader for Scientific Image Forgery Detection
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class ForgeryDataset(Dataset):
    """Dataset for loading authentic and forged scientific images"""
    
    def __init__(self, data_dir, mode='train', transform=None, img_size=(512, 512)):
        """
        Args:
            data_dir: Root directory containing train_images and train_masks folders
            mode: 'train' or 'val'
            transform: Albumentations transform pipeline
            img_size: Target image size (H, W)
        """
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.img_size = img_size
        
        # Paths
        self.authentic_dir = os.path.join(data_dir, 'train_images', 'authentic')
        self.forged_dir = os.path.join(data_dir, 'train_images', 'forged')
        self.mask_dir = os.path.join(data_dir, 'train_masks')
        
        # Load image paths
        self.samples = []
        self._load_samples()
        
    def _load_samples(self):
        """Load all image paths and labels"""
        # Authentic images (label=0, no mask)
        if os.path.exists(self.authentic_dir):
            authentic_files = [f for f in os.listdir(self.authentic_dir) if f.endswith('.png')]
            for img_file in authentic_files:
                self.samples.append({
                    'image_path': os.path.join(self.authentic_dir, img_file),
                    'label': 0,  # authentic
                    'mask_path': None,
                    'image_id': img_file.replace('.png', '')
                })
        
        # Forged images (label=1, with mask)
        if os.path.exists(self.forged_dir):
            forged_files = [f for f in os.listdir(self.forged_dir) if f.endswith('.png')]
            for img_file in forged_files:
                img_id = img_file.replace('.png', '')
                mask_path = os.path.join(self.mask_dir, f"{img_id}.npy")
                
                # Only add if mask exists
                if os.path.exists(mask_path):
                    self.samples.append({
                        'image_path': os.path.join(self.forged_dir, img_file),
                        'label': 1,  # forged
                        'mask_path': mask_path,
                        'image_id': img_id
                    })
        
        print(f"Loaded {len(self.samples)} samples")
        authentic_count = sum(1 for s in self.samples if s['label'] == 0)
        forged_count = sum(1 for s in self.samples if s['label'] == 1)
        print(f"  Authentic: {authentic_count}, Forged: {forged_count}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load or create mask
        if sample['mask_path'] is not None:
            mask = np.load(sample['mask_path'])
            # Handle multiple masks (take first one or combine)
            if mask.ndim == 3:
                mask = mask[0]  # Take first mask
            mask = mask.astype(np.float32)
        else:
            # Empty mask for authentic images
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return {
            'image': image,
            'mask': mask.unsqueeze(0) if mask.dim() == 2 else mask,
            'label': label,
            'image_id': sample['image_id']
        }


class TestDataset(Dataset):
    """Dataset for test images"""
    
    def __init__(self, test_dir, transform=None, img_size=(512, 512)):
        self.test_dir = test_dir
        self.transform = transform
        self.img_size = img_size
        
        self.image_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.png')])
        print(f"Loaded {len(self.image_files)} test images")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.test_dir, img_file)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return {
            'image': image,
            'image_id': img_file.replace('.png', ''),
            'original_size': original_size
        }


def get_train_transforms(img_size=(512, 512)):
    """Get training augmentation pipeline"""
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.GaussNoise(p=1.0),  # Fixed: removed var_limit parameter
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
        ], p=0.3),
        A.Affine(  # Fixed: using Affine instead of ShiftScaleRotate
            scale=(0.9, 1.1),
            translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
            rotate=(-15, 15),
            p=0.5
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_valid_transforms(img_size=(512, 512)):
    """Get validation transformation pipeline"""
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def create_data_loaders(data_dir, batch_size=8, num_workers=4, val_split=0.15, img_size=(512, 512), device='cuda'):
    """Create train and validation data loaders"""
    
    # Adjust settings for CPU vs GPU
    if device == 'cpu':
        # CPU: Use fewer workers and disable pin_memory
        num_workers = min(num_workers, 2)  # Max 2 workers on CPU
        pin_memory = False
        print(f"CPU mode: Using {num_workers} workers, pin_memory=False")
    else:
        # GPU: Use more workers and enable pin_memory
        pin_memory = True
        print(f"GPU mode: Using {num_workers} workers, pin_memory=True")
    
    # Create full dataset
    full_dataset = ForgeryDataset(
        data_dir=data_dir,
        mode='train',
        transform=None,
        img_size=img_size
    )
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply transforms
    train_dataset.dataset.transform = get_train_transforms(img_size)
    val_dataset.dataset.transform = get_valid_transforms(img_size)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


def create_test_loader(test_dir, batch_size=8, num_workers=4, img_size=(512, 512), device='cuda'):
    """Create test data loader"""
    
    # Adjust settings for CPU vs GPU
    if device == 'cpu':
        num_workers = min(num_workers, 2)
        pin_memory = False
    else:
        pin_memory = True
    
    test_dataset = TestDataset(
        test_dir=test_dir,
        transform=get_valid_transforms(img_size),
        img_size=img_size
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return test_loader
