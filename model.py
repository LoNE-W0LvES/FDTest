"""
Two-stage model: Classification + Segmentation
Stage 1: Classify if image is authentic or forged
Stage 2: If forged, segment the manipulated regions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchvision import models


class ForgeryDetectionModel(nn.Module):
    """
    Two-stage model for forgery detection
    - Stage 1: Binary classification (authentic vs forged)
    - Stage 2: Segmentation of forged regions
    """
    
    def __init__(self, 
                 encoder_name='resnet50',
                 encoder_weights='imagenet',
                 num_classes=2,
                 classification_threshold=0.5):
        super().__init__()
        
        self.classification_threshold = classification_threshold
        
        # Stage 1: Classifier backbone
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights
        )
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        encoder_channels = self.encoder.out_channels[-1]
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(encoder_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Stage 2: Segmentation decoder (U-Net style)
        self.segmentation_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None
        )
        
    def forward(self, x, return_features=False):
        """
        Forward pass
        Args:
            x: Input image tensor (B, 3, H, W)
            return_features: If True, return intermediate features
        Returns:
            Dictionary with classification logits and segmentation mask
        """
        # Stage 1: Classification
        features = self.encoder(x)
        pooled = self.global_pool(features[-1])
        class_logits = self.classifier(pooled)
        
        # Stage 2: Segmentation (always compute, but use selectively)
        seg_logits = self.segmentation_model(x)
        
        outputs = {
            'class_logits': class_logits,
            'seg_logits': seg_logits,
        }
        
        if return_features:
            outputs['features'] = features
            
        return outputs
    
    def predict(self, x):
        """
        Prediction mode: classify first, then segment if forged
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
            # Classification
            class_probs = F.softmax(outputs['class_logits'], dim=1)
            is_forged = class_probs[:, 1] > self.classification_threshold
            
            # Segmentation
            seg_probs = torch.sigmoid(outputs['seg_logits'])
            
            # Zero out segmentation for authentic predictions
            seg_probs = seg_probs * is_forged.view(-1, 1, 1, 1).float()
            
            return {
                'class_probs': class_probs,
                'is_forged': is_forged,
                'seg_probs': seg_probs
            }


class EfficientForgeryModel(nn.Module):
    """
    Efficient model using EfficientNet backbone
    """
    
    def __init__(self, 
                 model_name='efficientnet-b3',
                 num_classes=2,
                 classification_threshold=0.5):
        super().__init__()
        
        self.classification_threshold = classification_threshold
        
        # Classifier
        self.segmentation_model = smp.Unet(
            encoder_name=model_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None
        )
        
        # Use encoder for classification
        self.encoder = self.segmentation_model.encoder
        encoder_channels = self.encoder.out_channels[-1]
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(encoder_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Classification
        features = self.encoder(x)
        pooled = self.global_pool(features[-1])
        class_logits = self.classifier(pooled)
        
        # Segmentation
        seg_logits = self.segmentation_model(x)
        
        return {
            'class_logits': class_logits,
            'seg_logits': seg_logits,
        }
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
            class_probs = F.softmax(outputs['class_logits'], dim=1)
            is_forged = class_probs[:, 1] > self.classification_threshold
            
            seg_probs = torch.sigmoid(outputs['seg_logits'])
            seg_probs = seg_probs * is_forged.view(-1, 1, 1, 1).float()
            
            return {
                'class_probs': class_probs,
                'is_forged': is_forged,
                'seg_probs': seg_probs
            }


class CombinedLoss(nn.Module):
    """
    Combined loss for classification and segmentation
    """
    
    def __init__(self, 
                 class_weight=1.0,
                 seg_weight=1.0,
                 dice_weight=0.5,
                 bce_weight=0.5):
        super().__init__()
        
        self.class_weight = class_weight
        self.seg_weight = seg_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        
        self.class_criterion = nn.CrossEntropyLoss()
        self.bce_criterion = nn.BCEWithLogitsLoss()
        
    def dice_loss(self, pred, target, smooth=1e-6):
        """Dice loss for segmentation"""
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        return 1 - dice
    
    def forward(self, outputs, labels, masks):
        """
        Args:
            outputs: Model outputs dict with 'class_logits' and 'seg_logits'
            labels: Classification labels (B,)
            masks: Segmentation masks (B, 1, H, W)
        """
        # Classification loss
        class_loss = self.class_criterion(outputs['class_logits'], labels)
        
        # Segmentation loss (only for forged images)
        forged_mask = (labels == 1).float()
        
        if forged_mask.sum() > 0:
            seg_logits = outputs['seg_logits']
            
            # BCE loss
            bce_loss = self.bce_criterion(seg_logits, masks)
            
            # Dice loss
            dice_loss = self.dice_loss(seg_logits, masks)
            
            # Combined segmentation loss
            seg_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
            
            # Weight by number of forged images
            seg_loss = seg_loss * forged_mask.sum() / len(labels)
        else:
            seg_loss = torch.tensor(0.0, device=labels.device)
        
        # Total loss
        total_loss = self.class_weight * class_loss + self.seg_weight * seg_loss
        
        return {
            'total_loss': total_loss,
            'class_loss': class_loss,
            'seg_loss': seg_loss
        }


def create_model(model_type='unet_resnet50', pretrained=True):
    """
    Factory function to create models
    
    Args:
        model_type: Type of model ('unet_resnet50', 'unet_efficientnet', etc.)
        pretrained: Use pretrained weights
    """
    encoder_weights = 'imagenet' if pretrained else None
    
    if model_type == 'unet_resnet50':
        model = ForgeryDetectionModel(
            encoder_name='resnet50',
            encoder_weights=encoder_weights
        )
    elif model_type == 'unet_resnet34':
        model = ForgeryDetectionModel(
            encoder_name='resnet34',
            encoder_weights=encoder_weights
        )
    elif model_type == 'unet_efficientnet_b3':
        model = EfficientForgeryModel(
            model_name='efficientnet-b3'
        )
    elif model_type == 'unet_efficientnet_b4':
        model = EfficientForgeryModel(
            model_name='efficientnet-b4'
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model
