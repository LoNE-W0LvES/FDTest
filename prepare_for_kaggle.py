"""
Alternative: Prepare checkpoint for Kaggle (no TorchScript needed)
This is simpler and more reliable
"""
import os
import torch
from model import create_model

# Load your trained model
print("Loading model...")
checkpoint = torch.load('checkpoints/best_model.pth', weights_only=False)

# Create clean checkpoint with just essentials
clean_checkpoint = {
    'model_state_dict': checkpoint['model_state_dict'],
    'model_type': checkpoint.get('model_type', 'unet_resnet50'),
    'val_f1': checkpoint.get('val_f1'),
    'val_dice': checkpoint.get('val_dice'),
    'epoch': checkpoint.get('epoch'),
}

# Handle DataParallel - remove 'module.' prefix
state_dict = clean_checkpoint['model_state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
    new_state_dict[name] = v

clean_checkpoint['model_state_dict'] = new_state_dict

# Save clean checkpoint
output_path = 'model_kaggle.pth'
torch.save(clean_checkpoint, output_path)

print(f"✓ Clean checkpoint saved to {output_path}")
print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
print(f"Model type: {clean_checkpoint['model_type']}")
print(f"Validation F1: {clean_checkpoint['val_f1']}")

# Test loading
print("\nTesting model loading...")
loaded = torch.load(output_path, weights_only=False)
model = create_model(loaded['model_type'], pretrained=False)
model.load_state_dict(loaded['model_state_dict'])
model.eval()

# Test inference
dummy_input = torch.randn(1, 3, 512, 512)
with torch.no_grad():
    output = model(dummy_input)
    
print("✓ Model loads and runs successfully!")
print(f"Output keys: {output.keys()}")

print("\n" + "="*60)
print("UPLOAD THIS FILE TO KAGGLE:")
print(f"  {output_path}")
print("="*60)
