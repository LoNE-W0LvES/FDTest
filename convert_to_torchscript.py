"""
Convert trained model to TorchScript for Kaggle submission
Uses torch.jit.trace (more compatible than torch.jit.script)
"""
import os
import torch
from model import create_model

# Load your trained model
print("Loading model...")
checkpoint = torch.load('checkpoints/best_model.pth', weights_only=False)
model = create_model('unet_resnet50', pretrained=False)

# Handle DataParallel if needed
state_dict = checkpoint['model_state_dict']
try:
    model.load_state_dict(state_dict)
except RuntimeError:
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

model.eval()

print("Converting to TorchScript using trace...")
# Create dummy input (batch_size=1, channels=3, height=512, width=512)
dummy_input = torch.randn(1, 3, 512, 512)

# Use trace instead of script (more compatible)
with torch.no_grad():
    model_traced = torch.jit.trace(model, dummy_input)

# Save
output_path = 'model_traced.pt'
torch.jit.save(model_traced, output_path)

print(f"✓ Model saved to {output_path}")
print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

# Test loading
print("\nTesting model loading...")
loaded_model = torch.jit.load(output_path)
loaded_model.eval()

# Test inference
with torch.no_grad():
    test_output = loaded_model(dummy_input)
    
print("✓ Model loads and runs successfully!")
print(f"Output keys: {test_output.keys()}")
print(f"Class logits shape: {test_output['class_logits'].shape}")
print(f"Seg logits shape: {test_output['seg_logits'].shape}")
