"""
GPU Detection and Diagnostic Script
Run this to check if PyTorch can detect your GPU
"""
import torch
import sys

print("="*70)
print("GPU DIAGNOSTIC TOOL")
print("="*70)
print()

# Check PyTorch version
print(f"PyTorch Version: {torch.__version__}")
print()

# Check CUDA availability
print("CUDA Available:", torch.cuda.is_available())
print()

if torch.cuda.is_available():
    # CUDA is available
    print("✓ GPU DETECTED!")
    print("-"*70)
    
    # Number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    print()
    
    # Details for each GPU
    for i in range(num_gpus):
        print(f"GPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multi-Processor Count: {props.multi_processor_count}")
        print()
    
    # Current GPU
    print(f"Current GPU: {torch.cuda.current_device()}")
    print()
    
    # CUDA version
    print(f"CUDA Version: {torch.version.cuda}")
    print()
    
    # cuDNN version
    if torch.backends.cudnn.is_available():
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
    else:
        print("cuDNN: Not available")
    print()
    
    # Test GPU with a simple operation
    print("Testing GPU with simple operation...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✓ GPU computation successful!")
        print(f"  Result shape: {z.shape}")
        print(f"  Result device: {z.device}")
        del x, y, z
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"✗ GPU computation failed: {e}")
    print()
    
    # Memory info
    print("GPU Memory Status:")
    for i in range(num_gpus):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}:")
        print(f"    Allocated: {allocated:.2f} GB")
        print(f"    Reserved: {reserved:.2f} GB")
    
    print()
    print("="*70)
    print("RESULT: Your GPU is working correctly!")
    print("The training script will use GPU automatically.")
    print("="*70)
    
else:
    # CUDA is not available
    print("✗ NO GPU DETECTED")
    print("-"*70)
    print()
    print("Possible reasons:")
    print()
    print("1. NO NVIDIA GPU INSTALLED")
    print("   - Check if you have an NVIDIA GPU in your system")
    print("   - AMD or Intel GPUs are not supported by CUDA")
    print()
    print("2. CUDA DRIVERS NOT INSTALLED")
    print("   - Download from: https://developer.nvidia.com/cuda-downloads")
    print("   - Install CUDA Toolkit 11.7 or 11.8")
    print()
    print("3. PYTORCH INSTALLED WITHOUT CUDA SUPPORT")
    print("   - Current PyTorch may be CPU-only version")
    print("   - Reinstall with:")
    print("     pip uninstall torch torchvision")
    print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print()
    print("4. DRIVER VERSION MISMATCH")
    print("   - Update NVIDIA drivers to latest version")
    print("   - Download from: https://www.nvidia.com/drivers")
    print()
    
    # Check if CUDA is compiled in PyTorch
    print("PyTorch CUDA Compilation:")
    cuda_available_compiled = hasattr(torch, '_C') and hasattr(torch._C, '_cuda_isDriverSufficient')
    if cuda_available_compiled:
        print("  ✓ PyTorch was compiled with CUDA support")
        print("  ✗ But CUDA runtime is not available")
        print("  → Install CUDA drivers")
    else:
        print("  ✗ PyTorch was NOT compiled with CUDA support")
        print("  → Reinstall PyTorch with CUDA support")
    
    print()
    print("="*70)
    print("RESULT: Training will use CPU (very slow)")
    print("="*70)

print()
print("NEXT STEPS:")
print("-"*70)

if torch.cuda.is_available():
    print("✓ You're ready to train with GPU!")
    print()
    print("Run training with:")
    print("  python train.py --data_dir /path/to/data")
    print()
    print("Or for easier setup:")
    print("  python run.py /path/to/data")
else:
    print("To enable GPU support:")
    print()
    print("1. Verify you have NVIDIA GPU:")
    print("   - Windows: Open Device Manager → Display adapters")
    print("   - Linux: Run 'nvidia-smi' in terminal")
    print()
    print("2. Install CUDA Toolkit:")
    print("   https://developer.nvidia.com/cuda-downloads")
    print()
    print("3. Reinstall PyTorch with CUDA:")
    print("   pip uninstall torch torchvision")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print()
    print("4. Run this script again to verify:")
    print("   python check_gpu.py")

print()
print("="*70)
