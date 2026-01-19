#!/usr/bin/env python3
"""Test GPU availability and device detection."""

import torch

print("=" * 60)
print("GPU Detection Test")
print("=" * 60)

# Check CUDA availability
print(f"\nCUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

    # Memory info
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Total GPU memory: {total_memory:.2f} GB")

    # Test tensor creation on GPU
    print("\nTesting tensor creation on GPU...")
    device = torch.device('cuda')
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    z = torch.matmul(x, y)
    print(f"✓ Successfully created and multiplied tensors on GPU")
    print(f"  Result shape: {z.shape}, Device: {z.device}")

else:
    print("No GPU available, will use CPU")
    device = torch.device('cpu')
    print(f"Device: {device}")

print("\n" + "=" * 60)
print("Auto-detection test")
print("=" * 60)
auto_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Auto-detected device: {auto_device}")

print("\n✓ GPU detection test completed!")
