# verify_setup.py
import torch
import platform
import psutil
import os
from pathlib import Path

def verify_system():
    print("System Configuration:")
    print(f"macOS Version: {platform.mac_ver()[0]}")
    print(f"Python Version: {platform.python_version()}")
    print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    
    print("\nPyTorch Configuration:")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print(f"MPS Built: {torch.backends.mps.is_built()}")
    
    print("\nOllama Configuration:")
    ollama_config = Path.home() / ".ollama/config.json"
    if ollama_config.exists():
        print("Ollama config found ✓")
    else:
        print("Warning: Ollama config not found")
    
    print("\nEnvironment Variables:")
    mps_ratio = os.getenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "Not Set")
    mps_fallback = os.getenv("PYTORCH_ENABLE_MPS_FALLBACK", "Not Set")
    print(f"MPS Watermark Ratio: {mps_ratio}")
    print(f"MPS Fallback: {mps_fallback}")

if __name__ == "__main__":
    verify_system()