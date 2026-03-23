"""
SafeQuant Phase 1 — Install all dependencies.
Run: python phase1_problem/setup/install.py
"""
import subprocess
import sys

PACKAGES = [
    "torch",
    "transformers>=4.40.0",
    "accelerate",
    "bitsandbytes",
    "datasets",
    "huggingface_hub",
    "numpy",
    "matplotlib",
    "seaborn",
    "pandas",
    "tqdm",
    "scikit-learn",
]

def main():
    print("=== Installing SafeQuant dependencies ===\n")
    for pkg in PACKAGES:
        print(f"Installing {pkg}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", pkg,
            "--break-system-packages", "-q"
        ])
    print("\n=== All dependencies installed ===")
    
    # Verify critical imports
    print("\nVerifying imports...")
    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    
    import transformers
    print(f"  Transformers: {transformers.__version__}")
    
    import bitsandbytes
    print(f"  BitsAndBytes: {bitsandbytes.__version__}")
    
    print("\n=== Setup complete ===")

if __name__ == "__main__":
    main()
