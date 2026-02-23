"""
SafeQuant Phase 1: Install all dependencies.
Run: python phase1_problem/setup/install.py
"""
def main():
    print("Ensure that you have installed all the dependencies\n")
    
    # Ensure that the requirements.txt file exists and you have installed all the dependencies
    
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
    
    print("\nSetup complete")


if __name__ == "__main__":
    main()
