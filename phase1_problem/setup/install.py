"""
SafeQuant Phase 1: Install dependencies and verify environment.

Checks that PyTorch, Transformers, and BitsAndBytes are installed.
Phase 1 runs on CPU only; GPU is optional and not required.

Run: python phase1_problem/setup/install.py
"""
import transformers
import bitsandbytes


def main():
    print("Ensure that you have installed all the dependencies.\n")

    print("Verifying imports...")
    import torch

    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()} (phase1 uses CPU only)")

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("  Device: CPU (no GPU detected)")

    print(f"  Transformers: {transformers.__version__}")
    print(f"  BitsAndBytes: {bitsandbytes.__version__}")

    print("\nSetup complete")


if __name__ == "__main__":
    main()
