import os

_DEFAULT_OCEAN = "/ocean/projects/cis240116p/bhoomire/.cache/huggingface"
CACHE_DIR = (
    os.environ.get("HF_HOME")
    or os.environ.get("OCEAN_CACHE")
    or (_DEFAULT_OCEAN if os.path.isdir("/ocean/projects/cis240116p") else None)
    or os.path.expanduser("~/.cache/huggingface")
)
os.environ.setdefault("HF_HOME", CACHE_DIR)

from huggingface_hub import snapshot_download

MODELS_TO_DOWNLOAD = [
    # Llama-3.1-8B AWQ-4bit (re-download fresh)
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    # Llama-3.1-8B AQLM-2bit (re-download fresh)
    "ISTA-DASLab/Meta-Llama-3.1-8B-Instruct-AQLM-PV-2Bit-1x16-hf",
    # Mistral-7B-v0.2 FP32 base
    "mistralai/Mistral-7B-Instruct-v0.2",
    # Mistral-7B-v0.2 GPTQ-4bit
    "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
    # Mistral-7B-v0.2 AWQ-4bit
    "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
    # Mistral-7B-v0.2 AQLM-2bit
    "ISTA-DASLab/Mistral-7B-Instruct-v0.2-AQLM-2Bit-2x8",
    # Qwen2.5-7B AWQ-4bit
    "Qwen/Qwen2.5-7B-Instruct-AWQ",
]

def main():
    for model_id in MODELS_TO_DOWNLOAD:
        print(f"\n{'='*60}")
        print(f"  Downloading: {model_id}")
        print(f"{'='*60}")
        try:
            path = snapshot_download(
                model_id,
                cache_dir=CACHE_DIR,
            )
            print(f"  OK Downloaded to: {path}")
        except Exception as e:
            print(f"  FAIL Failed: {e}")

    print(f"\n{'='*60}")
    print("  All downloads complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
