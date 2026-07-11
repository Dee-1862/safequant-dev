import urllib.request, os, concurrent.futures
base_url = "https://raw.githubusercontent.com/andyrdt/refusal_direction/main/"
files = [
    "pipeline/run_pipeline.py",
    "pipeline/submodules/generate_directions.py",
    "pipeline/submodules/select_direction.py",
    "pipeline/submodules/evaluate_loss.py",
    "pipeline/submodules/evaluate_jailbreak.py",
    "pipeline/utils/hook_utils.py",
    "pipeline/utils/utils.py",
    "pipeline/model_utils/model_base.py",
    "pipeline/model_utils/llama2_model.py",
    "pipeline/model_utils/llama3_model.py",
    "pipeline/model_utils/model_factory.py",
    "dataset/load_dataset.py"
]

def download_file(file):
    os.makedirs(os.path.join("refusal_direction", os.path.dirname(file)), exist_ok=True)
    req = urllib.request.Request(base_url + file, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            with open(os.path.join("refusal_direction", file), 'wb') as f:
                f.write(response.read())
        print(f"Downloaded {file}")
    except Exception as e:
        print(f"Failed {file}: {e}")

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    executor.map(download_file, files)
