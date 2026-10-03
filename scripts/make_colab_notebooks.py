"""Build Google Colab (A100) versions of the heavy benchmark notebooks.

Takes notebooks/fanet_benchmark_<name>.ipynb and rewrites ONLY environment-specific parts
(data location, checkpoint/output dirs, num_workers). Model, split, loss and training logic are untouched.
Usage: python scripts/make_colab_notebooks.py isic_2018
"""
import json, os, sys, uuid, ast

SETUP_MD = """# Colab (A100) runner - {title}
**Runtime -> Change runtime type -> GPU (A100).**
Checkpoints / `benchmark_results.csv` / figures are written to Google Drive (`WORK_DIR`) so a disconnect can be resumed:
just re-run all cells, training continues from `*_resume.pth`.

Kaggle data download needs credentials: add Colab *Secrets* `KAGGLE_USERNAME` + `KAGGLE_KEY` (recommended) or upload `kaggle.json` when asked."""

SETUP_CODE = '''# ================================================================
# [Colab Setup]  Drive, GPU, deps, dataset download
# ================================================================
import os, sys, subprocess, shutil, json, zipfile
from google.colab import drive
drive.mount('/content/drive')

WORK_DIR = "/content/drive/MyDrive/FANet_runs/{slug}"
os.makedirs(WORK_DIR, exist_ok=True)
DATA_ROOT = "/content/data/{slug}"        # local SSD = fast IO (data is NOT kept on Drive)
os.makedirs(DATA_ROOT, exist_ok=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "albumentations", "tifffile", "opencv-python-headless"], check=True)

import torch
print("CUDA:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
torch.backends.cudnn.benchmark = True        # fixed input size -> faster convs on A100
NUM_WORKERS = min(8, os.cpu_count() or 2)

# ---- Kaggle credentials ----
try:
    from google.colab import userdata
    os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
except Exception:
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        from google.colab import files
        up = files.upload()                      # upload kaggle.json
        os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
        shutil.move(list(up.keys())[0], os.path.expanduser("~/.kaggle/kaggle.json"))
        os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

# ---- Dataset download (skipped if already present) ----
if not any(True for _ in os.scandir(DATA_ROOT)):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", "{kaggle_ds}", "-p", DATA_ROOT, "--unzip"], check=True)
print("Data ready:", DATA_ROOT, "| entries:", len(os.listdir(DATA_ROOT)))
'''

CONFIGS = {
    "isic_2018": dict(
        title="ISIC-2018 (512x512, 100 epochs)", slug="isic2018",
        kaggle_ds="tschandl/isic2018-challenge-task1-data-segmentation",
        data_dir_old='DATA_DIR = "/kaggle/input/datasets/tschandl/isic2018-challenge-task1-data-segmentation"',
    ),
}

def src(c): return ''.join(c['source'])
def set_src(c, s): c['source'] = s.splitlines(keepends=True)

def replace_once(nb, old, new):
    n = 0
    for c in nb['cells']:
        if c['cell_type'] == 'code' and old in src(c):
            set_src(c, src(c).replace(old, new)); n += 1
    assert n >= 1, f"pattern not found: {old!r}"
    return n

def build(name):
    cfg = CONFIGS[name]
    nb = json.load(open(f'notebooks/fanet_benchmark_{name}.ipynb', encoding='utf-8'))
    nb['cells'] = [c for c in nb['cells'] if not c.get('metadata', {}).get('colab_setup')]

    # environment-specific edits only
    replace_once(nb, cfg['data_dir_old'], 'DATA_DIR = DATA_ROOT  # downloaded by the Colab setup cell')
    replace_once(nb, 'num_workers=2', 'num_workers=NUM_WORKERS')
    replace_once(nb, 'CKPT_RESUME = "fanet_mfad_ISIC-2018_resume.pth"',
                 'CKPT_RESUME = os.path.join(WORK_DIR, "fanet_mfad_ISIC-2018_resume.pth")')
    replace_once(nb, 'CKPT_BEST   = "fanet_mfad_ISIC-2018_best.pth"',
                 'CKPT_BEST   = os.path.join(WORK_DIR, "fanet_mfad_ISIC-2018_best.pth")')
    replace_once(nb, 'if PREV_CKPT:\n    print(f"Found previous checkpoint',
                 '# Colab: resume from Drive if a previous session was interrupted\n'
                 'PREV_CKPT = CKPT_RESUME if os.path.exists(CKPT_RESUME) else PREV_CKPT\n'
                 'if PREV_CKPT:\n    print(f"Found previous checkpoint')
    replace_once(nb, "OUT_DIR = '/kaggle/working' if os.path.isdir('/kaggle/working') else '.'",
                 "OUT_DIR = WORK_DIR  # Google Drive")

    md = {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {"colab_setup": True},
          "source": SETUP_MD.format(title=cfg['title']).splitlines(keepends=True)}
    code = {"cell_type": "code", "execution_count": None, "id": uuid.uuid4().hex[:8],
            "metadata": {"colab_setup": True}, "outputs": [],
            "source": SETUP_CODE.format(slug=cfg['slug'], kaggle_ds=cfg['kaggle_ds']).splitlines(keepends=True)}
    nb['cells'][0:0] = [md, code]

    nb['metadata'] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
    }
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            ast.parse('\n'.join('pass' if l.strip().startswith(('!', '%')) else l for l in src(c).split('\n')))
    os.makedirs('notebooks/colab', exist_ok=True)
    out = f'notebooks/colab/fanet_benchmark_{name}_colab.ipynb'
    json.dump(nb, open(out, 'w', encoding='utf-8', newline='\n'), indent=1, ensure_ascii=False)
    print('wrote', out)

if __name__ == '__main__':
    for n in sys.argv[1:]:
        build(n)
