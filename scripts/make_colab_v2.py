import json, os, sys, uuid, ast

SETUP_MD = """# Colab (A100) runner - Benchmark V2: {title}
**Runtime -> Change runtime type -> GPU (A100).**
Outputs and checkpoints will be saved to Google Drive (`WORK_DIR`).
Kaggle data download needs credentials: add Colab *Secrets* `KAGGLE_USERNAME` + `KAGGLE_KEY`."""

SETUP_CODE = '''# ================================================================
# [Colab Setup] Drive, GPU, deps, dataset download
# ================================================================
import os, sys, subprocess, json, zipfile, glob
from google.colab import drive
drive.mount('/content/drive')

WORK_DIR = "/content/drive/MyDrive/FANet_runs_V2/{slug}"
os.makedirs(WORK_DIR, exist_ok=True)
DATA_ROOT = "/content/{slug}"
os.makedirs(DATA_ROOT, exist_ok=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "albumentations", "tifffile", "opencv-python-headless"], check=True)

import torch
print("CUDA:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
torch.backends.cudnn.benchmark = True

from google.colab import userdata
os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')
if not os.path.isdir(DATA_ROOT) or not os.listdir(DATA_ROOT):
    !mkdir -p {{DATA_ROOT}}
    !kaggle competitions download -c {kaggle_ds} -p {{DATA_ROOT}} --unzip || kaggle datasets download -d {kaggle_ds} -p {{DATA_ROOT}} --unzip
print("Data ready:", DATA_ROOT, "| entries:", len(os.listdir(DATA_ROOT)))
'''

CONFIGS = {
    "isic_2018": dict(
        title="ISIC-2018", slug="isic2018",
        kaggle_ds="tschandl/isic2018-challenge-task1-data-segmentation",
        data_dir_old='DATA_DIR = "/kaggle/input/datasets/tschandl/isic2018-challenge-task1-data-segmentation"',
    ),
    "dsb2018": dict(
        title="DSB-2018", slug="dsb2018",
        kaggle_ds="data-science-bowl-2018",
        data_dir_old='ZIP_PATH = _zips[0] if _zips else "/kaggle/input/data-science-bowl-2018/stage1_train.zip"',
    ),
}

def src(c): return ''.join(c['source'])
def set_src(c, s): c['source'] = s.splitlines(keepends=True)

def replace_once(nb, old, new):
    n = 0
    for c in nb['cells']:
        if c['cell_type'] == 'code' and old in src(c):
            set_src(c, src(c).replace(old, new)); n += 1
    return n

def build(name):
    cfg = CONFIGS[name]
    nb = json.load(open(f'notebooks/benchmark_v2/fanet_vs_mfad_{name}.ipynb', encoding='utf-8'))
    
    if name == 'isic_2018':
        replace_once(nb, cfg['data_dir_old'], 'DATA_DIR = DATA_ROOT  # downloaded by the Colab setup cell')
    elif name == 'dsb2018':
        replace_once(nb, cfg['data_dir_old'], 'ZIP_PATH = glob(os.path.join(DATA_ROOT, "stage1_train.zip"))[0] if glob(os.path.join(DATA_ROOT, "stage1_train.zip")) else os.path.join(DATA_ROOT, "stage1_train.zip")')

    replace_once(nb, "csv_path = f\"benchmark_results_{dataset_name.replace(' ', '_')}.csv\"",
                 "csv_path = os.path.join(WORK_DIR, f\"benchmark_results_{dataset_name.replace(' ', '_')}.csv\")")
    replace_once(nb, "plt.savefig(f'training_curves_{DATASET_NAME.replace(\" \", \"_\")}.png'",
                 "plt.savefig(os.path.join(WORK_DIR, f'training_curves_{DATASET_NAME.replace(\" \", \"_\")}.png')")
    replace_once(nb, "plt.savefig(f'qualitative_{DATASET_NAME.replace(\" \", \"_\")}.png'",
                 "plt.savefig(os.path.join(WORK_DIR, f'qualitative_{DATASET_NAME.replace(\" \", \"_\")}.png')")
    
    replace_once(nb, "ckpt_path = f\"{model_name}_{DATASET_NAME.replace(' ', '_')}_best.pth\"",
                 "ckpt_path = os.path.join(WORK_DIR, f\"{model_name}_{DATASET_NAME.replace(' ', '_')}_best.pth\")")
                 
    md = {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": SETUP_MD.format(title=cfg['title']).splitlines(keepends=True)}
    code = {"cell_type": "code", "execution_count": None, "id": uuid.uuid4().hex[:8], "metadata": {}, "outputs": [], "source": SETUP_CODE.format(slug=cfg['slug'], kaggle_ds=cfg['kaggle_ds']).splitlines(keepends=True)}
    nb['cells'][0:0] = [md, code]
    
    out = f'notebooks/colab/benchmark_v2_fanet_vs_mfad_{name}_colab.ipynb'
    json.dump(nb, open(out, 'w', encoding='utf-8', newline='\n'), indent=1, ensure_ascii=False)
    print('wrote', out)

if __name__ == '__main__':
    for n in sys.argv[1:]:
        build(n)
