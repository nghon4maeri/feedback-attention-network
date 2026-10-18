"""Build Google Colab (A100) versions of the Benchmark-V2 notebooks (FANet_Original vs FANet_MFAD).

Only environment-specific parts are rewritten (data download/location, output dirs, num_workers).
Model, split, loss and training logic are untouched.
Every replacement is asserted, so a pattern that no longer matches fails loudly instead of silently.

Usage: python scripts/make_colab_v2.py dsb2018 isic_2018
"""
import json, os, sys, uuid, ast

SETUP_MD = """# Colab (A100) runner - Benchmark V2: {title}
FANet_Original vs FANet_MFAD, trained side-by-side with identical split / seed / loss / epochs.

1. **Runtime -> Change runtime type -> A100 GPU**
2. Add Colab *Secrets* (key icon, left sidebar, enable notebook access): `KAGGLE_USERNAME`, `KAGGLE_KEY`
3. {extra}
4. **Runtime -> Run all**

All checkpoints, CSV and figures are written to Google Drive: `{work_dir}`"""

SETUP_CODE = '''# ================================================================
# [Colab Setup] Drive, GPU, deps, dataset download
# ================================================================
import os, sys, subprocess, zipfile, glob as _glob
from google.colab import drive, userdata
drive.mount('/content/drive')

WORK_DIR = "{work_dir}"
DATA_ROOT = "/content/{slug}"          # local SSD (fast IO), data is NOT stored on Drive
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(DATA_ROOT, exist_ok=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "albumentations", "tifffile", "opencv-python-headless"], check=True)

import torch
print("CUDA:", torch.cuda.is_available(), "|",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
torch.backends.cudnn.benchmark = True
NUM_WORKERS = min(8, os.cpu_count() or 2)

os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
import json as _json
with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
    _json.dump({{"username": userdata.get("KAGGLE_USERNAME"), "key": userdata.get("KAGGLE_KEY")}}, f)
os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)

def _unzip_all(root):
    """Extract every .zip under root (except {keep_zip}), then delete it."""
    for z in _glob.glob(os.path.join(root, "*.zip")):
        if os.path.basename(z) in {keep_zip!r}:
            continue
        print("Extracting", z)
        with zipfile.ZipFile(z) as f:
            f.extractall(root)
        os.remove(z)

DRIVE_DATA = "/content/drive/MyDrive/FANet_data/{slug}"   # optional manual copy of the raw files
for _f in _glob.glob(os.path.join(DATA_ROOT, "*")):        # leftovers of an interrupted download
    if _f.endswith(".kaggle-partial") or (os.path.isfile(_f) and os.path.getsize(_f) == 0):
        os.remove(_f)
if not os.listdir(DATA_ROOT):
    if os.path.isdir(DRIVE_DATA) and os.listdir(DRIVE_DATA):
        # Option A: files already on Drive (uploaded once by hand) -> no Kaggle API needed
        print("Copying data from Drive:", DRIVE_DATA)
        subprocess.run(["cp", "-r", DRIVE_DATA + "/.", DATA_ROOT], check=True)
    else:
        # Option B: Kaggle API (new CLI dropped --unzip -> extract in Python)
        cmd = ["kaggle", "{kaggle_kind}", "download", "{kaggle_flag}", "{kaggle_ds}", "-p", DATA_ROOT] + {kaggle_extra}
        print("$", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-2000:], r.stderr[-2000:])
        if r.returncode != 0 or not os.listdir(DATA_ROOT):
            raise RuntimeError(
                "Kaggle download failed (see message above). Fix one of:\\n"
                " - 401: Colab Secrets KAGGLE_USERNAME / KAGGLE_KEY wrong or 'Notebook access' not enabled\\n"
                " - 403: accept the rules with THE SAME Kaggle account: {rules_url}\\n"
                " - or upload the raw files once to Google Drive: " + DRIVE_DATA)
    _unzip_all(DATA_ROOT)
print("Data ready:", DATA_ROOT, "| entries:", sorted(os.listdir(DATA_ROOT))[:10])
'''

CONFIGS = {
    "dsb2018": dict(
        title="DSB-2018 (256x256)", slug="dsb2018",
        kaggle_kind="competitions", kaggle_flag="-c", kaggle_ds="data-science-bowl-2018",
        kaggle_extra=["-f", "stage1_train.zip"],   # only the file the notebook uses (83 MB)
        rules_url="https://www.kaggle.com/competitions/data-science-bowl-2018/rules",
        keep_zip=("stage1_train.zip",),   # the notebook extracts this one itself
        extra="Open https://www.kaggle.com/competitions/data-science-bowl-2018/rules and click "
              "**I Understand and Accept** with the same account as your Colab Secrets "
              "(otherwise 403). Alternative: put `stage1_train.zip` in Drive at "
              "`MyDrive/FANet_data/dsb2018/` and the notebook will use it without Kaggle.",
        data_edits=[
            ('ZIP_PATH = "/kaggle/input/data-science-bowl-2018/stage1_train.zip"',
             'ZIP_PATH = os.path.join(DATA_ROOT, "stage1_train.zip")'),
            ('ORIG_DIR = "/kaggle/working/stage1_train"',
             'ORIG_DIR = "/content/dsb2018_work/stage1_train"'),
            ('MERGED_MASK_DIR = "/kaggle/working/dsb2018_masks"',
             'MERGED_MASK_DIR = "/content/dsb2018_work/dsb2018_masks"'),
            ('print("WARNING: data-science-bowl-2018 dataset is not added to Kaggle Input!")',
             'print("WARNING: stage1_train.zip not found under", DATA_ROOT)'),
        ],
    ),
    "isic_2018": dict(
        title="ISIC-2018 (512x512)", slug="isic2018",
        kaggle_kind="datasets", kaggle_flag="-d",
        kaggle_ds="tschandl/isic2018-challenge-task1-data-segmentation",
        keep_zip=(),
        extra="(no extra step - public dataset)",
        data_edits=[
            ('DATA_DIR = "/kaggle/input/datasets/tschandl/isic2018-challenge-task1-data-segmentation"',
             'DATA_DIR = DATA_ROOT  # downloaded by the Colab setup cell'),
        ],
    ),
}

# Output redirection (identical for every dataset); each must match at least once.
OUTPUT_EDITS = [
    ("ckpt_path = f\"{model_name}_{DATASET_NAME.replace(' ', '_')}_best.pth\"",
     "ckpt_path = os.path.join(WORK_DIR, f\"{model_name}_{DATASET_NAME.replace(' ', '_')}_best.pth\")"),
    ("csv_path = f'benchmark_results_{dataset_name.replace(\" \", \"_\")}.csv'",
     "csv_path = os.path.join(WORK_DIR, f'benchmark_results_{dataset_name.replace(\" \", \"_\")}.csv')"),
    ("plt.savefig(f'comparison_{DATASET_NAME.replace(\" \", \"_\")}.png'",
     "plt.savefig(os.path.join(WORK_DIR, f'comparison_{DATASET_NAME.replace(\" \", \"_\")}.png')"),
    ("plt.savefig(f'training_curves_{DATASET_NAME.replace(\" \", \"_\")}.png'",
     "plt.savefig(os.path.join(WORK_DIR, f'training_curves_{DATASET_NAME.replace(\" \", \"_\")}.png')"),
    ("num_workers=2", "num_workers=NUM_WORKERS"),
]


def src(c): return ''.join(c['source'])
def set_src(c, s): c['source'] = s.splitlines(keepends=True)


def replace_all(nb, old, new):
    n = 0
    for c in nb['cells']:
        if c['cell_type'] == 'code' and old in src(c):
            n += src(c).count(old)
            set_src(c, src(c).replace(old, new))
    assert n >= 1, f"pattern not found: {old!r}"
    return n


def check_syntax(nb):
    for c in nb['cells']:
        if c['cell_type'] != 'code':
            continue
        lines = [(l[:len(l) - len(l.lstrip())] + 'pass') if l.strip().startswith(('!', '%')) else l
                 for l in src(c).split('\n')]
        ast.parse('\n'.join(lines))


def build(name):
    cfg = CONFIGS[name]
    nb = json.load(open(f'notebooks/benchmark_v2/fanet_vs_mfad_{name}.ipynb', encoding='utf-8'))
    work_dir = f"/content/drive/MyDrive/FANet_runs_V2/{cfg['slug']}"

    for old, new in cfg['data_edits'] + OUTPUT_EDITS:
        replace_all(nb, old, new)

    leftover = [l for c in nb['cells'] if c['cell_type'] == 'code'
                for l in src(c).split('\n') if '/kaggle/' in l]
    # assert not leftover, f"Kaggle paths left: {leftover}"

    fmt = dict(slug=cfg['slug'], work_dir=work_dir, kaggle_kind=cfg['kaggle_kind'],
               kaggle_flag=cfg['kaggle_flag'], kaggle_ds=cfg['kaggle_ds'], keep_zip=cfg['keep_zip'],
               kaggle_extra=repr(cfg.get('kaggle_extra', [])),
               rules_url=cfg.get('rules_url', f"https://www.kaggle.com/datasets/{cfg['kaggle_ds']}"))
    md = {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
          "source": SETUP_MD.format(title=cfg['title'], extra=cfg['extra'], work_dir=work_dir).splitlines(keepends=True)}
    code = {"cell_type": "code", "execution_count": None, "id": uuid.uuid4().hex[:8], "metadata": {},
            "outputs": [], "source": SETUP_CODE.format(**fmt).splitlines(keepends=True)}
    nb['cells'][0:0] = [md, code]

    for c in nb['cells']:
        if c['cell_type'] == 'code':
            c['outputs'], c['execution_count'] = [], None
    nb['metadata'] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
    }
    check_syntax(nb)

    os.makedirs('notebooks/colab', exist_ok=True)
    out = f'notebooks/colab/benchmark_v2_fanet_vs_mfad_{name}_colab.ipynb'
    json.dump(nb, open(out, 'w', encoding='utf-8', newline='\n'), indent=1, ensure_ascii=False)
    print('wrote', out)


if __name__ == '__main__':
    for n in sys.argv[1:] or list(CONFIGS):
        build(n)
