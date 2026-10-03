import json, os, shutil, subprocess, sys, glob

ROOT = os.getcwd()
NB = {
    "chasedb1":     "fanet_benchmark_chase_db1.ipynb",
    "cvc-clinicdb": "fanet_benchmark_cvc_clinicdb.ipynb",
    "drive":        "fanet_benchmark_drive.ipynb",
    "dsb2018":      "fanet_benchmark_dsb2018.ipynb",
    "em-dataset":   "fanet_benchmark_em_dataset.ipynb",
    "isic-2018":    "fanet_benchmark_isic_2018.ipynb",
    "kvasir-seg":   "fanet_benchmark_kvasir_seg.ipynb",
}
OUT = os.path.join(ROOT, "kaggle", "benchmark_kernels")

def ensure_kernelspec(path):
    j = json.load(open(path, encoding="utf-8"))
    md = j.setdefault("metadata", {})
    md["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    md["language_info"] = {"name": "python", "file_extension": ".py", "mimetype": "text/x-python"}
    json.dump(j, open(path, "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)

for _f in NB.values():
    ensure_kernelspec(os.path.join(ROOT, "notebooks", _f))

# patch DSB zip discovery (competition mount path differs between Kaggle layouts)
p = os.path.join(ROOT, "notebooks", NB["dsb2018"])
nbj = json.load(open(p, encoding="utf-8"))
old = 'ZIP_PATH = "/kaggle/input/data-science-bowl-2018/stage1_train.zip"'
new = ('_zips = glob.glob("/kaggle/input/**/stage1_train.zip", recursive=True)\n'
       'ZIP_PATH = _zips[0] if _zips else "/kaggle/input/data-science-bowl-2018/stage1_train.zip"')
for c in nbj["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    if old in s:
        c["source"] = s.replace(old, new).splitlines(keepends=True)
json.dump(nbj, open(p, "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)

only = sys.argv[1:] or list(NB)
for slug in only:
    d = os.path.join(OUT, slug)
    os.makedirs(d, exist_ok=True)
    meta_p = os.path.join(ROOT, "scratch_fanet", f"meta_{slug}", "kernel-metadata.json")
    meta = json.load(open(meta_p, encoding="utf-8")) if os.path.exists(meta_p) else {}
    meta.update({
        "id": f"namnguynnnn/fanet-benchmark-{slug}",
        "title": f"fanet-benchmark-{slug}",
        "code_file": NB[slug],
        "language": "python", "kernel_type": "notebook",
        "is_private": False, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
    })
    if slug == "dsb2018":
        meta["competition_sources"] = ["data-science-bowl-2018"]
    meta.pop("id_no", None)
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)
    shutil.copy(os.path.join(ROOT, "notebooks", NB[slug]), os.path.join(d, NB[slug]))
    r = subprocess.run(["kaggle", "kernels", "push", "-p", d], capture_output=True, text=True)
    print(slug, "->", (r.stdout + r.stderr).strip().replace("\n", " | "))
