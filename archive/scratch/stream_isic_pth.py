import os, time, requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

api = KaggleApi()
api.authenticate()

target_dir = r"D:\User\nghonamjiwoo\My Project\FANet\GDF-Netv2\results\isic2018"
os.makedirs(target_dir, exist_ok=True)
outfile = os.path.join(target_dir, "GDFNetV2_ISIC2018_best.pth")

with api.build_kaggle_client() as kaggle:
    req = ApiListKernelSessionOutputRequest()
    req.user_name = "namnguynnnn"
    req.kernel_slug = "gdfnet-v2-isic2018"
    api._set_paging(req, 20, None)
    resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)

for item in resp.files:
    if item.file_name == "GDFNetV2_ISIC2018_best.pth":
        print(f"[*] Starting download: {item.file_name} (129.38 MB)...", flush=True)
        r = requests.get(item.url, stream=True, timeout=120)
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        last_print = 0
        with open(outfile, "wb") as f:
            for chunk in r.iter_content(chunk_size=2 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    mb = downloaded / (1024 * 1024)
                    if mb - last_print >= 10 or downloaded == total:
                        print(f"    Progress: {mb:.1f} MB / {total / (1024*1024):.1f} MB ({downloaded*100/total:.1f}%)", flush=True)
                        last_print = mb
        print(f"[+] Download complete: {os.path.getsize(outfile)} bytes", flush=True)
