import os, time, requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

api = KaggleApi()
api.authenticate()

target_dir = r"D:\User\nghonamjiwoo\My Project\FANet\GDF-Netv2\results\isic2018"
os.makedirs(target_dir, exist_ok=True)

with api.build_kaggle_client() as kaggle:
    req = ApiListKernelSessionOutputRequest()
    req.user_name = "namnguynnnn"
    req.kernel_slug = "gdfnet-v2-isic2018"
    api._set_paging(req, 20, None)
    resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)

print(f"[+] Found {len(resp.files)} files in kernel output:")
for item in resp.files:
    fname = item.file_name
    outfile = os.path.join(target_dir, fname)
    print(f"[*] Downloading {fname}...")

    # Retry loop with streaming chunks
    for attempt in range(5):
        try:
            r = requests.get(item.url, stream=True, timeout=60)
            if r.status_code == 200:
                with open(outfile, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                sz = os.path.getsize(outfile)
                print(f"[+] Success: {fname} ({sz / 1e6:.2f} MB)")
                break
            else:
                print(f"[-] HTTP {r.status_code}, retrying...")
                time.sleep(3)
        except Exception as e:
            print(f"[-] Attempt {attempt+1} failed: {e}. Retrying in 3s...")
            time.sleep(3)

print("[+] All files downloaded successfully!")
