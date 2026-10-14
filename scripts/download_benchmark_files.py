import os
import sys
import re
import time
import requests
from kaggle.api.kaggle_api_extended import KaggleApi

def download_kernel_files(kernel_name, target_dir, pattern_str):
    os.makedirs(target_dir, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    pattern = re.compile(pattern_str)
    
    owner, slug = kernel_name.split('/')
    print(f"[*] Querying output files for {kernel_name}...")
    
    token = None
    downloaded = 0
    with api.build_kaggle_client() as kaggle_client:
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name = owner
            req.kernel_slug = slug
            api._set_paging(req, 100, token)
            resp = kaggle_client.kernels.kernels_api_client.list_kernel_session_output(req)
            
            for f in (resp.files or []):
                fname = f.file_name
                if pattern.search(fname):
                    dest = os.path.join(target_dir, os.path.basename(fname))
                    for attempt in range(5):
                        try:
                            print(f"[*] Downloading {fname} (attempt {attempt+1}) -> {dest}...")
                            r = requests.get(f.url, stream=True, timeout=60)
                            r.raise_for_status()
                            with open(dest, "wb") as out:
                                for chunk in r.iter_content(chunk_size=1024*1024):
                                    if chunk:
                                        out.write(chunk)
                            print(f"[+] Successfully saved {dest} ({os.path.getsize(dest)} bytes)")
                            downloaded += 1
                            break
                        except Exception as e:
                            print(f"[!] Attempt {attempt+1} failed: {e}. Retrying in 3s...")
                            time.sleep(3)
            
            token = resp.next_page_token
            if not token:
                break
                
    print(f"[+] Download complete for {kernel_name}: {downloaded} file(s).")

if __name__ == "__main__":
    download_kernel_files("namnguynnnn/fanet-benchmark-isic-2018", 
                          "results/benchmark_v2/isic-2018", 
                          r"(best\.pth|\.png|\.csv)$")
