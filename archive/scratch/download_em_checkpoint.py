import time
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

for attempt in range(5):
    try:
        print(f"[*] Download attempt {attempt + 1}/5 for EM dataset outputs...")
        api.kernels_output("namnguynnnn/gdfnet-v2-em-dataset", path="GDF-Netv2/results/em_dataset", force=True)
        print("[+] Successfully downloaded all outputs!")
        break
    except Exception as e:
        print(f"[-] Error: {e}. Retrying in 5 seconds...")
        time.sleep(5)
