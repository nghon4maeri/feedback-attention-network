import os
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

kernels = [
    "namnguynnnn/fanet-benchmark-cvc-clinicdb",
    "namnguynnnn/fanet-benchmark-chasedb1",
    "namnguynnnn/fanet-benchmark-em-dataset",
    "namnguynnnn/fanet-benchmark-isic-2018",
    "namnguynnnn/fanet-benchmark-kvasir-seg",
    "namnguynnnn/fanet-benchmark-drive"
]

os.makedirs("kaggle_logs_parsed", exist_ok=True)

for kernel in kernels:
    print(f"Fetching log for {kernel}...")
    try:
        # kernels_output_cli(kernel, path="kaggle_logs_parsed", force=True) 
        # But we don't want files, we just want the output log!
        response = api.kernel_status(kernel)
        if response.get("status") == "complete":
            # api.kernel_output(kernel) doesn't just return log. It downloads.
            # Let's download ONLY the .log file if possible, or just parse the notebook json.
            # Kaggle API doesn't allow filtering downloads. 
            pass
    except Exception as e:
        print(f"Error for {kernel}: {e}")

print("Done")
