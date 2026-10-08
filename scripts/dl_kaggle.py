import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
try:
    print("Fetching ISIC-2018 output...")
    api.kernels_output('namnguynnnn/fanet-benchmark-isic-2018', path='kaggle/outputs/isic-2018', force=True)
    print("Done ISIC!")
except Exception as e:
    print("Error:", e)
    
try:
    print("Fetching DSB2018 output...")
    api.kernels_output('namnguynnnn/fanet-benchmark-dsb2018', path='kaggle/outputs/dsb2018', force=True)
    print("Done DSB!")
except Exception as e:
    print("Error:", e)

