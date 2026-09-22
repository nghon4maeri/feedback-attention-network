import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.kernels_output_cli('namnguynnnn/rebuttal-cross-domain-eva', path='kaggle/downloads_rebuttal')
