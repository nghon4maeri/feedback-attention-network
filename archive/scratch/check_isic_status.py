import kaggle
import datetime

api = kaggle.KaggleApi()
api.authenticate()

ref = 'namnguynnnn/gdfnet-v2-isic2018'
status = api.kernels_status(ref)
print("ISIC2018 Status:", status.status)
print("Failure Message:", status.failure_message)

# Let's check kernel details
try:
    kernel_list = api.kernels_list(search='gdfnet-v2-isic2018', mine=True)
    for k in kernel_list:
        print("Found kernel:", k.ref, "Last run:", k.last_run_time)
except Exception as e:
    print("Error listing:", e)
