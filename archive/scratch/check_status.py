import kaggle

api = kaggle.KaggleApi()
api.authenticate()

kernels = api.kernels_list(mine=True, page_size=20)
print(f"{'Kernel Ref':<45} | {'Last Run Time':<25} | {'Status'}")
print("-" * 85)
for k in kernels:
    try:
        st = api.kernels_status(k.ref)
        status_str = getattr(st, 'status', 'N/A')
    except Exception as e:
        status_str = f"Err: {e}"
    print(f"{k.ref:<45} | {str(k.last_run_time):<25} | {status_str}")
