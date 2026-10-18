import subprocess, os

res = subprocess.run(["ps", "-u", "-p", "10500"], capture_output=True, text=True)
print(res.stdout)

# Check all python threads
res_threads = subprocess.run(["ps", "-T", "-p", "10500"], capture_output=True, text=True)
print(res_threads.stdout)
