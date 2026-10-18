import json, os, logging
logging.basicConfig(level=logging.DEBUG)

with open(os.path.expanduser("~/.config/colab-cli/sessions.json")) as f:
    data = json.load(f)

sess = data["fanet-colab"]
from colab_cli.runtime import ColabRuntime

runtime = ColabRuntime(
    sess["url"],
    sess["token"],
    kernel_id=sess.get("kernel_id"),
    session_id=sess.get("session_id")
)

print("Runtime created, attempting execution...")
try:
    runtime.execute_code("print('Hello from Colab GPU!')", output_hook=print)
    print("SUCCESS!")
except Exception as e:
    import traceback
    traceback.print_exc()
