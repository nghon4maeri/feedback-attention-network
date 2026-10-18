with open('generate_benchmark_notebook.py', 'r', encoding='utf-8') as f:
    text = f.read()

search_str = """# DATASET_NAME = "CHASE-DB1"
# DATA_DIR = "/kaggle/input/chasedb1"
# IMG_SUBDIR = "images"
# MASK_SUBDIR = "masks"
"""

replace_str = """# DATASET_NAME = "CHASE-DB1"
# DATA_DIR = "/kaggle/input/chasedb1"
# IMG_SUBDIR = "images"
# MASK_SUBDIR = "masks"

# DATASET_NAME = "EM-Dataset"
# DATA_DIR = "/kaggle/input/electron-microscopy-3d-segmentation"
# IMG_SUBDIR = "images" 
# MASK_SUBDIR = "masks"
"""

text = text.replace(search_str, replace_str)

with open('generate_benchmark_notebook.py', 'w', encoding='utf-8') as f:
    f.write(text)
