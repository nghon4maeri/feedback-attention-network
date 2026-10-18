import glob
import json
import re

for filepath in glob.glob('notebooks/*.ipynb'):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    changed = False
    for cell in data['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # Check if it's the dataset cell
            if "class UniversalDataset(Dataset):" in source or "exts = " in source:
                
                # 1. Add PIL if not there
                if "import cv2" in source and "from PIL import Image" not in source:
                    source = source.replace("import cv2\n", "import cv2\nfrom PIL import Image\n")
                    
                # 2. Add .gif to extensions
                if "*.gif" not in source:
                    source = source.replace("['*.jpg', '*.png', '*.tif', '*.tiff', '*.jpeg']", "['*.jpg', '*.png', '*.tif', '*.tiff', '*.jpeg', '*.gif']")
                
                # 3. Patch Image Read
                img_read_old = "image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)"
                img_read_new = """if self.images_path[index].lower().endswith('.gif'):
            image = cv2.cvtColor(np.array(Image.open(self.images_path[index]).convert('RGB')), cv2.COLOR_RGB2BGR)
        else:
            image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)"""
                if img_read_old in source:
                    source = source.replace(img_read_old, img_read_new)
                
                # 4. Patch Mask Read
                mask_read_old = "mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)"
                mask_read_new = """if self.masks_path[index].lower().endswith('.gif'):
            mask = np.array(Image.open(self.masks_path[index]).convert('L'))
        else:
            mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)"""
                if mask_read_old in source:
                    source = source.replace(mask_read_old, mask_read_new)

                # Split back into lines for Jupyter format
                lines = source.split('\n')
                cell['source'] = [line + ('\n' if i < len(lines)-1 else '') for i, line in enumerate(lines)]
                
                # Handle trailing empty newline if split created it
                if len(cell['source']) > 0 and cell['source'][-1] == '\n':
                    cell['source'] = cell['source'][:-1]

                changed = True
                
    if changed:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        print(f"Patched GIF support in {filepath}")
