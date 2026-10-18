import json, glob, ast
bad = 0
for p in glob.glob('notebooks/**/*.ipynb', recursive=True) + glob.glob('kaggle/benchmark_kernels/**/*.ipynb', recursive=True):
    nb = json.load(open(p, encoding='utf-8'))
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            s = '\n'.join('pass' if l.strip().startswith(('!', '%')) else l for l in ''.join(c['source']).split('\n'))
            try: ast.parse(s)
            except SyntaxError as e: bad += 1; print('BAD', p, e)
for p in ['rebuild_all_notebooks.py', 'generate_benchmark_v2.py']:
    ast.parse(open(p, encoding='utf-8').read())
print('syntax errors:', bad)
nb = json.load(open('notebooks/colab/fanet_benchmark_isic_2018_colab.ipynb', encoding='utf-8'))
for c in nb['cells']:
    s = ''.join(c['source'])
    if 'class DiceBCELoss' in s: print(s[:700])
