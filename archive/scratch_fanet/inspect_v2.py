import json, ast
p = 'notebooks/benchmark_v2/fanet_vs_mfad_kvasir_seg.ipynb'
nb = json.load(open(p, encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code': continue
    s = ''.join(c['source'])
    try: ast.parse(s)
    except SyntaxError as e:
        print('cell', i, e)
        ls = s.split('\n')
        print('\n'.join(f'{n+1}: {l}' for n, l in enumerate(ls) if 18 <= n+1 <= 28))
