import json, ast
p = 'notebooks/fanet_benchmark_cvc_clinicdb.ipynb'
nb = json.load(open(p, encoding='utf-8'))
old = "replace('\\', '/')"
new_ = "replace(chr(92), '/')"
n = 0
for c in nb['cells']:
    if c['cell_type'] != 'code':
        continue
    out = []
    for l in c['source']:
        if old in l:
            n += 1
            l = l.replace(old, new_)
        out.append(l)
    c['source'] = out
json.dump(nb, open(p, 'w', encoding='utf-8', newline='\n'), indent=1, ensure_ascii=False)
print('fixed lines', n)
for c in nb['cells']:
    if c['cell_type'] == 'code':
        s = '\n'.join('pass' if l.strip().startswith(('!', '%')) else l for l in ''.join(c['source']).split('\n'))
        ast.parse(s)
print('ok')
