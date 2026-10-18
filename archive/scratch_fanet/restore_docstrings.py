import re, ast, subprocess
subprocess.run(['git', 'checkout', '--', 'generate_benchmark_v2.py'], check=True)
p = 'generate_benchmark_v2.py'
lines = open(p, encoding='utf-8').read().split('\n')
out, i, n1, nm = [], 0, 0, 0
defre = re.compile(r'^\s*(def|class)\s.*:\s*$')
while i < len(lines):
    l = lines[i]
    prev_is_def = bool(out) and defre.match(out[-1] or '') is not None
    m1 = re.match(r'^(\s+)#(\S.*?)#\s*$', l)
    mo = re.match(r'^(\s+)#(\S.*)$', l)
    if prev_is_def and m1:
        out.append(f"{m1.group(1)}'''{m1.group(2)}'''"); n1 += 1
    elif prev_is_def and mo and not l.rstrip().endswith('#'):
        # multi-line docstring: find the lone '#' closing line
        j = i + 1
        while j < len(lines) and not re.match(r'^\s+#\s*$', lines[j]) and j - i < 15:
            j += 1
        if j < len(lines) and re.match(r'^\s+#\s*$', lines[j]):
            out.append(f"{mo.group(1)}'''{mo.group(2)}")
            out.extend(lines[i + 1:j])
            out.append(f"{mo.group(1)}'''")
            i = j; nm += 1
        else:
            out.append(l)
    else:
        out.append(l)
    i += 1
s = '\n'.join(out)
open(p, 'w', encoding='utf-8', newline='\n').write(s)
ast.parse(s)
print('single-line restored', n1, 'multi-line restored', nm)
