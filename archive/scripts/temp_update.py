import re
with open('docs/reports/README.md', 'r', encoding='utf-8') as f:
    text = f.read()

new_row = '| Phase 7E (Final): Comprehensive Benchmark V2 | Done (08/10) | Hoàn thành V2 Benchmark. DSB-2018: Dice +1.34%, Precision +2.38%. CVC-ClinicDB: Dice +4.11%, FPR giảm 3x. EM-Dataset: Precision +4.22%, FPR giảm >2x. CHASE-DB1 (Vessels) bị giảm nhẹ (Dice -2%). | docs/reports/2026-10-08_phase7e-final-benchmark.md |'

text = re.sub(r'(\|\s*Phase 7E:\s*[^\n]+)\n', r'\1\n' + new_row + '\n', text)

with open('docs/reports/README.md', 'w', encoding='utf-8') as f:
    f.write(text)
