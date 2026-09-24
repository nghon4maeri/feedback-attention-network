with open('overleaf_submission/sections/04_experiments.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = lines[:504]
with open('overleaf_submission/sections/04_experiments_fix.tex', 'r', encoding='utf-8') as f:
    new_lines.append(f.read())

new_lines.append('\n\n\\begin{figure}[t]\n')
new_lines.append('    \\centering\n')
new_lines.append('    \\includegraphics[width=\\linewidth]{paper_figures/pareto_frontier.pdf}\n')
new_lines.append("""    \\caption{The Precision-Recall Pareto Frontier on Kvasir-SEG (Sessile). While Phase 7B falls into the Monotonicity Trap (uncontrollable FPR), our Learned Residual Decoupling (Phase 7C/7D) restores architectural controllability. The resulting models smoothly traverse the Pareto boundary governed by the dataset's intrinsic morphological ambiguity.}\n""")
new_lines.append('    \\label{fig:pareto}\n')
new_lines.append('\\end{figure}\n')

with open('overleaf_submission/sections/04_experiments.tex', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
