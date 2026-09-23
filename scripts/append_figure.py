with open('overleaf_submission/sections/04_experiments.tex', 'a', encoding='utf-8') as f:
    f.write('\n\n\\begin{figure}[t]\n')
    f.write('    \\centering\n')
    f.write('    \\includegraphics[width=\\linewidth]{paper_figures/pareto_frontier.pdf}\n')
    f.write("""    \\caption{The Precision-Recall Pareto Frontier on Kvasir-SEG (Sessile). While Phase 7B falls into the Monotonicity Trap (uncontrollable FPR), our Learned Residual Decoupling (Phase 7C/7D) restores architectural controllability. The resulting models smoothly traverse the Pareto boundary governed by the dataset's intrinsic morphological ambiguity.}\n""")
    f.write('    \\label{fig:pareto}\n')
    f.write('\\end{figure}\n')
