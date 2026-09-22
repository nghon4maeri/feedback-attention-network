import os, re

def rollback_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return
        
    orig = content
    
    # Replace old >0.8 or 0.2/0.3 numbers with the true benchmark
    # M11 = 0.3428 (was 0.2517 or 0.8123)
    # M12 = 0.2183 (was 0.2995 or 0.8654)
    content = content.replace('0.2517', '0.3428')
    content = content.replace('0.2995', '0.2183')
    
    # We will just write a big warning block in the experiments section to override the narrative
    if "04_experiments.tex" in filepath:
        # Let's replace the whole Table 4 and subsequent text with the new realization.
        # Actually it's easier to just append a new subsection at the top of 04_experiments.tex
        new_intro = r"""
\subsection*{Emergency Retraction & Mechanistic Post-Mortem (The Soft-OR Monotonicity Trap)}
\textcolor{red}{\textbf{CRITICAL UPDATE:}} Initial evaluations of the proposed Detached Soft-OR gating ($M_{12}$) on the Kvasir-Sessile subset revealed a catastrophic performance collapse. Contrary to our hypothesis that severing the gradient graph would suppress false positives, $M_{12}$ exacerbates them, driving the False Positive Rate (FPR) to an unacceptable \textbf{52.31\%} (compared to the baseline $M_{11}$'s 21.31\%) and plummeting the Dice score to \textbf{0.2183} (vs $M_{11}$'s 0.3428).

Through rigorous temporal tracking, we identified the fundamental mathematical flaw in our Phase 7B design. The Soft-OR gate is defined as:
\[ \text{keep} = 1.0 - (1.0 - \text{fmask}) \times (1.0 - m_{\text{fg}}) \]
This operator is \textit{strictly monotonically increasing} with respect to the feedback prior $m_{\text{fg}}$. If the initial Otsu thresholding step ($t=0$) introduces massive false positive regions ($56.20\%$ FPR), the Soft-OR gate guarantees that $\text{keep} \ge m_{\text{fg}}$. Consequently, the network is mathematically paralyzed---it cannot shrink or prune noisy feedback, forcing the false positives to accumulate uncontrollably across recurrent iterations. Furthermore, the weights for $M_{12}$ were trained using standard DiceBCE loss, lacking the heavy False Positive penalty that an Adaptive Tversky Loss would provide.

The baseline $M_{11}$ achieves a Dice of \textbf{0.3428}. This reflects the true, brutal difficulty of the Sessile (flat) polyp domain. Future iterations must replace the Soft-OR gate with a non-monotonic learned gating mechanism (e.g., a decoupled attention mechanism that can both add \textit{and} subtract features).

"""
        if "Emergency Retraction" not in content:
            content = content.replace(r"\subsection{Diagnostic Verification: Gradient Restoration and Representation Alignment}", new_intro + "\n" + r"\subsection{Diagnostic Verification: Gradient Restoration and Representation Alignment}")
            
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Updated {filepath}")

for root, _, files in os.walk('.'):
    for file in files:
        if file.endswith('.tex') or file.endswith('.md'):
            rollback_file(os.path.join(root, file))
