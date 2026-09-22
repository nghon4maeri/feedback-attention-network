import re

with open("overleaf_submission/sections/03_method.tex", "r", encoding="utf-8") as f:
    content = f.read()

# We want to replace everything from "\subsection{Proposed Solution: Detached Soft-OR Gating}" to the end of the file.
split_token = r"\subsection{Proposed Solution: Detached Soft-OR Gating}"

if split_token in content:
    pre_content, _ = content.split(split_token, 1)
    
    new_method = r"""\subsection{The Monotonicity Trap of Mathematical Gating}
To eradicate the Feedback Trap, our initial design (Phase 7B) explored a \textbf{Detached Soft-OR Gating} mechanism:
\begin{equation}
  \text{keep}_{\text{soft\_or}} = 1.0 - (1.0 - a) \odot (1.0 - \text{detach}(m_{\text{fg}})),
\end{equation}
where $a$ is the internal spatial attention map and $m_{\text{fg}}$ is the recurrent feedback mask. While gradient detachment successfully severed the toxic recursive gradient loop, empirical temporal tracking (FPR evaluation across iterations $t=0$ to $t=3$) revealed a catastrophic flaw: the False Positive Rate exploded to $52.31\%$.

We diagnose this failure as the \textbf{Soft-OR Monotonicity Trap}. The Soft-OR gate $A \lor B$ is mathematically a strictly monotonically increasing operator with respect to the prior mask $B$. Given that the initialization mask ($t=0$, generated via Otsu thresholding) frequently yields massive false positives in boundary-ambiguous flat polyps, the Soft-OR gate ensures that $\text{keep}_{\text{soft\_or}} \ge m_{\text{fg}}$. 
Consequently, the network is physically paralyzed---it is mathematically incapable of subtracting, pruning, or suppressing the noisy false positives inherited from previous iterations.

% -----------------------------------------------------------------------------
\subsection{Proposed Solution: Phase 7C Learned Residual Decoupling}
To overcome the Monotonicity Trap, the gating mechanism must be fully differentiable, structurally decoupled from recursive gradient flow, and capable of both augmenting and \textit{pruning} the prior mask.

We propose \textbf{Learned Residual Decoupling}. We discard rigid Boolean mathematics and instead introduce a low-parameter dynamic $1\times 1$ convolutional spatial attention gate:
\begin{equation}
  \text{combined} = [a \;\|\; \text{detach}(m_{\text{fg}})],
\end{equation}
\begin{equation}
  G = \sigma\Big(\text{Conv}_{1 \times 1}(\text{combined})\Big),
\end{equation}
where $G \in (0, 1)^{B \times 1 \times H \times W}$ acts as a spatial modulating gate. The final feature selection tensor is computed via a residual addition:
\begin{equation}
  \label{eq:learned_residual}
  \text{keep}_{\text{proposed}} = (G \odot \text{detach}(m_{\text{fg}})) + a.
\end{equation}

\subsubsection{Theoretical Advantages}
Our Phase 7C architecture possesses three critical properties:
\begin{enumerate}
    \item \textbf{Non-Monotonicity (Pruning Capability):} Unlike the OR-gate, if the network detects a false positive in the recurrent prior $m_{\text{fg}}$, it can actively push the learned gate $G \to 0$ and $a \to 0$, forcing $\text{keep}_{\text{proposed}} \to 0$. This fully restores the network's ability to prune historical errors.
    \item \textbf{Gradient Preservation:} Because the $1 \times 1$ convolution is continuous and differentiable, dense gradients flow back into the feature mask $a$.
    \item \textbf{Structural Decoupling:} We strictly preserve $\text{detach}(m_{\text{fg}})$. The recurrent gradient loop remains severed, forcing the internal feature encoder $a$ (and the learned gate $G$) to take absolute responsibility for minimizing false positives.
\end{enumerate}

To ensure the network utilizes its new pruning capabilities, we combine this architecture with a \textbf{Curriculum Adaptive Tversky Loss}, placing heavy asymmetric penalties on False Positives ($\alpha=0.7$, $\beta=0.3$) once the network reaches basic stability.
"""
    
    with open("overleaf_submission/sections/03_method.tex", "w", encoding="utf-8") as f:
        f.write(pre_content + new_method)
    print("Updated 03_method.tex successfully.")
else:
    print("Could not find the split token in 03_method.tex!")
