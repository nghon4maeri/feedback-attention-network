"""
Generate publication-quality 2D scientific architectural diagram for GDF-Net:
Panel (a): Baseline (Hard Gating / Feedback Trap)
Panel (b): Proposed (Detached Soft-OR Gating)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_diagram():
    # Set up high-res figure with pure white background
    fig = plt.figure(figsize=(15, 14), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Color palette - clean, publication grade
    c_blue_fill = '#E8F0FE'
    c_blue_edge = '#1A73E8'
    c_orange_fill = '#FEF3E2'
    c_orange_edge = '#EA8600'
    c_red_fill = '#FCE8E6'
    c_red_edge = '#D93025'
    c_green_fill = '#E6F4EA'
    c_green_edge = '#137333'
    c_white_fill = '#FFFFFF'
    c_gray_edge = '#3C4043'
    c_text_dark = '#202124'
    c_text_red = '#C5221F'
    c_text_green = '#0D652D'

    # Font configurations
    font_box_header = {'fontsize': 13.5, 'fontweight': 'bold', 'color': '#202124', 'ha': 'center', 'va': 'center'}
    
    # Helper function to draw rounded boxes
    def draw_box(x, y, w, h, fill, edge, lw=2.0, ls='-', r=1.5):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle=f"round,pad=0,rounding_size={r}",
                             facecolor=fill, edgecolor=edge,
                             linewidth=lw, linestyle=ls, zorder=3)
        ax.add_patch(box)
        return box

    # =========================================================================
    # PART (A): BASELINE (HARD GATING / FEEDBACK TRAP) - TOP HALF
    # =========================================================================
    # Section Header
    ax.text(50, 97, "(a) Baseline (Hard Gating / Feedback Trap)",
            fontsize=17, fontweight='bold', color='#1A1A1A', ha='center', va='center')

    ax.text(50, 94.2, "Indicator step function causes gradient paralysis on internal attention, while toxic error leaks unconstrained into recurrent prior",
            fontsize=11.5, fontstyle='italic', color='#5F6368', ha='center', va='center')

    # 1. Top Central Box: Backbone fmask
    # Box center: (50, 83), width: 26, height: 7.5
    draw_box(37, 79.5, 26, 7.5, c_blue_fill, c_blue_edge, lw=2.2)
    ax.text(50, 84.7, "Backbone", **font_box_header)
    ax.text(50, 81.5, r"$\mathbf{f}_{\mathrm{mask}}$ (Internal Attention)", fontsize=12.5, fontweight='bold', color=c_blue_edge, ha='center', va='center')

    # 2. Horizontal Composite Block: Recurrent | Hard Gate | Output Feature
    # Recurrent m_fg: X from 12 to 33 (width 21), Y from 62.5 to 70.5 (height 8)
    draw_box(12, 62.5, 21, 8, c_orange_fill, c_orange_edge, lw=2.2)
    ax.text(22.5, 67.7, "Recurrent Prior", **font_box_header)
    ax.text(22.5, 64.7, r"$m_{\mathrm{fg}} = \mathcal{D}(m^{(t-1)})$", fontsize=12, fontweight='bold', color='#B06000', ha='center', va='center')

    # Hard Gate: X from 35 to 69 (width 34)
    draw_box(35, 62.5, 34, 8, c_red_fill, c_red_edge, lw=2.2)
    ax.text(52, 67.7, "Hard Gate Operator", fontsize=13.5, fontweight='bold', color=c_text_red, ha='center', va='center')
    ax.text(52, 64.7, r"$\max\left(\mathbb{I}(\mathbf{f}_{\mathrm{mask}} > 0.5),\; m_{\mathrm{fg}}\right)$", fontsize=13, fontweight='bold', color=c_text_red, ha='center', va='center')

    # Output Feature: X from 71 to 88 (width 17)
    draw_box(71, 62.5, 17, 8, c_white_fill, c_gray_edge, lw=2.2)
    ax.text(79.5, 67.7, "Output Feature", **font_box_header)
    ax.text(79.5, 64.7, r"$\mathbf{F}_{\mathrm{gated}}$", fontsize=13, fontweight='bold', color='#3C4043', ha='center', va='center')

    # --- Arrows in Part (a) ---
    # Downward forward arrow from Backbone to Hard Gate
    arrow_fwd_a = FancyArrowPatch((50, 79.5), (50, 70.8),
                                  arrowstyle='->,head_width=4,head_length=6',
                                  color='#202124', linewidth=2.2, zorder=4)
    ax.add_patch(arrow_fwd_a)
    ax.text(51.8, 75.2, "Forward", fontsize=11, fontweight='bold', color='#202124',
            bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none"), ha='left', va='center')

    # Forward arrow from Recurrent to Hard Gate (along upper part of boundary)
    arrow_rec_gate = FancyArrowPatch((33, 67.5), (35, 67.5),
                                     arrowstyle='->,head_width=3.5,head_length=5',
                                     color='#202124', linewidth=1.8, zorder=4)
    ax.add_patch(arrow_rec_gate)

    # Toxic Leakage red dashed arrow from Recurrent m_fg into Hard Gate
    arrow_toxic = FancyArrowPatch((33, 65.0), (35, 65.0),
                                  arrowstyle='->,head_width=4,head_length=6',
                                  color=c_red_edge, linewidth=2.0, linestyle='--', zorder=4)
    ax.add_patch(arrow_toxic)
    ax.text(34, 60.5, "Toxic Leakage", fontsize=11, fontweight='bold', color=c_text_red, ha='center', va='center')

    # Forward arrow from Hard Gate to Output Feature
    arrow_gate_out = FancyArrowPatch((69, 66.5), (71, 66.5),
                                     arrowstyle='->,head_width=3.5,head_length=5',
                                     color='#202124', linewidth=1.8, zorder=4)
    ax.add_patch(arrow_gate_out)

    # Backward gradient from Output Feature back to Hard Gate
    arrow_out_bwd = FancyArrowPatch((71, 64.5), (69, 64.5),
                                    arrowstyle='->,head_width=3.5,head_length=5',
                                    color=c_red_edge, linewidth=1.8, linestyle='--', zorder=4)
    ax.add_patch(arrow_out_bwd)

    # Blocked gradient from Hard Gate back up to Backbone fmask
    arrow_blocked = FancyArrowPatch((44, 70.5), (44, 79.5),
                                    arrowstyle='->,head_width=4,head_length=6',
                                    connectionstyle="arc3,rad=-0.15",
                                    color=c_red_edge, linewidth=2.2, linestyle='--', zorder=4)
    ax.add_patch(arrow_blocked)
    ax.text(35, 75.2, r"$\nabla \equiv 0$ (Blocked)", fontsize=12, fontweight='bold', color=c_text_red,
            bbox=dict(boxstyle="square,pad=0.25", fc="#FFF0F0", ec=c_red_edge, lw=1.2), ha='center', va='center')

    # Red dashed loop from Output Feature routing underneath back to boundary between Recurrent and Hard Gate
    # Path: (79.5, 62.5) -> (79.5, 54.0) -> (34.0, 54.0) -> (34.0, 62.2)
    ax.plot([79.5, 79.5, 34.0, 34.0], [62.5, 54.0, 54.0, 61.8],
            color=c_red_edge, linewidth=2.0, linestyle='--', zorder=4)
    arrow_loop = FancyArrowPatch((34.0, 59.5), (34.0, 62.2),
                                 arrowstyle='->,head_width=4,head_length=6',
                                 color=c_red_edge, linewidth=2.0, linestyle='--', zorder=4)
    ax.add_patch(arrow_loop)
    ax.text(56.5, 52.2, "Recursive Error Amplification Loop (Backward Pass)",
            fontsize=11.5, fontweight='bold', color=c_text_red, ha='center', va='center')


    # =========================================================================
    # SEPARATOR
    # =========================================================================
    ax.plot([6, 94], [48.5, 48.5], color='#CFD8DC', linewidth=1.5, linestyle=':', zorder=2)


    # =========================================================================
    # PART (B): PROPOSED (DETACHED SOFT-OR GATING) - BOTTOM HALF
    # =========================================================================
    # Section Header
    ax.text(50, 45.5, "(b) Proposed GDF-Net (Detached Soft-OR Gating)",
            fontsize=17, fontweight='bold', color='#1A1A1A', ha='center', va='center')

    ax.text(50, 42.8, "The Feedback Firewall mathematically severs recursive error propagation while restoring healthy, uncertainty-guided gradient flow",
            fontsize=11.5, fontstyle='italic', color='#5F6368', ha='center', va='center')

    # 1. Top Central Box: Backbone fmask (Centered above Soft-OR Gate at x=57)
    # Box center: (57, 30.5), width: 26, height: 7.5
    draw_box(44, 27.0, 26, 7.5, c_blue_fill, c_blue_edge, lw=2.2)
    ax.text(57, 32.2, "Backbone", **font_box_header)
    ax.text(57, 29.0, r"$\mathbf{f}_{\mathrm{mask}}$ (Internal Attention)", fontsize=12.5, fontweight='bold', color=c_blue_edge, ha='center', va='center')

    # 2. Bottom Row: Recurrent | [detach() Barrier] | Soft-OR Gate | Output Feature
    # Recurrent m_fg: X from 8 to 23 (width 15), Y from 12.5 to 20.5 (height 8)
    draw_box(8, 12.5, 15, 8, c_orange_fill, c_orange_edge, lw=2.2)
    ax.text(15.5, 17.7, "Recurrent Prior", **font_box_header)
    ax.text(15.5, 14.7, r"$m_{\mathrm{fg}} = \mathcal{D}(m^{(t-1)})$", fontsize=12, fontweight='bold', color='#B06000', ha='center', va='center')

    # detach() Barrier Box: X from 26 to 37 (width 11)
    draw_box(26, 12.5, 11, 8, '#FFF5F5', c_red_edge, lw=2.0, ls='--')
    ax.text(31.5, 17.7, r"$\mathtt{detach()}$", fontsize=12.5, fontweight='bold', color=c_text_red, ha='center', va='center')
    ax.text(31.5, 14.7, "Severed", fontsize=11.5, fontweight='bold', color=c_text_red, ha='center', va='center')

    # Soft-OR Gate: X from 41 to 73 (width 32)
    draw_box(41, 12.5, 32, 8, c_green_fill, c_green_edge, lw=2.2)
    ax.text(57, 17.7, "Soft-OR Gating Module", fontsize=13.5, fontweight='bold', color=c_text_green, ha='center', va='center')
    ax.text(57, 14.7, r"$1.0 - (1.0 - \mathbf{f}_{\mathrm{mask}})(1.0 - m_{\mathrm{fg}})$", fontsize=12.5, fontweight='bold', color=c_text_green, ha='center', va='center')

    # Output Feature: X from 76 to 92 (width 16)
    draw_box(76, 12.5, 16, 8, c_white_fill, c_gray_edge, lw=2.2)
    ax.text(84, 17.7, "Output Feature", **font_box_header)
    ax.text(84, 14.7, r"$\mathbf{F}_{\mathrm{gated}}$", fontsize=13, fontweight='bold', color='#3C4043', ha='center', va='center')

    # --- Arrows and Paths in Part (b) ---
    # Forward path: Recurrent -> detach() -> Soft-OR Gate
    arrow_rec_det = FancyArrowPatch((23, 16.5), (26, 16.5),
                                    arrowstyle='->,head_width=3.5,head_length=5',
                                    color='#202124', linewidth=1.8, zorder=4)
    ax.add_patch(arrow_rec_det)

    arrow_det_gate = FancyArrowPatch((37, 16.5), (41, 16.5),
                                     arrowstyle='->,head_width=3.5,head_length=5',
                                     color='#202124', linewidth=1.8, zorder=4)
    ax.add_patch(arrow_det_gate)

    # Red severed barrier cross mark between detach and Soft-OR Gate
    ax.plot([38.2, 39.8], [15.2, 17.8], color=c_red_edge, linewidth=3.0, zorder=5)
    ax.plot([38.2, 39.8], [17.8, 15.2], color=c_red_edge, linewidth=3.0, zorder=5)

    # Red dashed arrow terminating at detach barrier (backward gradient blocked!)
    arrow_det_bwd = FancyArrowPatch((41, 14.5), (39.8, 14.5),
                                    arrowstyle='->,head_width=3.5,head_length=5',
                                    color=c_red_edge, linewidth=1.8, linestyle='--', zorder=4)
    ax.add_patch(arrow_det_bwd)
    ax.text(39.0, 9.8, "Blocked at Firewall", fontsize=10.5, fontweight='bold', color=c_text_red, ha='center', va='center')

    # Soft-OR Gate to Output Feature (Forward)
    arrow_gate_out_b = FancyArrowPatch((73, 17.0), (76, 17.0),
                                       arrowstyle='->,head_width=3.5,head_length=5',
                                       color='#202124', linewidth=1.8, zorder=4)
    ax.add_patch(arrow_gate_out_b)

    # Output Feature to Soft-OR Gate (Backward gradient)
    arrow_out_gate_b = FancyArrowPatch((76, 15.0), (73, 15.0),
                                       arrowstyle='->,head_width=3.5,head_length=5',
                                       color=c_green_edge, linewidth=1.8, linestyle='--', zorder=4)
    ax.add_patch(arrow_out_gate_b)

    # Dual connection between Backbone fmask and Soft-OR Gate:
    # Centered and aligned neatly between x=54 and x=60
    # 1. Forward arrow (downwards, solid black)
    arrow_fwd_b = FancyArrowPatch((52.5, 27.0), (52.5, 20.8),
                                  arrowstyle='->,head_width=4,head_length=6',
                                  color='#202124', linewidth=2.2, zorder=4)
    ax.add_patch(arrow_fwd_b)
    ax.text(50.5, 23.9, "Forward", fontsize=11, fontweight='bold', color='#202124', ha='right', va='center')

    # 2. Healthy Flow gradient arrow (upwards, green dashed)
    arrow_healthy = FancyArrowPatch((59.5, 20.8), (59.5, 27.0),
                                    arrowstyle='->,head_width=4.5,head_length=6.5',
                                    color=c_green_edge, linewidth=2.4, linestyle='--', zorder=4)
    ax.add_patch(arrow_healthy)
    ax.text(62.0, 23.9, r"Healthy Flow: $\nabla \propto (1.0 - m_{\mathrm{fg}})$",
            fontsize=12, fontweight='bold', color=c_text_green,
            bbox=dict(boxstyle="square,pad=0.25", fc="#F0FFF4", ec=c_green_edge, lw=1.2),
            ha='left', va='center')

    # Upward green dashed arrow from below into Soft-OR Gate
    arrow_up_green = FancyArrowPatch((57, 4.8), (57, 12.2),
                                     arrowstyle='->,head_width=4,head_length=6',
                                     color=c_green_edge, linewidth=2.2, linestyle='--', zorder=4)
    ax.add_patch(arrow_up_green)
    ax.text(57, 2.8, "Supervision from Downstream Segmentation Loss",
            fontsize=11.5, fontweight='bold', color=c_text_green, ha='center', va='center')

    # Save outputs
    output_png = 'papers/overleaf_submission/figures/architecture_diagram.png'
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Successfully generated 2D scientific architecture diagram at: {output_png}")

if __name__ == '__main__':
    create_diagram()
