"""
Script: generate_architec.py
Purpose: Render publication-quality high-resolution (300 DPI) architectural comparison
         diagram between Baseline (Hard Gating / Feedback Trap) and Proposed GDF-Net
         (Pre-trained ResNet-34 + Decoupled MFAD Decoder).
Output: architec.png
"""

import os
from graphviz import Digraph


def generate_architecture_diagram(output_filename="architec"):
    dot = Digraph(
        name="GDFNet_vs_Baseline",
        format="png",
        comment="Architecture Comparison: Baseline vs GDF-Net",
    )

    # Global Graph Attributes
    dot.attr(
        dpi="300",
        rankdir="TB",
        bgcolor="#FFFFFF",
        compound="true",
        nodesep="0.45",
        ranksep="0.45",
        fontname="Helvetica",
    )

    # Default Node Styling
    dot.attr(
        "node",
        shape="box",
        style="filled,rounded",
        fontname="Helvetica",
        fontsize="10",
        margin="0.16,0.10",
        penwidth="1.3",
    )

    # Default Edge Styling
    dot.attr(
        "edge",
        fontname="Helvetica",
        fontsize="9",
        penwidth="1.4",
        color="#37474F",
        arrowsize="0.85",
    )

    # =========================================================================
    # (a) BASELINE ARCHITECTURE (HARD GATING / COUPLED FEEDBACK TRAP)
    # =========================================================================
    with dot.subgraph(name="cluster_baseline") as a:
        a.attr(
            label="(a) Baseline Architecture (Scratch CNN + Coupled Hard-Gating Feedback Trap)",
            fontname="Helvetica-Bold",
            fontsize="13",
            style="rounded,filled",
            color="#CFD8DC",
            bgcolor="#FAFAFA",
            margin="16",
        )

        a.node(
            "a_in_img",
            "Input Image I\n[3 x H x W]",
            fillcolor="#E1F5FE",
            color="#0288D1",
        )
        a.node(
            "a_in_mask",
            "Recurrent Prior m^(t-1)\n[1 x H x W]\n(Coupled Backward Graph)",
            fillcolor="#FFE0B2",
            color="#F57C00",
        )

        a.node(
            "a_enc",
            "Custom CNN Encoder (Scratch)\n"
            "----------------------------------------\n"
            "E1: Conv3x3 + SE (32 ch, H x W)\n"
            "E2: Conv3x3 + SE (64 ch, H/2 x W/2)\n"
            "E3: Conv3x3 + SE (128 ch, H/4 x W/4)\n"
            "E4: Conv3x3 + SE (256 ch, H/8 x W/8)\n"
            "[Trained from Scratch; No Pretrained Weights]",
            fillcolor="#E3F2FD",
            color="#1976D2",
        )

        a.node(
            "a_mixpool",
            "MixPool Module (Fatal Pathologies)\n"
            "----------------------------------------\n"
            "• Internal Attention: f_mask = Sigmoid(Conv(x))\n"
            "• Hard Gating: keep = max( I(f_mask > 0.5), m_fg )\n"
            "----------------------------------------\n"
            "[!] Gradient Paralysis: grad(f_mask) = 0\n"
            "[!] Toxic Leakage: Error Loops back into m",
            fillcolor="#FFEBEE",
            color="#C62828",
            penwidth="1.8",
        )

        a.node(
            "a_dec",
            "Custom CNN Decoder\n"
            "----------------------------------------\n"
            "D1: ConvTranspose (128 ch, H/4 x W/4)\n"
            "D2: ConvTranspose (64 ch, H/2 x W/2)\n"
            "D3: ConvTranspose (32 ch, H x W)\n"
            "D4: ConvTranspose (16 ch, H x W)\n"
            "Output Head: Conv1x1 (17 -> 1)",
            fillcolor="#ECEFF1",
            color="#546E7A",
        )

        a.node(
            "a_out",
            "Prediction Output y^(t)\n[1 x H x W]\n(Irreversible FP Error Drift)",
            fillcolor="#FFCDD2",
            color="#B71C1C",
        )

        # Baseline Forward Flows
        a.edge("a_in_img", "a_enc", label="Forward", color="#1565C0")
        a.edge("a_in_mask", "a_mixpool", label="Prior m_fg", color="#E65100")
        a.edge("a_enc", "a_mixpool", label="Features x", color="#1565C0")
        a.edge("a_mixpool", "a_dec", label="Gated Features", color="#C62828")
        a.edge("a_enc", "a_dec", label="Skip Connections", style="dashed", color="#78909C")
        a.edge("a_dec", "a_out", label="Logits", color="#37474F")

        # Coupled Feedback Trap Loop
        a.edge(
            "a_out",
            "a_in_mask",
            label="Coupled Feedback Loop (t=1 -> t=2)\n[Feedback Trap: Toxic Gradient & Error Expansion]",
            style="dashed",
            color="#D32F2F",
            penwidth="2.0",
            fontcolor="#B71C1C",
        )

    # =========================================================================
    # (b) PROPOSED GDF-NET (PRE-TRAINED RESNET-34 + MFAD + FIREWALL)
    # =========================================================================
    with dot.subgraph(name="cluster_proposed") as b:
        b.attr(
            label="(b) Proposed GDF-Net Architecture (Pre-trained ResNet-34 + Feedback Firewall + Decoupled MFAD)",
            fontname="Helvetica-Bold",
            fontsize="13",
            style="rounded,filled",
            color="#C8E6C9",
            bgcolor="#F1F8E9",
            margin="16",
        )

        b.node(
            "b_in_img",
            "Input Image I\n[3 x H x W]",
            fillcolor="#E1F5FE",
            color="#0288D1",
        )
        b.node(
            "b_in_mask",
            "Recurrent Prior m^(t-1)\n[1 x H x W]",
            fillcolor="#FFE0B2",
            color="#F57C00",
        )

        b.node(
            "b_firewall",
            "Feedback Firewall\n"
            "----------------------------------------\n"
            "m_detached = m^(t-1).detach()\n"
            "[OK] Severed Graph: grad(m^(t-1)) = 0\n"
            "Eliminates Representation Collapse & BN Drift",
            fillcolor="#FFF3E0",
            color="#EF6C00",
            penwidth="1.8",
        )

        b.node(
            "b_gate",
            "Learned Spatial Gate\n"
            "----------------------------------------\n"
            "x_guide = Sigmoid(Conv1x1(m_detached))\n"
            "[Learned Dynamic Pruning of Past Errors]",
            fillcolor="#FFF8E1",
            color="#FFA000",
        )

        b.node(
            "b_stem",
            "4-Channel Input Stem\n"
            "----------------------------------------\n"
            "x_in = [ I_RGB || x_guide ] (4 channels)\n"
            "Conv7x7 (4 -> 64, stride=2)\n"
            "Ch 0..2: ImageNet Pre-trained Weights\n"
            "Ch 3: Zero-Initialized (Stable Warm-up)",
            fillcolor="#E0F2F1",
            color="#00897B",
        )

        b.node(
            "b_resnet",
            "Pre-trained ResNet-34 Encoder Backbone\n"
            "----------------------------------------\n"
            "s1: Stem Conv + BN + ReLU (64 ch, H/2 x W/2)\n"
            "s2: Layer 1 (3 BasicBlocks, 64 ch, H/4 x W/4)\n"
            "s3: Layer 2 (4 BasicBlocks, 128 ch, H/8 x W/8)\n"
            "s4: Layer 3 (6 BasicBlocks, 256 ch, H/16 x W/16)\n"
            "b : Layer 4 (3 BasicBlocks, 512 ch, H/32 x W/32)\n"
            "[Transfer Learning from ImageNet-1K]",
            fillcolor="#E8EAF6",
            color="#3949AB",
            penwidth="1.6",
        )

        b.node(
            "b_mfad",
            "Multi-Scale Decoupled Decoder (MFAD)\n"
            "----------------------------------------\n"
            "d1: ConvTranspose + Skip s4 (256 ch, H/16 x W/16)\n"
            "d2: ConvTranspose + Skip s3 (128 ch, H/8 x W/8)\n"
            "d3: ConvTranspose + Skip s2 (64 ch, H/4 x W/4)\n"
            "d4: ConvTranspose + Skip s1 (64 ch, H/2 x W/2)\n"
            "d5: ConvTranspose + ConvBlock (32 ch, H x W)\n"
            "----------------------------------------\n"
            "Continuous Modulation at All 5 Resolutions:\n"
            "F_guided = F * ( 1.0 + Att_layer(m_guide) )\n"
            "[Dense, Non-Zero Gradient Backpropagation]",
            fillcolor="#E8F5E9",
            color="#2E7D32",
            penwidth="1.8",
        )

        b.node(
            "b_head",
            "Output Projection Head\n"
            "Conv1x1 (32 -> 1)\n"
            "Full-Resolution Logits",
            fillcolor="#C8E6C9",
            color="#1B5E20",
        )

        b.node(
            "b_out",
            "Refined Prediction y^(t)\n[1 x H x W]\n(Sharp Boundaries, 65% Lower FPR)",
            fillcolor="#A5D6A7",
            color="#1B5E20",
        )

        # Proposed Forward Flows
        b.edge("b_in_mask", "b_firewall", label="Feedback Prior", color="#E65100")
        b.edge("b_firewall", "b_gate", label="Detached", color="#E65100")
        b.edge("b_in_img", "b_stem", label="RGB Input", color="#0288D1")
        b.edge("b_gate", "b_stem", label="Spatial Gate", color="#FFA000")
        b.edge("b_stem", "b_resnet", label="Hierarchical Features", color="#3949AB")
        b.edge("b_resnet", "b_mfad", label="Bottleneck (b) & Skips (s1..s4)", color="#3949AB")
        b.edge(
            "b_firewall",
            "b_mfad",
            label="Multi-scale Spatial Prior Guidance",
            style="dashed",
            color="#2E7D32",
            penwidth="1.4",
        )
        b.edge("b_mfad", "b_head", label="Decoder d5", color="#2E7D32")
        b.edge("b_head", "b_out", label="Sigmoid", color="#1B5E20")

        # Decoupled Recurrent Loop
        b.edge(
            "b_out",
            "b_in_mask",
            label="Decoupled Recurrent Loop (t=1 -> t=2)\n[Guidance Only; Zero Backward Gradient via .detach()]",
            style="dashed",
            color="#2E7D32",
            penwidth="2.0",
            fontcolor="#1B5E20",
        )

    # Render directly to architec.png with cleanup
    rendered_path = dot.render(output_filename, cleanup=True)
    print(f"Publication-ready diagram successfully created: {rendered_path}")
    return rendered_path


if __name__ == "__main__":
    generate_architecture_diagram("architec")
