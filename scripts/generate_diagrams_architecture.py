"""
Generate publication-grade 2D scientific architecture diagrams using mingrammer/diagrams (Diagram as Code):
Panel (a): Baseline Architecture (Hard Gating / Feedback Trap)
Panel (b): Proposed GDF-Net Architecture (Detached Soft-OR Gating)
Combines them into papers/overleaf_submission/figures/architecture_diagram.png
"""

import os
from PIL import Image
from diagrams import Diagram, Cluster, Edge
from diagrams.c4 import Node

def generate_panel_a():
    graph_attr = {
        "bgcolor": "white",
        "pad": "0.6",
        "nodesep": "1.0",
        "ranksep": "0.9",
        "fontsize": "15",
        "fontname": "Helvetica-Bold",
        "fontcolor": "#1A1A1A",
        "splines": "spline",
        "dpi": "300",
    }
    out_file = "papers/overleaf_submission/figures/architecture_baseline_diagrams"
    with Diagram(
        "(a) Baseline Architecture (Hard Gating / Feedback Trap)",
        filename=out_file,
        direction="TB",
        show=False,
        graph_attr=graph_attr,
    ):
        bb = Node(
            "Backbone\nf_mask (Internal Attention)",
            shape="box",
            style="rounded,filled",
            fillcolor="#E8F0FE",
            color="#1A73E8",
            fontcolor="#1A73E8",
            fixedsize="false",
            labelloc="c",
            penwidth="2.2",
            fontsize="11",
            fontname="Helvetica-Bold",
        )
        with Cluster("", graph_attr={"rank": "same", "style": "invis"}):
            rec = Node(
                "Recurrent Prior\nm_fg = D(m^(t-1))",
                shape="box",
                style="rounded,filled",
                fillcolor="#FEF3E2",
                color="#EA8600",
                fontcolor="#B06000",
                fixedsize="false",
                labelloc="c",
                penwidth="2.2",
                fontsize="11",
                fontname="Helvetica-Bold",
            )
            gate = Node(
                "Hard Gate Operator\nmax( Indicator(f_mask > 0.5), m_fg )",
                shape="box",
                style="rounded,filled",
                fillcolor="#FCE8E6",
                color="#D93025",
                fontcolor="#C5221F",
                fixedsize="false",
                labelloc="c",
                penwidth="2.2",
                fontsize="11",
                fontname="Helvetica-Bold",
            )
            out = Node(
                "Output Feature\nF_gated",
                shape="box",
                style="rounded,filled",
                fillcolor="#FFFFFF",
                color="#3C4043",
                fontcolor="#202124",
                fixedsize="false",
                labelloc="c",
                penwidth="2.2",
                fontsize="11",
                fontname="Helvetica-Bold",
            )

        bot_anchor = Node(
            "Recursive Error Amplification Loop (Backward Pass)",
            shape="plaintext",
            fontcolor="#D93025",
            fontname="Helvetica-Bold",
            fontsize="11",
        )

        bb >> Edge(label=" Forward", color="#202124", penwidth="2.0", fontname="Helvetica-Bold", fontsize="10") >> gate
        gate >> Edge(label=" grad = 0 (Blocked)", color="#D93025", style="dashed", constraint="false", penwidth="2.2", fontname="Helvetica-Bold", fontsize="10") >> bb
        rec >> Edge(label=" Toxic Leakage", color="#D93025", style="dashed", constraint="false", penwidth="2.0", fontname="Helvetica-Bold", fontsize="10") >> gate
        gate >> Edge(label=" Forward", color="#202124", constraint="false", penwidth="1.8", fontname="Helvetica-Bold", fontsize="10") >> out
        gate >> Edge(style="invis") >> bot_anchor
        out >> Edge(color="#D93025", style="dashed", penwidth="2.0", constraint="false") >> bot_anchor >> Edge(color="#D93025", style="dashed", penwidth="2.0", constraint="false") >> rec

    print(f"Generated {out_file}.png")


def generate_panel_b():
    graph_attr = {
        "bgcolor": "white",
        "pad": "0.6",
        "nodesep": "0.9",
        "ranksep": "0.9",
        "fontsize": "15",
        "fontname": "Helvetica-Bold",
        "fontcolor": "#1A1A1A",
        "splines": "spline",
        "dpi": "300",
    }
    out_file = "papers/overleaf_submission/figures/architecture_proposed_diagrams"
    with Diagram(
        "(b) Proposed GDF-Net Architecture (Detached Soft-OR Gating)",
        filename=out_file,
        direction="TB",
        show=False,
        graph_attr=graph_attr,
    ):
        bb = Node(
            "Backbone\nf_mask (Internal Attention)",
            shape="box",
            style="rounded,filled",
            fillcolor="#E8F0FE",
            color="#1A73E8",
            fontcolor="#1A73E8",
            fixedsize="false",
            labelloc="c",
            penwidth="2.2",
            fontsize="11",
            fontname="Helvetica-Bold",
        )

        with Cluster("", graph_attr={"rank": "same", "style": "invis"}):
            rec = Node(
                "Recurrent Prior\nm_fg = D(m^(t-1))",
                shape="box",
                style="rounded,filled",
                fillcolor="#FEF3E2",
                color="#EA8600",
                fontcolor="#B06000",
                fixedsize="false",
                labelloc="c",
                penwidth="2.2",
                fontsize="11",
                fontname="Helvetica-Bold",
            )
            detach_box = Node(
                "Feedback Firewall\ndetach() [Severed X]",
                shape="box",
                style="rounded,filled,dashed",
                fillcolor="#FFF5F5",
                color="#D93025",
                fontcolor="#C5221F",
                fixedsize="false",
                labelloc="c",
                penwidth="2.2",
                fontsize="11",
                fontname="Helvetica-Bold",
            )
            gate = Node(
                "Soft-OR Gating Module\n1.0 - (1.0 - f_mask)(1.0 - m_fg)",
                shape="box",
                style="rounded,filled",
                fillcolor="#E6F4EA",
                color="#137333",
                fontcolor="#0D652D",
                fixedsize="false",
                labelloc="c",
                penwidth="2.2",
                fontsize="11",
                fontname="Helvetica-Bold",
            )
            out = Node(
                "Output Feature\nF_gated",
                shape="box",
                style="rounded,filled",
                fillcolor="#FFFFFF",
                color="#3C4043",
                fontcolor="#202124",
                fixedsize="false",
                labelloc="c",
                penwidth="2.2",
                fontsize="11",
                fontname="Helvetica-Bold",
            )

        loss_node = Node(
            "Supervision from Downstream Loss\n(Dice + BCE Objective)",
            shape="box",
            style="rounded,filled",
            fillcolor="#F1F8E9",
            color="#558B2F",
            fontcolor="#33691E",
            fixedsize="false",
            labelloc="c",
            penwidth="1.8",
            fontsize="11",
            fontname="Helvetica-Bold",
        )

        bb >> Edge(label=" Forward", color="#202124", penwidth="2.0", fontname="Helvetica-Bold", fontsize="10") >> gate
        gate >> Edge(label=" Healthy Flow: grad ~ (1.0 - m_fg)", color="#137333", style="dashed", constraint="false", penwidth="2.2", fontname="Helvetica-Bold", fontsize="10") >> bb
        rec >> Edge(label=" Forward", color="#202124", constraint="false", penwidth="1.8", fontname="Helvetica-Bold", fontsize="10") >> detach_box
        detach_box >> Edge(label=" Decoupled Prior", color="#202124", constraint="false", penwidth="1.8", fontname="Helvetica-Bold", fontsize="10") >> gate
        gate >> Edge(label=" Blocked at Firewall", color="#D93025", style="dashed", constraint="false", penwidth="2.0", fontname="Helvetica-Bold", fontsize="10") >> detach_box
        gate >> Edge(label=" Forward", color="#202124", constraint="false", penwidth="1.8", fontname="Helvetica-Bold", fontsize="10") >> out
        gate >> Edge(style="invis") >> loss_node
        loss_node >> Edge(label=" Direct Gradient Supervision", color="#137333", style="dashed", penwidth="2.0", constraint="false", fontname="Helvetica-Bold", fontsize="10") >> gate

    print(f"Generated {out_file}.png")


def combine_diagrams():
    img_a_path = "papers/overleaf_submission/figures/architecture_baseline_diagrams.png"
    img_b_path = "papers/overleaf_submission/figures/architecture_proposed_diagrams.png"
    if not (os.path.exists(img_a_path) and os.path.exists(img_b_path)):
        print("Missing panel images, cannot combine.")
        return

    im_a = Image.open(img_a_path)
    im_b = Image.open(img_b_path)

    max_w = max(im_a.width, im_b.width)
    gap = 60
    total_h = im_a.height + im_b.height + gap

    combined = Image.new("RGB", (max_w, total_h), (255, 255, 255))
    offset_a_x = (max_w - im_a.width) // 2
    combined.paste(im_a, (offset_a_x, 0))

    offset_b_x = (max_w - im_b.width) // 2
    combined.paste(im_b, (offset_b_x, im_a.height + gap))

    out_combined = "papers/overleaf_submission/figures/architecture_diagram.png"
    combined.save(out_combined, dpi=(300, 300))
    print(f"Saved combined 2D scientific architecture diagram at {out_combined}")


if __name__ == "__main__":
    generate_panel_a()
    generate_panel_b()
    combine_diagrams()
