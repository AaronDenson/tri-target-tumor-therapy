"""
nanocarrier_cascade.py
======================
High-Resolution Visual Cascade for T4 Triple-Lock Nanocarrier
ISEF / SEFH / TXSEF Poster Ready
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_nanocarrier_cascade(save_path="nanocarrier_cascade.png", figsize=(12,8)):
    """
    Draws a colorful cascade diagram for the T4 triple-lock nanocarrier mechanism:
    Stage 1: EGFRvIII-targeting
    Stage 2: MMP-2-cleavable shell removal
    Stage 3: miR-210-triggered siRNA release
    Stage 4: STAT3 silencing
    """

    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Color palette for stages
    stage_colors = ["#FF6F61", "#6B5B95", "#88B04B", "#F7CAC9"]
    text_colors = ["white", "white", "white", "black"]

    # Stage positions
    stages = ["Stage 1:\nEGFRvIII Binding",
              "Stage 2:\nMMP-2 Cleavage",
              "Stage 3:\nmiR-210 Activation",
              "Stage 4:\nSTAT3 Silencing"]

    y_positions = [7, 5, 3, 1.2]

    # Draw stages as rounded rectangles
    for i, (stage, color, text_color, y) in enumerate(zip(stages, stage_colors, text_colors, y_positions)):
        rect = patches.FancyBboxPatch(
            (2, y), width=8, height=1.2,
            boxstyle="round,pad=0.3",
            facecolor=color,
            edgecolor="black",
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(6, y + 0.6, stage, ha='center', va='center', fontsize=16, fontweight='bold', color=text_color)

    # Draw arrows connecting stages
    for y_start, y_end in zip(y_positions[:-1], y_positions[1:]):
        ax.annotate("",
                    xy=(6, y_end + 1.2), xycoords='data',
                    xytext=(6, y_start), textcoords='data',
                    arrowprops=dict(arrowstyle="->,head_width=0.5,head_length=0.8", color='black', linewidth=2))

    # Extra graphics: small icons representing each trigger
    # Stage 1: small "antenna" for aptamer binding
    ax.plot([3.5], [7.5], marker='o', markersize=15, color="#FFD700", label="EGFRvIII Aptamer")
    # Stage 2: scissors icon for MMP-2
    ax.text(3.5, 5.5, "\u2702", fontsize=28, color="#FFFFFF", ha='center', va='center')
    # Stage 3: RNA helix for miR-210
    ax.text(8.5, 3.5, "\u27E8\u1D49\u27E9", fontsize=22, color="#FFFFFF", ha='center', va='center')
    # Stage 4: siRNA/knockdown
    ax.plot([6], [1.8], marker='*', markersize=20, color="#FF4500", label="STAT3 siRNA Released")

    # Poster title
    ax.text(6, 8.8, "T4 Triple-Lock Nanocarrier Cascade", ha='center', va='center', fontsize=20, fontweight='bold')

    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Nanocarrier cascade diagram saved at: {save_path}")
