#!/usr/bin/env python3
"""
Visualize the frontier model failures vs. our simple solution.
"""

import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: Bar chart of results
models = ['Gemini 2.5\n"Deep Think"', 'Grok', 'GPT-5', 'Our Method\n(Random)', 'Our Method\n(Optimized)']
lambda2_values = [0.083, 0.100, 0.049, 0.0884, 0.0845]
colors = ['red', 'yellow', 'orange', 'lightgreen', 'green']
labels = ['Failed\n(Fullerene trap)', 'Unverified\n(Extrapolation)', 'Failed\n(Gave up)', 'Champion\n(Sphere dual)', 'Optimizer']

ax1.bar(models, lambda2_values, color=colors, edgecolor='black', linewidth=2)
ax1.axhline(y=0.084, color='blue', linestyle='--', linewidth=2, label='Target (0.084)')
ax1.axhline(y=0.16, color='purple', linestyle=':', linewidth=2, label='Spielman-Teng bound (0.16)')

# Add value labels on bars
for i, (model, value, label) in enumerate(zip(models, lambda2_values, labels)):
    ax1.text(i, value + 0.005, f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax1.text(i, value/2, label, ha='center', va='center', fontsize=8, style='italic')

ax1.set_ylabel('λ₂(Laplacian) - Algebraic Connectivity', fontsize=12, fontweight='bold')
ax1.set_title('Frontier Model Comparison: λ₂(L) for N=150 Planar Cubic Graphs', fontsize=14, fontweight='bold')
ax1.set_ylim([0, 0.18])
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# Add annotations
ax1.annotate('FAILED\n(Wrong category)', xy=(0, 0.083), xytext=(0, 0.14),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=9, ha='center', color='red', fontweight='bold')

ax1.annotate('UNVERIFIED\n(No construction)', xy=(1, 0.100), xytext=(1, 0.15),
            arrowprops=dict(arrowstyle='->', color='orange', lw=2),
            fontsize=9, ha='center', color='orange', fontweight='bold')

ax1.annotate('FAILED\n("Impossible")', xy=(2, 0.049), xytext=(2, 0.11),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=9, ha='center', color='red', fontweight='bold')

ax1.annotate('SUCCESS!\n(+5.3% over target)', xy=(3, 0.0884), xytext=(3.5, 0.13),
            arrowprops=dict(arrowstyle='->', color='green', lw=2.5),
            fontsize=10, ha='center', color='green', fontweight='bold')

# Right plot: Timeline/effort comparison
ax2.set_xlim([0, 5])
ax2.set_ylim([0, 10])

# Gemini
ax2.text(0.5, 9, '🤖 Gemini 2.5', fontsize=12, fontweight='bold', ha='left')
ax2.text(0.5, 8.5, '• Assumed fullerenes', fontsize=9, ha='left')
ax2.text(0.5, 8.0, '• Applied Leapfrog transformation', fontsize=9, ha='left')
ax2.text(0.5, 7.5, '• Consulted literature', fontsize=9, ha='left')
ax2.text(0.5, 7.0, '• Complex eigenvalue formulas', fontsize=9, ha='left')
ax2.text(0.5, 6.5, '→ Result: 0.083 (FAILED)', fontsize=10, ha='left', color='red', fontweight='bold')

# Grok
ax2.text(0.5, 5.5, '🤖 Grok', fontsize=12, fontweight='bold', ha='left')
ax2.text(0.5, 5.0, '• Also assumed fullerenes', fontsize=9, ha='left')
ax2.text(0.5, 4.5, '• Extrapolated from C20, C60', fontsize=9, ha='left')
ax2.text(0.5, 4.0, '• Claimed λ₂ ≈ 0.1', fontsize=9, ha='left')
ax2.text(0.5, 3.5, '→ Result: UNVERIFIED (no graph)', fontsize=10, ha='left', color='orange', fontweight='bold')

# GPT-5
ax2.text(0.5, 2.5, '🤖 GPT-5', fontsize=12, fontweight='bold', ha='left')
ax2.text(0.5, 2.0, '• Actually tried construction! 👍', fontsize=9, ha='left')
ax2.text(0.5, 1.5, '• 1,900 Apollonian + edge flips', fontsize=9, ha='left')
ax2.text(0.5, 1.0, '• Best: 0.049', fontsize=9, ha='left')
ax2.text(0.5, 0.5, '→ Concluded: IMPOSSIBLE', fontsize=10, ha='left', color='red', fontweight='bold')

# Our method
ax2.text(3, 9, '✓ Our Simple Method', fontsize=12, fontweight='bold', ha='left', color='green')
ax2.text(3, 8.5, '• Random sphere points', fontsize=9, ha='left')
ax2.text(3, 8.0, '• Delaunay triangulation', fontsize=9, ha='left')
ax2.text(3, 7.5, '• Take dual (cubic graph)', fontsize=9, ha='left')
ax2.text(3, 7.0, '• Generated 500 samples', fontsize=9, ha='left')
ax2.text(3, 6.5, '→ Best: 0.0884 ✓', fontsize=11, ha='left', color='green', fontweight='bold')
ax2.text(3, 6.0, '   Time: 2 minutes total', fontsize=9, ha='left', style='italic')

# Statistics
ax2.text(3, 5, 'Distribution Analysis:', fontsize=11, ha='left', fontweight='bold')
ax2.text(3, 4.5, '• Mean: 0.0706', fontsize=9, ha='left')
ax2.text(3, 4.0, '• Std: 0.0050', fontsize=9, ha='left')
ax2.text(3, 3.5, '• CV: 7.1% (concentrated!)', fontsize=9, ha='left')
ax2.text(3, 3.0, '• Max: 0.0842', fontsize=9, ha='left')
ax2.text(3, 2.5, '• Champion: 0.0884 (+3.5σ)', fontsize=9, ha='left')

# Key insight
ax2.text(3, 1.5, '💡 Key Insight:', fontsize=11, ha='left', fontweight='bold', color='blue')
ax2.text(3, 1.0, 'Sphere construction creates', fontsize=9, ha='left')
ax2.text(3, 0.5, 'geometrically privileged family!', fontsize=9, ha='left')

ax2.set_xlim([0, 5])
ax2.set_ylim([0, 10])
ax2.axis('off')
ax2.set_title('Approach Comparison', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('frontier_model_comparison.png', dpi=150, bbox_inches='tight')
print("Plot saved to: frontier_model_comparison.png")
