"""
Visualization functions for planar graphs and their duals.

Since planar graphs can be embedded in 2D, we can create beautiful
visualizations showing the primal triangulation and dual cubic graph.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, Tuple
import numpy as np


def visualize_planar_dual_pair(
    G_tri: nx.Graph,
    G_dual: nx.Graph,
    title: str = "Planar Triangulation and Dual",
    figsize: Tuple[int, int] = (16, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize a planar triangulation and its dual side-by-side.

    Args:
        G_tri: Planar triangulation
        G_dual: Dual cubic graph
        title: Overall figure title
        figsize: Figure size (width, height)
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Get planar layouts
    pos_tri = nx.planar_layout(G_tri)
    pos_dual = nx.planar_layout(G_dual)

    # Plot triangulation
    ax1.set_title(f"Triangulation\n{G_tri.number_of_nodes()} vertices, {G_tri.number_of_edges()} edges",
                  fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Draw triangulation
    nx.draw_networkx_edges(G_tri, pos_tri, ax=ax1, edge_color='gray', width=1.5, alpha=0.6)
    nx.draw_networkx_nodes(G_tri, pos_tri, ax=ax1,
                          node_color='lightblue',
                          node_size=300,
                          edgecolors='darkblue',
                          linewidths=2)
    nx.draw_networkx_labels(G_tri, pos_tri, ax=ax1, font_size=8, font_weight='bold')

    # Plot dual
    ax2.set_title(f"Dual Cubic Graph\n{G_dual.number_of_nodes()} vertices, {G_dual.number_of_edges()} edges",
                  fontsize=14, fontweight='bold')
    ax2.axis('off')

    # Draw dual
    nx.draw_networkx_edges(G_dual, pos_dual, ax=ax2, edge_color='gray', width=2, alpha=0.6)
    nx.draw_networkx_nodes(G_dual, pos_dual, ax=ax2,
                          node_color='lightcoral',
                          node_size=400,
                          edgecolors='darkred',
                          linewidths=2)
    nx.draw_networkx_labels(G_dual, pos_dual, ax=ax2, font_size=8, font_weight='bold')

    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_planar_optimization_comparison(
    G_tri_before: nx.Graph,
    G_dual_before: nx.Graph,
    G_tri_after: nx.Graph,
    G_dual_after: nx.Graph,
    metrics_before: dict,
    metrics_after: dict,
    title: str = "Planar Graph Optimization",
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize before/after optimization for planar graphs.

    Shows triangulation and dual for both initial and optimized graphs.

    Args:
        G_tri_before: Initial triangulation
        G_dual_before: Initial dual
        G_tri_after: Optimized triangulation
        G_dual_after: Optimized dual
        metrics_before: Dict with 'trees' and 'lambda2' for initial
        metrics_after: Dict with 'trees' and 'lambda2' for final
        title: Overall title
        figsize: Figure size
        save_path: Optional save path

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)

    # Before - Triangulation
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title(f"Initial Triangulation\n{G_tri_before.number_of_nodes()} vertices",
                  fontsize=12, fontweight='bold')
    ax1.axis('off')

    pos_tri_before = nx.planar_layout(G_tri_before)
    nx.draw_networkx_edges(G_tri_before, pos_tri_before, ax=ax1, edge_color='gray', width=1.5, alpha=0.6)
    nx.draw_networkx_nodes(G_tri_before, pos_tri_before, ax=ax1,
                          node_color='lightblue', node_size=200,
                          edgecolors='darkblue', linewidths=1.5)

    # Before - Dual
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title(f"Initial Dual Cubic\n{G_dual_before.number_of_nodes()} vertices",
                  fontsize=12, fontweight='bold')
    ax2.axis('off')

    pos_dual_before = nx.planar_layout(G_dual_before)
    nx.draw_networkx_edges(G_dual_before, pos_dual_before, ax=ax2, edge_color='gray', width=2, alpha=0.6)
    nx.draw_networkx_nodes(G_dual_before, pos_dual_before, ax=ax2,
                          node_color='lightcoral', node_size=250,
                          edgecolors='darkred', linewidths=1.5)

    # After - Triangulation
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title(f"Optimized Triangulation\n{G_tri_after.number_of_nodes()} vertices",
                  fontsize=12, fontweight='bold', color='green')
    ax3.axis('off')

    pos_tri_after = nx.planar_layout(G_tri_after)
    nx.draw_networkx_edges(G_tri_after, pos_tri_after, ax=ax3, edge_color='gray', width=1.5, alpha=0.6)
    nx.draw_networkx_nodes(G_tri_after, pos_tri_after, ax=ax3,
                          node_color='lightgreen', node_size=200,
                          edgecolors='darkgreen', linewidths=2)

    # After - Dual
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title(f"Optimized Dual Cubic\n{G_dual_after.number_of_nodes()} vertices",
                  fontsize=12, fontweight='bold', color='green')
    ax4.axis('off')

    pos_dual_after = nx.planar_layout(G_dual_after)
    nx.draw_networkx_edges(G_dual_after, pos_dual_after, ax=ax4, edge_color='gray', width=2, alpha=0.6)
    nx.draw_networkx_nodes(G_dual_after, pos_dual_after, ax=ax4,
                          node_color='#90EE90', node_size=250,
                          edgecolors='darkgreen', linewidths=2)

    # Add metrics text
    metrics_text = (
        f"Before: ln(trees)={metrics_before['trees']:.3f}, λ₂={metrics_before['lambda2']:.3f}\n"
        f"After:  ln(trees)={metrics_after['trees']:.3f}, λ₂={metrics_after['lambda2']:.3f}\n"
        f"Improvement: Δln(trees)={metrics_after['trees']-metrics_before['trees']:+.3f}, "
        f"Δλ₂={metrics_after['lambda2']-metrics_before['lambda2']:+.3f}"
    )

    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_planar_overlaid(
    G_tri: nx.Graph,
    G_dual: nx.Graph,
    title: str = "Planar Triangulation with Dual Overlay",
    figsize: Tuple[int, int] = (12, 12),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize triangulation and dual overlaid on same axes.

    This shows the beautiful duality relationship visually.

    Args:
        G_tri: Planar triangulation
        G_dual: Dual cubic graph
        title: Figure title
        figsize: Figure size
        save_path: Optional save path

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')

    # Get planar layouts
    pos_tri = nx.planar_layout(G_tri)

    # For dual, we'd ideally place vertices at face centroids of triangulation
    # For now, use a separate planar layout (not geometrically exact but illustrative)
    pos_dual = nx.planar_layout(G_dual)

    # Draw triangulation (primal) in blue
    nx.draw_networkx_edges(G_tri, pos_tri, ax=ax,
                          edge_color='blue', width=1.5, alpha=0.4,
                          style='solid', label='Triangulation')
    nx.draw_networkx_nodes(G_tri, pos_tri, ax=ax,
                          node_color='lightblue', node_size=200,
                          edgecolors='darkblue', linewidths=1.5, alpha=0.7)

    # Draw dual in red (offset slightly for visibility)
    # Scale and offset the dual layout
    pos_dual_scaled = {k: (v[0] * 0.8, v[1] * 0.8) for k, v in pos_dual.items()}

    nx.draw_networkx_edges(G_dual, pos_dual_scaled, ax=ax,
                          edge_color='red', width=2.5, alpha=0.5,
                          style='dashed', label='Dual')
    nx.draw_networkx_nodes(G_dual, pos_dual_scaled, ax=ax,
                          node_color='lightcoral', node_size=300,
                          edgecolors='darkred', linewidths=2, alpha=0.7,
                          node_shape='s')  # Square nodes for dual

    # Legend
    blue_patch = mpatches.Patch(color='blue', label=f'Triangulation ({G_tri.number_of_nodes()} vertices)', alpha=0.5)
    red_patch = mpatches.Patch(color='red', label=f'Dual Cubic ({G_dual.number_of_nodes()} vertices)', alpha=0.5)
    ax.legend(handles=[blue_patch, red_patch], loc='upper right', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_alternating_trajectory(
    trajectory: list,
    title: str = "Alternating Optimization Trajectory",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize the trajectory of alternating optimization.

    Shows how spanning trees and λ₂ evolve over cycles.

    Args:
        trajectory: List of dicts with 'cycle', 'trees', 'lambda2'
        title: Figure title
        figsize: Figure size
        save_path: Optional save path

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Extract data
    iterations = list(range(len(trajectory)))
    trees_vals = [t['trees'] for t in trajectory]
    lambda2_vals = [t['lambda2'] for t in trajectory]
    cycles = [t['cycle'] for t in trajectory]

    # Plot spanning trees
    ax1.plot(iterations, trees_vals, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('ln(spanning trees)', fontsize=12, color='blue')
    ax1.set_title('Spanning Trees Evolution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='blue')

    # Mark cycle boundaries
    cycle_changes = [0] + [i for i in range(1, len(cycles)) if cycles[i] != cycles[i-1]] + [len(cycles)-1]
    for i in cycle_changes[1:-1]:
        ax1.axvline(x=i, color='gray', linestyle='--', alpha=0.5)

    # Plot λ₂
    ax2.plot(iterations, lambda2_vals, 's-', linewidth=2, markersize=8, color='red')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('λ₂ (dual)', fontsize=12, color='red')
    ax2.set_title('Expansion (λ₂) Evolution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor='red')

    # Mark cycle boundaries
    for i in cycle_changes[1:-1]:
        ax2.axvline(x=i, color='gray', linestyle='--', alpha=0.5)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
