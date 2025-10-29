"""Gradio web interface for planar cubic graph optimization."""

import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cubic_graph_optimizer.planar.triangulation import (
    random_triangulation_from_sphere,
    triangulation_to_dual_cubic,
    optimize_triangulation,
)
from cubic_graph_optimizer.planar.alternating_planar import alternating_planar_optimization
from cubic_graph_optimizer.planar.visualization import (
    visualize_planar_dual_pair,
    visualize_planar_optimization_comparison,
    visualize_alternating_trajectory,
)
from cubic_graph_optimizer.core.spanning_trees import count_spanning_trees, get_second_eigenvalue
import pynauty


def optimize_planar_single(n_points, objective, max_iterations, seed):
    """Optimize planar graph for single objective."""

    try:
        # Generate initial triangulation
        G_tri_init, _ = random_triangulation_from_sphere(n_points, seed=int(seed))
        G_dual_init = triangulation_to_dual_cubic(G_tri_init)

        initial_trees = count_spanning_trees(G_tri_init)
        initial_lambda2 = get_second_eigenvalue(G_dual_init)

        # Determine what to optimize
        if objective == "Spanning Trees":
            G_tri_opt, final_value, stats = optimize_triangulation(
                G_tri_init,
                count_spanning_trees,
                maximize=True,
                max_iterations=max_iterations,
                verbose=False,
            )
            obj_name = "ln(trees)"
        else:  # Expansion
            # Optimize dual's λ₂
            def dual_lambda2(G_tri):
                return get_second_eigenvalue(triangulation_to_dual_cubic(G_tri))

            G_tri_opt, final_value, stats = optimize_triangulation(
                G_tri_init,
                dual_lambda2,
                maximize=False,
                max_iterations=max_iterations,
                verbose=False,
            )
            obj_name = "λ₂ (dual)"

        G_dual_opt = triangulation_to_dual_cubic(G_tri_opt)

        final_trees = count_spanning_trees(G_tri_opt)
        final_lambda2 = get_second_eigenvalue(G_dual_opt)

        # Analyze automorphisms of dual
        adjacency = {i: list(G_dual_opt.neighbors(i)) for i in range(G_dual_opt.number_of_nodes())}
        g = pynauty.Graph(number_of_vertices=G_dual_opt.number_of_nodes(),
                         directed=False, adjacency_dict=adjacency)
        autgrp = pynauty.autgrp(g)

        # Create visualization
        metrics_before = {'trees': initial_trees, 'lambda2': initial_lambda2}
        metrics_after = {'trees': final_trees, 'lambda2': final_lambda2}

        fig = visualize_planar_optimization_comparison(
            G_tri_init, G_dual_init,
            G_tri_opt, G_dual_opt,
            metrics_before, metrics_after,
            title=f"Planar Graph Optimization - {objective}"
        )

        # Format results
        results = f"""
🎯 **Optimization Results**

**Graph Sizes:**
- Triangulation: {G_tri_opt.number_of_nodes()} vertices, {G_tri_opt.number_of_edges()} edges
- Dual Cubic: {G_dual_opt.number_of_nodes()} vertices, {G_dual_opt.number_of_edges()} edges

**Initial State:**
- ln(spanning trees): {initial_trees:.6f}
- λ₂ (dual): {initial_lambda2:.6f}

**Final State:**
- ln(spanning trees): {final_trees:.6f}
- λ₂ (dual): {final_lambda2:.6f}

**Improvement:**
- Δ ln(trees): {final_trees - initial_trees:+.6f}
- Δ λ₂: {final_lambda2 - initial_lambda2:+.6f}

**Optimization Stats:**
- Iterations: {stats['iterations']}
- Diagonal flips: {stats['flips_performed']}
- Automorphism group (dual): {int(autgrp[1])}

**Optimized {obj_name}: {final_value:.6f}**
"""

        return fig, results

    except Exception as e:
        return None, f"Error: {str(e)}"


def optimize_planar_alternating(n_points, max_cycles, max_iterations_per_phase, seed):
    """Run alternating optimization for planar graphs."""

    try:
        # Generate initial triangulation
        G_tri_init, _ = random_triangulation_from_sphere(n_points, seed=int(seed))
        G_dual_init = triangulation_to_dual_cubic(G_tri_init)

        initial_trees = count_spanning_trees(G_tri_init)
        initial_lambda2 = get_second_eigenvalue(G_dual_init)

        # Run alternating optimization
        G_tri_opt, G_dual_opt, stats = alternating_planar_optimization(
            G_tri_init,
            objectives=['spanning_trees', 'expansion'],
            max_cycles=max_cycles,
            max_iterations_per_phase=max_iterations_per_phase,
            verbose=False,
        )

        final_trees = count_spanning_trees(G_tri_opt)
        final_lambda2 = get_second_eigenvalue(G_dual_opt)

        # Analyze automorphisms
        adjacency = {i: list(G_dual_opt.neighbors(i)) for i in range(G_dual_opt.number_of_nodes())}
        g = pynauty.Graph(number_of_vertices=G_dual_opt.number_of_nodes(),
                         directed=False, adjacency_dict=adjacency)
        autgrp = pynauty.autgrp(g)

        # Create comparison visualization
        metrics_before = {'trees': initial_trees, 'lambda2': initial_lambda2}
        metrics_after = {'trees': final_trees, 'lambda2': final_lambda2}

        fig_comparison = visualize_planar_optimization_comparison(
            G_tri_init, G_dual_init,
            G_tri_opt, G_dual_opt,
            metrics_before, metrics_after,
            title="Planar Alternating Optimization"
        )

        # Create trajectory plot
        fig_trajectory = visualize_alternating_trajectory(
            stats['trajectory'],
            title="Alternating Optimization Trajectory"
        )

        # Format results
        results = f"""
🔄 **Alternating Optimization Results**

**Graph Sizes:**
- Triangulation: {G_tri_opt.number_of_nodes()} vertices
- Dual Cubic: {G_dual_opt.number_of_nodes()} vertices

**Initial:**
- ln(trees): {initial_trees:.6f}
- λ₂ (dual): {initial_lambda2:.6f}

**Final:**
- ln(trees): {final_trees:.6f}
- λ₂ (dual): {final_lambda2:.6f}

**Improvement:**
- Δ ln(trees): {final_trees - initial_trees:+.6f}
- Δ λ₂: {final_lambda2 - initial_lambda2:+.6f}

**Optimization:**
- Cycles completed: {stats['cycles_completed']}
- Converged: {stats['converged']}
- Automorphism group: {int(autgrp[1])}

**Status:** {"✓ Converged to fixed point!" if stats['converged'] else "⟲ Limit cycling"}
"""

        return fig_comparison, fig_trajectory, results

    except Exception as e:
        return None, None, f"Error: {str(e)}"


# Create the Gradio interface
def create_app():
    with gr.Blocks(title="Planar Cubic Graph Optimizer", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🔷 Planar Cubic Graph Optimizer

        Optimize **planar** cubic graphs using diagonal flips on triangulations!

        ## What makes this special?
        - **Planar duality**: Triangulation ↔ Dual cubic graph
        - **Spanning trees**: Equal in primal and dual!
        - **Beautiful visualization**: See the planar embeddings
        - **Alternating optimization**: Like general graphs, but planar!

        ## Method
        1. Generate random points on sphere → convex hull → triangulation
        2. Apply diagonal flips to optimize
        3. Compute dual cubic graph
        4. Track both primal (triangulation) and dual properties
        """)

        with gr.Tabs():
            # Tab 1: Single Objective
            with gr.Tab("📊 Single Objective"):
                gr.Markdown("""
                ### Optimize for Spanning Trees or Expansion

                - **Spanning Trees**: Optimize triangulation (dual improves equally!)
                - **Expansion**: Optimize dual cubic graph's λ₂
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Parameters")

                        n_points_single = gr.Slider(
                            4, 30, value=12, step=1,
                            label="Points on Sphere",
                            info="More points → larger graphs (dual has ~2n-4 vertices)"
                        )

                        objective_single = gr.Radio(
                            choices=["Spanning Trees", "Expansion (λ₂)"],
                            value="Spanning Trees",
                            label="Objective"
                        )

                        max_iter_single = gr.Slider(
                            10, 200, value=50, step=10,
                            label="Max Iterations per Phase"
                        )

                        seed_single = gr.Number(value=42, label="Random Seed")

                        run_single_btn = gr.Button("🚀 Optimize", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        results_single = gr.Textbox(label="Results", lines=20)
                        plot_single = gr.Plot(label="Visualization")

                run_single_btn.click(
                    fn=optimize_planar_single,
                    inputs=[n_points_single, objective_single, max_iter_single, seed_single],
                    outputs=[plot_single, results_single]
                )

                gr.Examples(
                    examples=[
                        [4, "Spanning Trees", 50, 42],
                        [12, "Spanning Trees", 100, 42],  # Should find icosahedron!
                        [12, "Expansion (λ₂)", 100, 42],
                        [22, "Expansion (λ₂)", 100, 42],  # N≈40 dual
                    ],
                    inputs=[n_points_single, objective_single, max_iter_single, seed_single],
                )

            # Tab 2: Alternating Optimization
            with gr.Tab("🔄 Alternating"):
                gr.Markdown("""
                ### Alternating Between Spanning Trees and Expansion

                Just like general cubic graphs, alternating often works better!

                - **N=12**: Should converge to icosahedron (optimal for both!)
                - **Larger N**: Exhibits limit cycling
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Parameters")

                        n_points_alt = gr.Slider(
                            4, 30, value=12, step=1,
                            label="Points on Sphere"
                        )

                        max_cycles_alt = gr.Slider(
                            2, 20, value=10, step=1,
                            label="Max Cycles"
                        )

                        max_iter_phase_alt = gr.Slider(
                            10, 200, value=50, step=10,
                            label="Max Iterations per Phase"
                        )

                        seed_alt = gr.Number(value=42, label="Random Seed")

                        run_alt_btn = gr.Button("🔄 Run Alternating", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        results_alt = gr.Textbox(label="Results", lines=20)
                        plot_alt_comparison = gr.Plot(label="Before/After Comparison")
                        plot_alt_trajectory = gr.Plot(label="Optimization Trajectory")

                run_alt_btn.click(
                    fn=optimize_planar_alternating,
                    inputs=[n_points_alt, max_cycles_alt, max_iter_phase_alt, seed_alt],
                    outputs=[plot_alt_comparison, plot_alt_trajectory, results_alt]
                )

                gr.Examples(
                    examples=[
                        [12, 10, 50, 42],  # Icosahedron!
                        [22, 10, 50, 42],  # N≈40 dual
                        [16, 10, 50, 123],
                    ],
                    inputs=[n_points_alt, max_cycles_alt, max_iter_phase_alt, seed_alt],
                )

        gr.Markdown("""
        ---
        ### 📖 Notes

        - **Planar constraint** limits expansion (λ₂ worse than non-planar)
        - **Spanning trees** nearly as good as non-planar!
        - **Icosahedron** (N=12) is optimal for BOTH objectives
        - **Visualization** shows actual planar embeddings

        Built with [Claude Code](https://claude.com/claude-code)
        """)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False, server_name="0.0.0.0", server_port=7861)
