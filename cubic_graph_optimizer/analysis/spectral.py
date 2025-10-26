"""
Spectral analysis utilities for graphs.
"""

import networkx as nx
import numpy as np


def analyze_eigenvalue_distribution(k_values, samples_per_k=100, include_optimized=True,
                                  optimization_method='greedy', random_seed=42,
                                  plot_histograms=False):
    """
    Analyze the distribution of eigenvalues in random vs optimized cubic graphs.
    Compares to McKay's semicircle law for random regular graphs.
    
    Args:
        k_values: list of k values to test
        samples_per_k: number of random graphs to sample per k
        include_optimized: whether to also analyze optimized graphs
        optimization_method: which method to use for optimization
        random_seed: base random seed
        plot_histograms: whether to plot eigenvalue histograms (requires matplotlib)
    
    Returns:
        dict: eigenvalue statistics indexed by k
    """
    from ..core.spanning_trees import count_spanning_trees
    from ..optimization.methods import optimize_with_restarts
    
    results = {}
    
    for k in k_values:
        n = 2 * k
        print(f"\nAnalyzing k={k} (n={n})...")
        
        # Sample random graphs
        random_eigenvalues = []
        random_max_nontrivial = []
        random_is_ramanujan = []
        random_spanning_trees = []
        
        for i in range(samples_per_k):
            G = nx.random_regular_graph(3, n, seed=random_seed + i)
            eigs = sorted(nx.adjacency_spectrum(G).real, reverse=True)
            
            max_nt = max(abs(eigs[1]), abs(eigs[-1]))
            is_ram = max_nt <= 2 * np.sqrt(2)
            T = count_spanning_trees(G)
            
            random_eigenvalues.append(eigs[1:-1])  # Exclude λ₁=3 and λₙ for bulk
            random_max_nontrivial.append(max_nt)
            random_is_ramanujan.append(is_ram)
            random_spanning_trees.append(T)
        
        ramanujan_rate = np.mean(random_is_ramanujan)
        
        # Flatten bulk eigenvalues for distribution analysis
        bulk_eigs_flat = np.concatenate(random_eigenvalues)
        
        results[k] = {
            'n': n,
            'random_max_nontrivial_mean': np.mean(random_max_nontrivial),
            'random_max_nontrivial_std': np.std(random_max_nontrivial),
            'random_ramanujan_rate': ramanujan_rate,
            'random_spanning_trees_mean': np.mean(random_spanning_trees),
            'random_spanning_trees_std': np.std(random_spanning_trees),
            'random_bulk_mean': np.mean(bulk_eigs_flat),
            'random_bulk_std': np.std(bulk_eigs_flat),
            'random_bulk_eigenvalues': bulk_eigs_flat,  # Store for histogram
            'ramanujan_bound': 2 * np.sqrt(2)
        }
        
        print(f"  Random graphs:")
        print(f"    Ramanujan rate: {ramanujan_rate:.1%}")
        print(f"    Max |λ| (non-trivial): {results[k]['random_max_nontrivial_mean']:.4f} ± {results[k]['random_max_nontrivial_std']:.4f}")
        print(f"    Bulk eigenvalues: μ={results[k]['random_bulk_mean']:.4f}, σ={results[k]['random_bulk_std']:.4f}")
        print(f"    ln(spanning trees): {results[k]['random_spanning_trees_mean']:.4f} ± {results[k]['random_spanning_trees_std']:.4f}")
        
        if include_optimized:
            # Optimize a graph
            print(f"  Optimizing...")
            G_opt, T_opt, _ = optimize_with_restarts(
                n, method=optimization_method, restarts=1,
                base_seed=random_seed + 10000, verbose=False
            )
            
            eigs_opt = sorted(nx.adjacency_spectrum(G_opt).real, reverse=True)
            bulk_opt = eigs_opt[1:-1]  # Exclude extremes
            
            max_nt_opt = max(abs(eigs_opt[1]), abs(eigs_opt[-1]))
            is_ram_opt = max_nt_opt <= 2 * np.sqrt(2)
            
            results[k]['optimized_max_nontrivial'] = max_nt_opt
            results[k]['optimized_is_ramanujan'] = is_ram_opt
            results[k]['optimized_spanning_trees'] = T_opt
            results[k]['optimized_lambda_2'] = eigs_opt[1]
            results[k]['optimized_lambda_n'] = eigs_opt[-1]
            results[k]['optimized_bulk_mean'] = np.mean(bulk_opt)
            results[k]['optimized_bulk_std'] = np.std(bulk_opt)
            results[k]['optimized_bulk_eigenvalues'] = np.array(bulk_opt)
            
            print(f"  Optimized graph:")
            print(f"    Ramanujan: {is_ram_opt}")
            print(f"    Max |λ| (non-trivial): {max_nt_opt:.4f}")
            print(f"    λ₂: {eigs_opt[1]:.4f}, λₙ: {eigs_opt[-1]:.4f}")
            print(f"    Bulk eigenvalues: μ={results[k]['optimized_bulk_mean']:.4f}, σ={results[k]['optimized_bulk_std']:.4f}")
            print(f"    ln(spanning trees): {T_opt:.4f}")
            print(f"    Improvement over random mean: {T_opt - results[k]['random_spanning_trees_mean']:.4f}")
            
            # Check if bulk distribution differs significantly
            bulk_shift = abs(results[k]['optimized_bulk_mean'] - results[k]['random_bulk_mean'])
            bulk_shape_change = abs(results[k]['optimized_bulk_std'] - results[k]['random_bulk_std'])
            print(f"    Bulk distribution shift: μ change = {bulk_shift:.4f}, σ change = {bulk_shape_change:.4f}")
    
    # Print summary table
    print(f"\n{'='*90}")
    print("EIGENVALUE DISTRIBUTION SUMMARY")
    print(f"{'='*90}")
    print(f"{'k':<4} {'Rand Ram%':<12} {'Opt Ram':<10} {'Rand λ₂':<12} {'Opt λ₂':<12} {'Opt λₙ':<12} {'Δbulk_μ':<10}")
    print(f"{'-'*90}")
    
    for k in sorted(results.keys()):
        r = results[k]
        rand_ram = f"{r['random_ramanujan_rate']:.1%}"
        opt_ram = "Yes" if r.get('optimized_is_ramanujan', False) else "No"
        
        # Average λ₂ from random graphs (approximate from max_nontrivial)
        rand_l2 = f"{r['random_max_nontrivial_mean']:.3f}"
        opt_l2 = f"{r.get('optimized_lambda_2', 0):.3f}"
        opt_ln = f"{r.get('optimized_lambda_n', 0):.3f}"
        
        # Bulk distribution change
        delta_bulk_mu = r.get('optimized_bulk_mean', 0) - r['random_bulk_mean']
        
        print(f"{k:<4} {rand_ram:<12} {opt_ram:<10} {rand_l2:<12} {opt_l2:<12} {opt_ln:<12} {delta_bulk_mu:<10.4f}")
    
    print(f"{'='*90}")
    print(f"Ramanujan bound (2√2): {2*np.sqrt(2):.4f}")
    print(f"McKay semicircle: radius 2√2, centered at 0")
    print(f"{'='*90}")
    
    # Optional: plot histograms if matplotlib available
    if plot_histograms:
        try:
            import matplotlib.pyplot as plt
            
            for k in k_values:
                r = results[k]
                
                plt.figure(figsize=(12, 5))
                
                # Plot random bulk distribution
                plt.subplot(1, 2, 1)
                plt.hist(r['random_bulk_eigenvalues'], bins=30, density=True, alpha=0.7, label='Random')
                
                # Overlay McKay semicircle
                x = np.linspace(-2.828, 2.828, 200)
                radius = 2 * np.sqrt(2)
                d = 3
                y = (d / (2 * np.pi)) * np.sqrt(np.maximum(0, 4*(d-1) - x**2)) / (d**2 - x**2 + 1e-10)
                plt.plot(x, y, 'r-', linewidth=2, label='McKay semicircle')
                
                plt.xlabel('Eigenvalue')
                plt.ylabel('Density')
                plt.title(f'Random Graphs (k={k})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Plot optimized if available
                if 'optimized_bulk_eigenvalues' in r:
                    plt.subplot(1, 2, 2)
                    plt.hist(r['optimized_bulk_eigenvalues'], bins=20, density=True, alpha=0.7, label='Optimized')
                    plt.plot(x, y, 'r-', linewidth=2, label='McKay semicircle')
                    
                    plt.xlabel('Eigenvalue')
                    plt.ylabel('Density')
                    plt.title(f'Optimized Graph (k={k})')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f'eigenvalue_dist_k{k}.png', dpi=150)
                print(f"Saved histogram: eigenvalue_dist_k{k}.png")
                plt.close()
        
        except ImportError:
            print("matplotlib not available for plotting")
    
    return results


def get_spectral_properties(G):
    """
    Compute spectral properties of a graph.
    
    Args:
        G: NetworkX graph
        
    Returns:
        dict: Dictionary containing spectral properties
    """
    adj_eigenvalues = sorted(nx.adjacency_spectrum(G).real, reverse=True)
    
    properties = {
        'lambda_1': adj_eigenvalues[0],  # Should be 3 for cubic graphs
        'lambda_2': adj_eigenvalues[1],  # Second largest
        'lambda_n': adj_eigenvalues[-1],  # Most negative
        'spectral_gap': adj_eigenvalues[0] - adj_eigenvalues[1],
        'eigenvalues': adj_eigenvalues
    }
    
    # Algebraic connectivity (Fiedler value) if connected
    if nx.is_connected(G):
        laplacian_spectrum = nx.laplacian_spectrum(G)
        properties['algebraic_connectivity'] = sorted(laplacian_spectrum)[1]
    
    return properties