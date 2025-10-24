"""
Optimization methods for cubic graphs.
"""

import networkx as nx
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from ..core.spanning_trees import count_spanning_trees
from ..core.graph_operations import get_valid_whitehead_flips, apply_whitehead_flip


def gradient_ascent_greedy(G, max_iterations=1000, verbose=True):
    """
    Greedy gradient ascent: always take the best improving flip.
    
    Args:
        G: NetworkX graph
        max_iterations: Maximum iterations
        verbose: Whether to print progress
        
    Returns:
        tuple: (optimized_graph, final_ln_spanning_trees)
    """
    current_value = count_spanning_trees(G)
    
    if verbose:
        print(f"Initial T value: {current_value:.6f}")
    
    for iteration in range(max_iterations):
        valid_flips = get_valid_whitehead_flips(G)
        
        if not valid_flips:
            if verbose:
                print(f"No valid flips available at iteration {iteration}")
            break
        
        # Evaluate all flips and find the best one
        best_flip = None
        best_value = current_value
        
        for edge1, edge2, flip_type in valid_flips:
            # Apply flip temporarily
            G_temp = G.copy()
            apply_whitehead_flip(G_temp, edge1, edge2, flip_type)
            
            new_value = count_spanning_trees(G_temp)
            
            if new_value > best_value:
                best_value = new_value
                best_flip = (edge1, edge2, flip_type)
        
        if best_flip is None:
            if verbose:
                print(f"Local maximum reached at iteration {iteration}")
                print(f"Final T value: {current_value:.6f}")
            break
        
        # Apply best flip
        apply_whitehead_flip(G, best_flip[0], best_flip[1], best_flip[2])
        current_value = best_value
        
        if verbose and (iteration % 10 == 0 or iteration < 10):
            print(f"Iteration {iteration}: T = {current_value:.6f}, improvement = {current_value - count_spanning_trees(G):.6f}")
    
    return G, current_value


def gradient_ascent_first_improvement(G, max_iterations=1000, verbose=True):
    """
    First-improvement gradient ascent: take the first flip that improves T.
    Much faster than greedy, but may not reach as good a local optimum.
    
    Args:
        G: NetworkX graph
        max_iterations: Maximum iterations
        verbose: Whether to print progress
        
    Returns:
        tuple: (optimized_graph, final_ln_spanning_trees)
    """
    current_value = count_spanning_trees(G)
    
    if verbose:
        print(f"Initial T value: {current_value:.6f}")
    
    for iteration in range(max_iterations):
        valid_flips = get_valid_whitehead_flips(G)
        
        if not valid_flips:
            if verbose:
                print(f"No valid flips available at iteration {iteration}")
            break
        
        # Shuffle to avoid bias
        random.shuffle(valid_flips)
        
        improved = False
        for edge1, edge2, flip_type in valid_flips:
            # Apply flip temporarily
            G_temp = G.copy()
            apply_whitehead_flip(G_temp, edge1, edge2, flip_type)
            
            new_value = count_spanning_trees(G_temp)
            
            if new_value > current_value:
                # Apply this flip
                apply_whitehead_flip(G, edge1, edge2, flip_type)
                current_value = new_value
                improved = True
                
                if verbose and (iteration % 10 == 0 or iteration < 10):
                    print(f"Iteration {iteration}: T = {current_value:.6f}")
                break
        
        if not improved:
            if verbose:
                print(f"Local maximum reached at iteration {iteration}")
                print(f"Final T value: {current_value:.6f}")
            break
    
    return G, current_value


def calibrate_temperature(G, num_samples=100):
    """
    Estimate typical magnitude of changes in T by sampling random flips.
    Returns suggested T0 for simulated annealing.
    
    Args:
        G: NetworkX graph
        num_samples: Number of random flips to sample
        
    Returns:
        float: Suggested initial temperature
    """
    current_value = count_spanning_trees(G)
    deltas = []
    
    for _ in range(num_samples):
        valid_flips = get_valid_whitehead_flips(G)
        if not valid_flips:
            break
        
        edge1, edge2, flip_type = random.choice(valid_flips)
        G_temp = G.copy()
        apply_whitehead_flip(G_temp, edge1, edge2, flip_type)
        new_value = count_spanning_trees(G_temp)
        
        deltas.append(abs(new_value - current_value))
    
    if not deltas:
        return 0.1  # Default fallback
    
    # T0 should make moderate changes acceptably likely
    # If typical |ΔT| = x, set T0 so exp(-x/T0) ≈ 0.5-0.7
    median_delta = np.median(deltas)
    T0 = median_delta / 0.5  # exp(-median/T0) ≈ 0.6 at start
    
    return max(T0, 0.01)  # Ensure non-zero


def simulated_annealing(G, max_iterations=1000, T0=None, cooling_rate=0.97, 
                        adaptive=True, verbose=True):
    """
    Simulated annealing optimization.
    
    Args:
        G: NetworkX graph
        max_iterations: Maximum iterations
        T0: Initial temperature (auto-calibrated if None)
        cooling_rate: Multiply temperature by this each iteration (0.95-0.99 typical)
        adaptive: If True, adjust cooling based on acceptance rate
        verbose: Whether to print progress
        
    Returns:
        tuple: (optimized_graph, final_ln_spanning_trees)
    """
    current_value = count_spanning_trees(G)
    best_G = G.copy()
    best_value = current_value
    
    # Auto-calibrate temperature if not provided
    if T0 is None:
        T0 = calibrate_temperature(G)
        if verbose:
            print(f"Auto-calibrated T0 = {T0:.4f}")
    
    temperature = T0
    accepts = 0
    total_attempts = 0
    
    if verbose:
        print(f"Initial T value: {current_value:.6f}\n")
    
    for iteration in range(max_iterations):
        valid_flips = get_valid_whitehead_flips(G)
        
        if not valid_flips:
            if verbose:
                print(f"No valid flips available at iteration {iteration}")
            break
        
        # Pick random flip to consider
        edge1, edge2, flip_type = random.choice(valid_flips)
        
        # Evaluate flip
        G_temp = G.copy()
        apply_whitehead_flip(G_temp, edge1, edge2, flip_type)
        new_value = count_spanning_trees(G_temp)
        
        delta = new_value - current_value
        total_attempts += 1
        
        # Accept or reject based on metropolis criterion
        if delta > 0:
            # Always accept improvements
            accept = True
        else:
            # Accept worse moves with probability exp(delta / temperature)
            accept_prob = np.exp(delta / temperature)
            accept = random.random() < accept_prob
        
        if accept:
            apply_whitehead_flip(G, edge1, edge2, flip_type)
            current_value = new_value
            accepts += 1
            
            # Track best solution found
            if current_value > best_value:
                best_G = G.copy()
                best_value = current_value
        
        # Cool down
        temperature *= cooling_rate
        
        # Adaptive cooling: adjust rate based on acceptance
        if adaptive and iteration > 0 and iteration % 50 == 0:
            accept_rate = accepts / total_attempts
            if accept_rate > 0.6:
                cooling_rate = max(0.90, cooling_rate - 0.01)  # Cool faster
            elif accept_rate < 0.2:
                cooling_rate = min(0.99, cooling_rate + 0.01)  # Cool slower
            
            if verbose:
                print(f"Iter {iteration}: T_val = {current_value:.6f}, best = {best_value:.6f}, "
                      f"temp = {temperature:.4f}, accept_rate = {accept_rate:.2f}")
            
            accepts = 0
            total_attempts = 0
        elif verbose and iteration % 50 == 0:
            print(f"Iter {iteration}: T_val = {current_value:.6f}, best = {best_value:.6f}, "
                  f"temp = {temperature:.4f}")
    
    if verbose:
        print(f"\nFinal T value: {best_value:.6f}")
    
    return best_G, best_value


def _run_single_optimization(args):
    """
    Helper function for parallel optimization.
    Takes a tuple of (n, seed, method, method_kwargs).
    """
    n, seed, method, method_kwargs = args
    
    # Generate starting graph
    G = nx.random_regular_graph(3, n, seed=seed)
    
    # Run optimization
    if method == 'greedy':
        G_opt, T_opt = gradient_ascent_greedy(G, **method_kwargs)
    elif method == 'first':
        G_opt, T_opt = gradient_ascent_first_improvement(G, **method_kwargs)
    elif method == 'sa':
        G_opt, T_opt = simulated_annealing(G, **method_kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return G_opt, T_opt, seed


def optimize_with_restarts(n, method='greedy', restarts=1, parallel=False, 
                          base_seed=42, verbose=True, **method_kwargs):
    """
    Run optimization with multiple restarts and return the best result.
    
    Args:
        n: number of vertices
        method: 'greedy', 'first', or 'sa'
        restarts: int or 'auto'
            - int: run this many times with different random starts
            - 'auto': keep running until no improvement for patience rounds
        parallel: bool or int
            - False: run sequentially
            - True: use all available cores
            - int: use this many processes
        base_seed: base random seed (each restart gets base_seed + restart_num)
        verbose: print progress
        **method_kwargs: passed to the optimization method
    
    Returns:
        tuple: (best_graph, best_T_value, num_trials_run)
    """
    if restarts == 'auto':
        return _optimize_with_auto_restarts(n, method, parallel, base_seed, verbose, **method_kwargs)
    
    if restarts < 1:
        raise ValueError("restarts must be >= 1 or 'auto'")
    
    if verbose and restarts > 1:
        print(f"Running {restarts} restarts...")
    
    best_G = None
    best_T = float('-inf')
    best_seed = None
    
    if parallel and restarts > 1:
        # Parallel execution
        n_processes = parallel if isinstance(parallel, int) else multiprocessing.cpu_count()
        n_processes = min(n_processes, restarts)
        
        if verbose:
            print(f"Using {n_processes} parallel processes")
        
        # Prepare arguments for each trial
        args_list = [(n, base_seed + i, method, method_kwargs) for i in range(restarts)]
        
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            results = list(executor.map(_run_single_optimization, args_list))
        
        # Find best result
        for G_opt, T_opt, seed in results:
            if T_opt > best_T:
                best_G = G_opt
                best_T = T_opt
                best_seed = seed
        
        if verbose:
            print(f"Best result: T={best_T:.6f} (from seed {best_seed})")
    
    else:
        # Sequential execution
        for restart in range(restarts):
            seed = base_seed + restart
            G = nx.random_regular_graph(3, n, seed=seed)
            
            if method == 'greedy':
                G_opt, T_opt = gradient_ascent_greedy(G, verbose=False, **method_kwargs)
            elif method == 'first':
                G_opt, T_opt = gradient_ascent_first_improvement(G, verbose=False, **method_kwargs)
            elif method == 'sa':
                G_opt, T_opt = simulated_annealing(G, verbose=False, **method_kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            if T_opt > best_T:
                best_G = G_opt
                best_T = T_opt
                best_seed = seed
                if verbose and restarts > 1:
                    print(f"  Restart {restart+1}/{restarts}: New best! T={best_T:.6f}")
            elif verbose and restarts > 1:
                print(f"  Restart {restart+1}/{restarts}: T={T_opt:.6f}")
    
    return best_G, best_T, restarts


def _optimize_with_auto_restarts(n, method, parallel, base_seed, verbose, **method_kwargs):
    """
    Run optimization with automatic restarts until no improvement is seen.
    Stops after patience=5 rounds without improvement.
    """
    patience = 5
    no_improvement_count = 0
    trial = 0
    best_T = float('-inf')
    best_G = None
    
    if verbose:
        print("Running with auto restarts (patience=5)...")
    
    while no_improvement_count < patience:
        seed = base_seed + trial
        G = nx.random_regular_graph(3, n, seed=seed)
        
        if method == 'greedy':
            G_opt, T_opt = gradient_ascent_greedy(G, verbose=False, **method_kwargs)
        elif method == 'first':
            G_opt, T_opt = gradient_ascent_first_improvement(G, verbose=False, **method_kwargs)
        elif method == 'sa':
            G_opt, T_opt = simulated_annealing(G, verbose=False, **method_kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if T_opt > best_T:
            best_G = G_opt
            best_T = T_opt
            no_improvement_count = 0
            if verbose:
                print(f"  Trial {trial+1}: New best! T={best_T:.6f}")
        else:
            no_improvement_count += 1
            if verbose:
                print(f"  Trial {trial+1}: T={T_opt:.6f} (no improvement: {no_improvement_count}/{patience})")
        
        trial += 1
        
        # Safety: max 50 trials
        if trial >= 50:
            if verbose:
                print(f"  Reached maximum of 50 trials")
            break
    
    if verbose:
        print(f"Auto restarts complete after {trial} trials")
    
    return best_G, best_T, trial


def run_optimization_sweep(k_values, methods=['greedy', 'first', 'sa'], 
                          restarts=1, parallel=False,
                          save_graphs=True, random_seed=42, verbose=True):
    """
    Run optimization for multiple k values and collect results.
    
    Args:
        k_values: list of k values to test (or range object)
        methods: which methods to run ('greedy', 'first', 'sa')
        restarts: int or 'auto' - number of random restarts per method
        parallel: bool or int - use parallel processing for restarts
        save_graphs: whether to save best graph for each k
        random_seed: base seed for reproducibility
        verbose: whether to print detailed progress
    
    Returns:
        dict: results indexed by k
    """
    from ..utils.io import save_graph
    from ..analysis.properties import analyze_graph_properties
    
    results = {}
    
    for k in k_values:
        n = 2 * k
        print(f"\n{'='*70}")
        print(f"OPTIMIZING k={k} (n={n} vertices)")
        print(f"{'='*70}")
        
        # For large graphs, use lightweight mode
        lightweight_mode = (k >= 30)
        
        k_results = {
            'k': k,
            'n': n
        }
        
        best_G = None
        best_T = float('-inf')
        best_method = None
        
        # Run requested methods
        if 'greedy' in methods:
            print(f"\nRunning Greedy with {restarts} restarts...")
            G_opt, T_opt, num_trials = optimize_with_restarts(
                n, method='greedy', restarts=restarts, parallel=parallel,
                base_seed=random_seed, verbose=verbose, max_iterations=100
            )
            k_results['greedy_T'] = T_opt
            k_results['greedy_trials'] = num_trials
            if T_opt > best_T:
                best_G, best_T, best_method = G_opt, T_opt, 'greedy'
            print(f"  Final T: {T_opt:.6f}")
        
        if 'first' in methods:
            print(f"\nRunning First-improvement with {restarts} restarts...")
            G_opt, T_opt, num_trials = optimize_with_restarts(
                n, method='first', restarts=restarts, parallel=parallel,
                base_seed=random_seed + 1000, verbose=verbose, max_iterations=100
            )
            k_results['first_T'] = T_opt
            k_results['first_trials'] = num_trials
            if T_opt > best_T:
                best_G, best_T, best_method = G_opt, T_opt, 'first'
            print(f"  Final T: {T_opt:.6f}")
        
        if 'sa' in methods:
            print(f"\nRunning Simulated Annealing with {restarts} restarts...")
            G_opt, T_opt, num_trials = optimize_with_restarts(
                n, method='sa', restarts=restarts, parallel=parallel,
                base_seed=random_seed + 2000, verbose=verbose, max_iterations=500
            )
            k_results['sa_T'] = T_opt
            k_results['sa_trials'] = num_trials
            if T_opt > best_T:
                best_G, best_T, best_method = G_opt, T_opt, 'sa'
            print(f"  Final T: {T_opt:.6f}")
        
        k_results['best_method'] = best_method
        k_results['best_T'] = best_T
        
        # Analyze best graph
        print(f"\n{'='*70}")
        print(f"Best result for k={k}: {best_method} with T={best_T:.6f}")
        print(f"{'='*70}")
        
        if verbose:
            analyze_graph_properties(best_G, f"Optimal k={k}", 
                                   compute_girth=not lightweight_mode,
                                   compute_cycles=not lightweight_mode,
                                   compute_automorphisms=True)
        
        # Save graph if requested
        if save_graphs:
            filename = f"optimal_cubic_k{k}.pkl"
            save_graph(best_G, filename, best_T, compute_all_properties=not lightweight_mode)
        
        results[k] = k_results
    
    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'k':<4} {'n':<6} {'Method':<8} {'ln(trees)':<12} {'Trials':<8}")
    print(f"{'-'*70}")
    for k in sorted(results.keys()):
        r = results[k]
        method = r['best_method']
        trials_key = f'{method}_trials'
        trials = r.get(trials_key, 'N/A')
        print(f"{r['k']:<4} {r['n']:<6} {method:<8} {r['best_T']:<12.6f} {trials:<8}")
    print(f"{'='*70}")
    
    return results