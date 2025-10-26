"""
Optimization methods for cubic graphs.
"""

import networkx as nx
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from ..core.spanning_trees import count_spanning_trees, get_second_eigenvalue
from ..core.graph_operations import get_valid_whitehead_flips, apply_whitehead_flip


def get_objective_function(objective='spanning_trees'):
    """
    Get the objective function to optimize.

    Args:
        objective: One of 'spanning_trees' (maximize ln(trees)) or
                  'expansion' (minimize λ₂, i.e., maximize spectral gap)

    Returns:
        tuple: (objective_func, maximize, name)
               objective_func: function that takes graph G and returns objective value
               maximize: True if we want to maximize, False if minimize
               name: display name for the objective
    """
    if objective == 'spanning_trees':
        return count_spanning_trees, True, 'ln(spanning trees)'
    elif objective == 'expansion':
        return get_second_eigenvalue, False, 'λ₂ (second eigenvalue)'
    else:
        raise ValueError(f"Unknown objective: {objective}. Choose 'spanning_trees' or 'expansion'")


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


def gradient_ascent_first_improvement(G, max_iterations=1000, verbose=True, objective='spanning_trees'):
    """
    First-improvement gradient ascent: take the first flip that improves the objective.
    Much faster than greedy, but may not reach as good a local optimum.

    Args:
        G: NetworkX graph
        max_iterations: Maximum iterations
        verbose: Whether to print progress
        objective: 'spanning_trees' (maximize) or 'expansion' (minimize λ₂)

    Returns:
        tuple: (optimized_graph, final_objective_value)
    """
    obj_func, maximize, obj_name = get_objective_function(objective)
    current_value = obj_func(G)

    if verbose:
        print(f"Initial {obj_name} value: {current_value:.6f}")

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

            new_value = obj_func(G_temp)

            # Check if improved (maximize or minimize based on objective)
            is_better = (new_value > current_value) if maximize else (new_value < current_value)

            if is_better:
                # Apply this flip
                apply_whitehead_flip(G, edge1, edge2, flip_type)
                current_value = new_value
                improved = True

                if verbose and (iteration % 10 == 0 or iteration < 10):
                    print(f"Iteration {iteration}: {obj_name} = {current_value:.6f}")
                break

        if not improved:
            if verbose:
                print(f"Local optimum reached at iteration {iteration}")
                print(f"Final {obj_name} value: {current_value:.6f}")
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


def compute_default_lambda(n):
    """
    Compute default lambda parameter for multi-flip moves based on graph size.

    The diameter of the graph space under Whitehead flips scales roughly with N,
    so multi-flip jump sizes should scale accordingly.

    Args:
        n: Number of vertices in the graph

    Returns:
        float: Default lambda value (expected number of flips per move)
    """
    # Use N/10 as a baseline heuristic
    lambda_base = n / 10.0

    # Ensure a minimum of 1.0 for small graphs
    return max(1.0, lambda_base)


def gradient_ascent_first_multifold(G, max_iterations=1000, lambda_jump=None,
                                     jump_prob=0.1, max_attempts_per_iteration=100, verbose=True,
                                     objective='spanning_trees'):
    """
    First-improvement gradient ascent with mixture distribution for flip counts.

    Uses a mixture of single flips and multi-flips:
    - With probability jump_prob: perform k ~ Poisson(λ_jump) flips (big jump)
    - With probability 1-jump_prob: perform exactly 1 flip (local search)

    This creates "fat-tailed" exploration: mostly efficient single flips, but occasional
    large jumps to escape local optima.

    Args:
        G: NetworkX graph
        max_iterations: Maximum number of successful moves
        lambda_jump: Mean for Poisson when taking big jumps (default: N/5)
        jump_prob: Probability of attempting a big jump (default: 0.1)
        max_attempts_per_iteration: Max random multi-flips to try before giving up
        verbose: Whether to print progress
        objective: 'spanning_trees' (maximize) or 'expansion' (minimize λ₂)

    Returns:
        tuple: (optimized_graph, final_objective_value)
    """
    n = G.number_of_nodes()

    # Auto-compute lambda_jump if not provided
    if lambda_jump is None:
        lambda_jump = max(2.0, n / 5.0)

    obj_func, maximize, obj_name = get_objective_function(objective)
    current_value = obj_func(G)
    total_flips_performed = 0
    num_jumps = 0

    if verbose:
        print(f"Initial {obj_name} value: {current_value:.6f}")
        print(f"N = {n}, λ_jump = {lambda_jump:.2f}, p_jump = {jump_prob:.2f}")
        print(f"(Mixture: mostly single flips, occasional big jumps)\n")

    for iteration in range(max_iterations):
        improved = False

        # Try random multi-flip moves until we find an improving one
        for attempt in range(max_attempts_per_iteration):
            # Sample number of flips using mixture distribution
            if random.random() < jump_prob:
                # Take a big jump: sample from Poisson(lambda_jump)
                num_flips = 0
                while num_flips < 1:
                    num_flips = np.random.poisson(lambda_jump)
                is_jump = True
            else:
                # Single flip (most common case)
                num_flips = 1
                is_jump = False

            # Create a temporary graph for the multi-flip move
            G_temp = G.copy()

            # Apply num_flips random Whitehead flips sequentially
            valid_move = True
            flips_in_attempt = 0
            for _ in range(num_flips):
                valid_flips = get_valid_whitehead_flips(G_temp)

                if not valid_flips:
                    valid_move = False
                    break

                edge1, edge2, flip_type = random.choice(valid_flips)
                apply_whitehead_flip(G_temp, edge1, edge2, flip_type)
                flips_in_attempt += 1

            total_flips_performed += flips_in_attempt

            if not valid_move:
                continue  # Try another multi-flip sequence

            # Evaluate the final state
            new_value = obj_func(G_temp)

            # Check if improved (maximize or minimize based on objective)
            is_better = (new_value > current_value) if maximize else (new_value < current_value)

            if is_better:
                # Found an improvement! Apply it
                G = G_temp
                current_value = new_value
                improved = True
                if is_jump:
                    num_jumps += 1

                if verbose and (iteration % 10 == 0 or iteration < 10):
                    jump_str = " (BIG JUMP!)" if is_jump else ""
                    print(f"Iteration {iteration}: {obj_name} = {current_value:.6f} "
                          f"(found after {attempt+1} attempts, {flips_in_attempt} flips{jump_str}, "
                          f"total flips: {total_flips_performed})")
                break

        if not improved:
            if verbose:
                print(f"\nLocal optimum reached at iteration {iteration}")
                print(f"Final {obj_name} value: {current_value:.6f}")
                print(f"Total flips performed: {total_flips_performed}")
                print(f"Big jumps taken: {num_jumps}/{iteration} ({100*num_jumps/max(1,iteration):.1f}%)")
            break

    return G, current_value


def simulated_annealing_multifold(G, max_iterations=1000, T0=None, cooling_rate=0.97,
                                  lambda_jump=None, jump_prob_max=0.15, verbose=True):
    """
    Multi-fold simulated annealing with mixture distribution for flip counts.

    Uses a mixture of single flips and multi-flips:
    - With probability p(T) = jump_prob_max * (T/T0): perform k ~ Poisson(λ_jump) flips (big jump)
    - With probability 1-p(T): perform exactly 1 flip (local search)

    This creates "fat-tailed" exploration: mostly efficient single flips, but occasional
    large jumps to escape local optima. Jump probability decreases with temperature.

    Args:
        G: NetworkX graph
        max_iterations: Maximum number of moves to attempt
        T0: Initial temperature (auto-calibrated if None)
        cooling_rate: Multiply temperature by this each iteration (0.95-0.99 typical)
        lambda_jump: Mean for Poisson when taking big jumps (default: N/5)
        jump_prob_max: Maximum probability of big jump at T=T0 (default: 0.15)
        verbose: Whether to print progress

    Returns:
        tuple: (optimized_graph, final_ln_spanning_trees)
    """
    n = G.number_of_nodes()

    # Auto-compute lambda_jump if not provided
    if lambda_jump is None:
        lambda_jump = max(2.0, n / 5.0)  # Bigger jumps when we do jump

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
    total_flips = 0
    total_flips_overall = 0
    num_jumps = 0

    if verbose:
        print(f"Initial T value: {current_value:.6f}")
        print(f"N = {n}, λ_jump = {lambda_jump:.2f}, p_jump_max = {jump_prob_max:.2f}")
        print(f"(Mixture: mostly single flips, occasional big jumps)\n")

    for iteration in range(max_iterations):
        # Determine number of flips using mixture distribution
        p_jump = jump_prob_max * (temperature / T0)  # Jump probability decreases with T

        if random.random() < p_jump:
            # Take a big jump: sample from Poisson(lambda_jump)
            num_flips = 0
            while num_flips < 1:
                num_flips = np.random.poisson(lambda_jump)
            num_jumps += 1
        else:
            # Single flip (most common case)
            num_flips = 1

        # Create a temporary graph for the multi-flip move
        G_temp = G.copy()
        flips_applied = []

        # Apply num_flips Whitehead flips sequentially
        valid_move = True
        for _ in range(num_flips):
            valid_flips = get_valid_whitehead_flips(G_temp)

            if not valid_flips:
                # Can't complete the desired number of flips
                valid_move = False
                break

            # Pick a random flip
            edge1, edge2, flip_type = random.choice(valid_flips)
            apply_whitehead_flip(G_temp, edge1, edge2, flip_type)
            flips_applied.append((edge1, edge2, flip_type))

        if not valid_move:
            if verbose:
                print(f"No valid flips available at iteration {iteration}")
            break

        # Evaluate the final state after all flips
        new_value = count_spanning_trees(G_temp)
        delta = new_value - current_value
        total_attempts += 1
        total_flips += len(flips_applied)
        total_flips_overall += len(flips_applied)

        # Accept or reject the entire multi-flip move based on Metropolis criterion
        if delta > 0:
            # Always accept improvements
            accept = True
        else:
            # Accept worse moves with probability exp(delta / temperature)
            accept_prob = np.exp(delta / temperature)
            accept = random.random() < accept_prob

        if accept:
            # Apply all flips to the actual graph
            G = G_temp
            current_value = new_value
            accepts += 1

            # Track best solution found
            if current_value > best_value:
                best_G = G.copy()
                best_value = current_value

        # Cool down
        temperature *= cooling_rate

        # Periodic progress reporting
        if verbose and iteration % 50 == 0:
            accept_rate = accepts / total_attempts if total_attempts > 0 else 0
            avg_flips = total_flips / total_attempts if total_attempts > 0 else 0
            jump_rate = num_jumps / (iteration + 1)
            print(f"Iter {iteration}: T_val = {current_value:.6f}, best = {best_value:.6f}, "
                  f"temp = {temperature:.4f}, p_jump = {p_jump:.3f}, "
                  f"accept_rate = {accept_rate:.2f}, avg_flips = {avg_flips:.1f}, jumps = {num_jumps}")
            accepts = 0
            total_attempts = 0
            total_flips = 0

    if verbose:
        print(f"\nFinal T value: {best_value:.6f}")
        print(f"Total flips performed: {total_flips_overall}")
        print(f"Big jumps taken: {num_jumps}/{max_iterations} ({100*num_jumps/max_iterations:.1f}%)")

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
    elif method == 'first-multifold':
        G_opt, T_opt = gradient_ascent_first_multifold(G, **method_kwargs)
    elif method == 'sa':
        G_opt, T_opt = simulated_annealing(G, **method_kwargs)
    elif method == 'sa-multifold':
        G_opt, T_opt = simulated_annealing_multifold(G, **method_kwargs)
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
            elif method == 'first-multifold':
                G_opt, T_opt = gradient_ascent_first_multifold(G, verbose=False, **method_kwargs)
            elif method == 'sa':
                G_opt, T_opt = simulated_annealing(G, verbose=False, **method_kwargs)
            elif method == 'sa-multifold':
                G_opt, T_opt = simulated_annealing_multifold(G, verbose=False, **method_kwargs)
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
        elif method == 'first-multifold':
            G_opt, T_opt = gradient_ascent_first_multifold(G, verbose=False, **method_kwargs)
        elif method == 'sa':
            G_opt, T_opt = simulated_annealing(G, verbose=False, **method_kwargs)
        elif method == 'sa-multifold':
            G_opt, T_opt = simulated_annealing_multifold(G, verbose=False, **method_kwargs)
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