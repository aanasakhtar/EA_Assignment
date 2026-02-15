"""
Compare different selection scheme combinations for TSP
Runs multiple trials and plots Avg. BSF and Avg. ASF convergence curves
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'TSP'))

from TSP.tsp_solver import TSPSolver, load_qatar_dataset


def run_scheme_trials(parent_selection, survivor_selection, num_trials=5, verbose=True):
    """
    Run multiple trials of EA with a specific selection scheme combination
    
    Args:
        parent_selection: Parent selection method
        survivor_selection: Survivor selection method
        num_trials: Number of independent trials
        verbose: Whether to print progress
        
    Returns:
        Dictionary with averaged metrics across trials
    """
    
    # Load dataset once
    cities, distance_matrix = load_qatar_dataset()
    
    # Storage for results across trials
    all_best_fitness_histories = []
    all_avg_fitness_histories = []
    
    if verbose:
        print(f"\nRunning {num_trials} trials: parent='{parent_selection}', survivor='{survivor_selection}'")
    
    for trial in range(num_trials):
        # Set different seed for each trial
        np.random.seed(42 + trial)
        
        # Create solver
        solver = TSPSolver(
            cities=cities,
            distance_matrix=distance_matrix,
            population_size=300,
            generations=1000,  # Reduced for faster testing
            crossover_rate=0.85,
            mutation_rate=0.25,
            elitism_count=5,
            # offspring_size=600,
            tournament_size=3,
            survivor_selection=survivor_selection,
            parent_selection=parent_selection,
            enable_hall_of_fame=True,
        )
        
        # Run evolution silently
        if verbose:
            print(f"  Trial {trial + 1}/{num_trials}...", end=" ", flush=True)
        
        best_tour = solver.evolve(verbose=False)
        
        if verbose:
            print(f"Best: {solver.best_fitness_history[-1]:.2f}")
        
        # Store histories
        all_best_fitness_histories.append(solver.best_fitness_history)
        all_avg_fitness_histories.append(solver.avg_fitness_history)
    
    # Average across trials
    all_best_fitness_histories = np.array(all_best_fitness_histories)
    all_avg_fitness_histories = np.array(all_avg_fitness_histories)
    
    avg_best_fitness = np.mean(all_best_fitness_histories, axis=0)
    avg_avg_fitness = np.mean(all_avg_fitness_histories, axis=0)
    
    std_best_fitness = np.std(all_best_fitness_histories, axis=0)
    std_avg_fitness = np.std(all_avg_fitness_histories, axis=0)
    
    return {
        'parent_selection': parent_selection,
        'survivor_selection': survivor_selection,
        'avg_best_fitness': avg_best_fitness,
        'avg_avg_fitness': avg_avg_fitness,
        'std_best_fitness': std_best_fitness,
        'std_avg_fitness': std_avg_fitness,
        'num_trials': num_trials,
    }


def plot_comparison(results):
    """
    Plot BSF and ASF side by side for a selection scheme combination
    
    Args:
        results: Dictionary from run_scheme_trials()
    """
    parent = results['parent_selection']
    survivor = results['survivor_selection']
    generations = len(results['avg_best_fitness'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    generations_range = np.arange(generations)
    
    # Plot 1: Best So Far (BSF)
    ax1.plot(generations_range, results['avg_best_fitness'], 'b-', linewidth=2, label='Avg. BSF')
    ax1.fill_between(
        generations_range,
        results['avg_best_fitness'] - results['std_best_fitness'],
        results['avg_best_fitness'] + results['std_best_fitness'],
        alpha=0.2, color='blue'
    )
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Best Distance', fontsize=12)
    ax1.set_title(f'Avg. Best So Far (BSF)\nParent: {parent}, Survivor: {survivor}', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Plot 2: Average So Far (ASF)
    ax2.plot(generations_range, results['avg_avg_fitness'], 'g-', linewidth=2, label='Avg. ASF')
    ax2.fill_between(
        generations_range,
        results['avg_avg_fitness'] - results['std_avg_fitness'],
        results['avg_avg_fitness'] + results['std_avg_fitness'],
        alpha=0.2, color='green'
    )
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Average Distance', fontsize=12)
    ax2.set_title(f'Avg. Average So Far (ASF)\nParent: {parent}, Survivor: {survivor}', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    
    # Save figure
    filename = f"comparison_{parent}_{survivor}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    
    plt.show()


def main():
    """Main function to compare selection schemes"""
    
    print("="*70)
    print("TSP SELECTION SCHEME COMPARISON")
    print("Multiple EA Runs with Different Selection Combinations")
    print("="*70)
    
    # Define combinations to test
    parent_selections = ['tournament', 'fitness_proportional', 'rank']
    survivor_selections = ['generational', 'mu_plus_lambda']
    
    # Store all results
    all_results = []
    
    # Run comparisons
    print("\nPhase 1: Running EA trials for each combination...")
    print("-" * 70)
    
    for parent_selection in parent_selections:
        for survivor_selection in survivor_selections:
            results = run_scheme_trials(
                parent_selection=parent_selection,
                survivor_selection=survivor_selection,
                num_trials=5,
                verbose=True
            )
            all_results.append(results)
    
    # Generate plots
    print("\n" + "="*70)
    print("Phase 2: Generating comparison plots...")
    print("-" * 70)
    
    for results in all_results:
        parent = results['parent_selection']
        survivor = results['survivor_selection']
        print(f"\nPlotting: parent='{parent}', survivor='{survivor}'")
        plot_comparison(results)
    
    # Generate summary statistics table
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"{'Parent Selection':<20} {'Survivor Selection':<20} {'Final Best':<15} {'Final Avg':<15}")
    print("-" * 70)
    
    for results in all_results:
        parent = results['parent_selection']
        survivor = results['survivor_selection']
        final_best = results['avg_best_fitness'][-1]
        final_avg = results['avg_avg_fitness'][-1]
        
        print(f"{parent:<20} {survivor:<20} {final_best:>14.2f} {final_avg:>14.2f}")
    
    print("\n" + "="*70)
    print("All plots saved! Check the workspace for comparison_*.png files")
    print("="*70)


if __name__ == "__main__":
    main()
