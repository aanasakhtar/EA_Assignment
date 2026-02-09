"""
Compare different EA configurations
Run multiple experiments and analyze results
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))

from ea_framework import EAFramework


def run_parameter_comparison(solver_class, problem_data, param_configs, num_runs=5):
    """
    Run EA with different parameter configurations and compare results
    
    Args:
        solver_class: The solver class (TSPSolver or ExamSchedulingSolver)
        problem_data: Problem-specific data (dict)
        param_configs: List of dicts with parameter configurations
        num_runs: Number of runs per configuration
        
    Returns:
        Dictionary with results for each configuration
    """
    results = {}
    
    for config_name, params in param_configs.items():
        print(f"\nTesting configuration: {config_name}")
        print(f"Parameters: {params}")
        
        config_results = {
            'best_fitnesses': [],
            'avg_fitnesses': [],
            'convergence_histories': [],
            'runtimes': []
        }
        
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}...", end=' ')
            
            # Create solver with current parameters
            solver = solver_class(**problem_data, **params)
            
            # Run evolution
            import time
            start_time = time.time()
            solver.evolve(verbose=False)
            runtime = time.time() - start_time
            
            # Store results
            config_results['best_fitnesses'].append(solver.best_fitness)
            config_results['avg_fitnesses'].append(solver.avg_fitness_history[-1])
            config_results['convergence_histories'].append(solver.best_fitness_history)
            config_results['runtimes'].append(runtime)
            
            print(f"Best: {solver.best_fitness:.2f}, Time: {runtime:.2f}s")
        
        # Calculate statistics
        results[config_name] = {
            'mean_best': np.mean(config_results['best_fitnesses']),
            'std_best': np.std(config_results['best_fitnesses']),
            'mean_runtime': np.mean(config_results['runtimes']),
            'convergence_histories': config_results['convergence_histories'],
            'all_best': config_results['best_fitnesses']
        }
    
    return results


def plot_comparison(results, title="Parameter Comparison", save_path=None):
    """
    Plot comparison of different configurations
    
    Args:
        results: Results from run_parameter_comparison
        title: Plot title
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Box plot of best fitnesses
    ax1 = axes[0, 0]
    config_names = list(results.keys())
    best_fitnesses = [results[name]['all_best'] for name in config_names]
    ax1.boxplot(best_fitnesses, labels=config_names)
    ax1.set_ylabel('Best Fitness')
    ax1.set_title('Distribution of Best Fitnesses')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Mean best fitness with error bars
    ax2 = axes[0, 1]
    means = [results[name]['mean_best'] for name in config_names]
    stds = [results[name]['std_best'] for name in config_names]
    x_pos = np.arange(len(config_names))
    ax2.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(config_names, rotation=45, ha='right')
    ax2.set_ylabel('Mean Best Fitness')
    ax2.set_title('Mean Performance (with std dev)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Convergence curves
    ax3 = axes[1, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(config_names)))
    for i, name in enumerate(config_names):
        histories = results[name]['convergence_histories']
        # Average convergence
        max_len = max(len(h) for h in histories)
        avg_history = np.mean([
            np.pad(h, (0, max_len - len(h)), constant_values=h[-1]) 
            for h in histories
        ], axis=0)
        ax3.plot(avg_history, label=name, color=colors[i], linewidth=2)
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Best Fitness')
    ax3.set_title('Average Convergence Curves')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Runtime comparison
    ax4 = axes[1, 1]
    runtimes = [results[name]['mean_runtime'] for name in config_names]
    ax4.bar(x_pos, runtimes, alpha=0.7, color='steelblue')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(config_names, rotation=45, ha='right')
    ax4.set_ylabel('Mean Runtime (seconds)')
    ax4.set_title('Computational Efficiency')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()


def print_comparison_report(results):
    """Print detailed comparison report"""
    print("\n" + "="*70)
    print("PARAMETER COMPARISON REPORT")
    print("="*70)
    
    config_names = list(results.keys())
    
    # Find best configuration
    best_mean_fitness = min(results[name]['mean_best'] for name in config_names)
    best_config = [name for name in config_names 
                  if results[name]['mean_best'] == best_mean_fitness][0]
    
    print(f"\nBest Configuration: {best_config}")
    print(f"  Mean Best Fitness: {results[best_config]['mean_best']:.2f}")
    print(f"  Std Dev: {results[best_config]['std_best']:.2f}")
    print(f"  Mean Runtime: {results[best_config]['mean_runtime']:.2f}s")
    
    print("\nAll Configurations:")
    print("-" * 70)
    print(f"{'Configuration':<25} {'Mean Best':<12} {'Std Dev':<12} {'Runtime (s)':<12}")
    print("-" * 70)
    
    # Sort by mean best fitness
    sorted_names = sorted(config_names, 
                         key=lambda x: results[x]['mean_best'])
    
    for name in sorted_names:
        mean_best = results[name]['mean_best']
        std_best = results[name]['std_best']
        runtime = results[name]['mean_runtime']
        marker = " ← BEST" if name == best_config else ""
        print(f"{name:<25} {mean_best:<12.2f} {std_best:<12.2f} {runtime:<12.2f}{marker}")
    
    print("-" * 70)
    
    # Statistical comparison
    print("\nStatistical Analysis:")
    for name in sorted_names:
        improvement = ((results[sorted_names[0]]['mean_best'] - 
                       results[name]['mean_best']) / 
                      results[sorted_names[0]]['mean_best'] * 100)
        if name != sorted_names[0]:
            print(f"  {name}: {improvement:.2f}% worse than best")


# Example usage for TSP
def example_tsp_comparison():
    """Example: Compare TSP solver with different parameters"""
    from TSP.tsp_solver import TSPSolver, load_qatar_dataset
    
    print("Loading dataset...")
    cities, distance_matrix = load_qatar_dataset()
    
    problem_data = {
        'cities': cities,
        'distance_matrix': distance_matrix
    }
    
    # Define configurations to test
    param_configs = {
        'Default': {
            'population_size': 100,
            'generations': 300,
            'crossover_rate': 0.85,
            'mutation_rate': 0.15,
            'elitism_count': 5,
            'tournament_size': 5
        },
        'Large Population': {
            'population_size': 200,
            'generations': 300,
            'crossover_rate': 0.85,
            'mutation_rate': 0.15,
            'elitism_count': 5,
            'tournament_size': 5
        },
        'High Mutation': {
            'population_size': 100,
            'generations': 300,
            'crossover_rate': 0.85,
            'mutation_rate': 0.25,
            'elitism_count': 5,
            'tournament_size': 5
        },
        'Strong Elitism': {
            'population_size': 100,
            'generations': 300,
            'crossover_rate': 0.85,
            'mutation_rate': 0.15,
            'elitism_count': 15,
            'tournament_size': 5
        }
    }
    
    # Run comparison
    results = run_parameter_comparison(
        TSPSolver, 
        problem_data, 
        param_configs, 
        num_runs=3
    )
    
    # Print report
    print_comparison_report(results)
    
    # Plot comparison
    plot_comparison(
        results, 
        title="TSP Parameter Comparison - Qatar Dataset",
        save_path="TSP_parameter_comparison.png"
    )


if __name__ == "__main__":
    print("="*70)
    print("EA Parameter Comparison Tool")
    print("="*70)
    print("\nThis script compares different EA parameter configurations.")
    print("Modify the param_configs dictionary to test your own settings.")
    print("\nRunning example TSP comparison...\n")
    
    example_tsp_comparison()
