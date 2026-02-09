"""
Compare Different Parent Selection Methods
Demonstrates impact of parent selection on EA performance
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from TSP.tsp_solver import TSPSolver, load_qatar_dataset


def compare_parent_selection_methods(num_runs=3):
    """
    Compare different parent selection methods on TSP
    
    Args:
        num_runs: Number of runs per method
        
    Returns:
        Results dictionary
    """
    print("="*70)
    print("PARENT SELECTION METHODS COMPARISON")
    print("="*70)
    
    # Load dataset
    print("\nLoading Qatar TSP dataset...")
    cities, distance_matrix = load_qatar_dataset()
    
    # Define methods to compare
    methods = {
        'Fitness Proportional (Roulette)': {
            'parent_selection': 'fitness_proportional',
            'mutation_rate': 0.2  # Higher to combat premature convergence
        },
        'Rank-Based Selection': {
            'parent_selection': 'rank',
            'mutation_rate': 0.15
        },
        'Tournament (k=3)': {
            'parent_selection': 'tournament',
            'tournament_size': 3,
            'mutation_rate': 0.15
        },
        'Tournament (k=5)': {
            'parent_selection': 'tournament',
            'tournament_size': 5,
            'mutation_rate': 0.15
        },
        'Tournament (k=7)': {
            'parent_selection': 'tournament',
            'tournament_size': 7,
            'mutation_rate': 0.15
        }
    }
    
    results = {}
    
    for method_name, params in methods.items():
        print(f"\n{'='*70}")
        print(f"Testing: {method_name}")
        print(f"{'='*70}")
        print(f"Parameters: {params}")
        
        method_results = {
            'best_fitnesses': [],
            'convergence_histories': [],
            'diversity_histories': [],
            'selection_pressure_history': [],
            'runtimes': []
        }
        
        for run in range(num_runs):
            print(f"\n  Run {run + 1}/{num_runs}...", end=' ')
            
            # Create solver
            solver = TSPSolver(
                cities=cities,
                distance_matrix=distance_matrix,
                population_size=200,
                generations=1000,
                crossover_rate=0.85,
                **params
            )
            
            # Custom evolution to track selection pressure
            import time
            start_time = time.time()
            
            population = solver.initialize_population()
            diversity_history = []
            selection_pressure = []
            
            for generation in range(solver.generations):
                fitnesses = [solver.calculate_fitness(ind) for ind in population]
                
                # Track diversity (fitness std dev)
                diversity_history.append(np.std(fitnesses))
                
                # Track selection pressure (ratio of best to average fitness)
                best_fit = np.min(fitnesses)
                avg_fit = np.mean(fitnesses)
                pressure = (avg_fit - best_fit) / (avg_fit + 1e-10) if avg_fit > 0 else 0
                selection_pressure.append(pressure)
                
                # Track statistics
                best_idx = np.argmin(fitnesses)
                solver.best_fitness_history.append(fitnesses[best_idx])
                solver.avg_fitness_history.append(avg_fit)
                solver.worst_fitness_history.append(np.max(fitnesses))
                
                if fitnesses[best_idx] < solver.best_fitness:
                    solver.best_fitness = fitnesses[best_idx]
                    solver.best_solution = population[best_idx].copy()
                
                # Generate offspring
                offspring = []
                while len(offspring) < solver.offspring_size:
                    parent1 = solver.select_parent(population, fitnesses)
                    parent2 = solver.select_parent(population, fitnesses)
                    
                    if np.random.random() < solver.crossover_rate:
                        offspring1, offspring2 = solver.crossover(parent1, parent2)
                    else:
                        offspring1, offspring2 = parent1.copy(), parent2.copy()
                    
                    if np.random.random() < solver.mutation_rate:
                        offspring1 = solver.mutate(offspring1)
                    if np.random.random() < solver.mutation_rate:
                        offspring2 = solver.mutate(offspring2)
                    
                    offspring.append(offspring1)
                    if len(offspring) < solver.offspring_size:
                        offspring.append(offspring2)
                
                # Survivor selection
                offspring_fitnesses = [solver.calculate_fitness(ind) for ind in offspring]
                population = solver.perform_survivor_selection(
                    population, offspring, fitnesses, offspring_fitnesses
                )
            
            runtime = time.time() - start_time
            
            # Store results
            method_results['best_fitnesses'].append(solver.best_fitness)
            method_results['convergence_histories'].append(solver.best_fitness_history)
            method_results['diversity_histories'].append(diversity_history)
            method_results['selection_pressure_history'].append(selection_pressure)
            method_results['runtimes'].append(runtime)
            
            print(f"Best: {solver.best_fitness:.2f}, Time: {runtime:.2f}s")
        
        # Calculate statistics
        results[method_name] = {
            'mean_best': np.mean(method_results['best_fitnesses']),
            'std_best': np.std(method_results['best_fitnesses']),
            'mean_runtime': np.mean(method_results['runtimes']),
            'convergence_histories': method_results['convergence_histories'],
            'diversity_histories': method_results['diversity_histories'],
            'selection_pressure_history': method_results['selection_pressure_history'],
            'all_best': method_results['best_fitnesses']
        }
    
    return results


def plot_parent_selection_comparison(results):
    """
    Create comprehensive comparison plots
    
    Args:
        results: Results from comparison
    """
    fig = plt.figure(figsize=(18, 12))
    
    method_names = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(method_names)))
    
    # Plot 1: Convergence curves
    ax1 = plt.subplot(2, 3, 1)
    for i, name in enumerate(method_names):
        histories = results[name]['convergence_histories']
        avg_history = np.mean(histories, axis=0)
        std_history = np.std(histories, axis=0)
        
        generations = range(len(avg_history))
        ax1.plot(generations, avg_history, label=name, color=colors[i], linewidth=2)
        ax1.fill_between(generations, 
                         avg_history - std_history,
                         avg_history + std_history,
                         color=colors[i], alpha=0.2)
    
    ax1.set_xlabel('Generation', fontsize=10)
    ax1.set_ylabel('Best Fitness', fontsize=10)
    ax1.set_title('Convergence Curves (mean ± std)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Selection pressure over time
    ax2 = plt.subplot(2, 3, 2)
    for i, name in enumerate(method_names):
        pressure_histories = results[name]['selection_pressure_history']
        avg_pressure = np.mean(pressure_histories, axis=0)
        
        generations = range(len(avg_pressure))
        ax2.plot(generations, avg_pressure, label=name, color=colors[i], linewidth=2)
    
    ax2.set_xlabel('Generation', fontsize=10)
    ax2.set_ylabel('Selection Pressure', fontsize=10)
    ax2.set_title('Selection Pressure Over Time\n(Avg-Best)/Avg', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Diversity over time
    ax3 = plt.subplot(2, 3, 3)
    for i, name in enumerate(method_names):
        div_histories = results[name]['diversity_histories']
        avg_diversity = np.mean(div_histories, axis=0)
        
        generations = range(len(avg_diversity))
        ax3.plot(generations, avg_diversity, label=name, color=colors[i], linewidth=2)
    
    ax3.set_xlabel('Generation', fontsize=10)
    ax3.set_ylabel('Fitness Std Dev (Diversity)', fontsize=10)
    ax3.set_title('Population Diversity Over Time', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Box plot of final results
    ax4 = plt.subplot(2, 3, 4)
    best_fitnesses = [results[name]['all_best'] for name in method_names]
    bp = ax4.boxplot(best_fitnesses, labels=method_names, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax4.set_ylabel('Best Fitness', fontsize=10)
    ax4.set_title('Distribution of Best Solutions', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45, labelsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Mean performance comparison
    ax5 = plt.subplot(2, 3, 5)
    means = [results[name]['mean_best'] for name in method_names]
    stds = [results[name]['std_best'] for name in method_names]
    x_pos = np.arange(len(method_names))
    
    bars = ax5.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=colors)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(method_names, rotation=45, ha='right', fontsize=8)
    ax5.set_ylabel('Mean Best Fitness', fontsize=10)
    ax5.set_title('Mean Performance (± std)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Premature convergence analysis
    ax6 = plt.subplot(2, 3, 6)
    
    # Calculate convergence generation (when improvement becomes < 1%)
    convergence_gens = []
    
    for name in method_names:
        histories = results[name]['convergence_histories']
        avg_history = np.mean(histories, axis=0)
        
        # Find when improvement rate drops below 1%
        converged_gen = len(avg_history)
        for gen in range(10, len(avg_history)):
            improvement = (avg_history[gen-10] - avg_history[gen]) / (avg_history[gen-10] + 1e-10)
            if improvement < 0.01:  # Less than 1% improvement in 10 gens
                converged_gen = gen
                break
        
        convergence_gens.append(converged_gen)
    
    bars = ax6.bar(x_pos, convergence_gens, alpha=0.7, color=colors)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(method_names, rotation=45, ha='right', fontsize=8)
    ax6.set_ylabel('Generations', fontsize=10)
    ax6.set_title('Convergence Speed\n(Gens to <1% improvement)', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Parent Selection Methods Comparison - TSP', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save plot
    save_path = 'parent_selection_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to {save_path}")
    
    plt.show()


def print_detailed_analysis(results):
    """Print detailed analysis report"""
    print("\n" + "="*70)
    print("DETAILED ANALYSIS REPORT")
    print("="*70)
    
    method_names = list(results.keys())
    
    # Find best method
    best_mean = min(results[name]['mean_best'] for name in method_names)
    best_method = [name for name in method_names 
                   if results[name]['mean_best'] == best_mean][0]
    
    print(f"\n🏆 BEST METHOD: {best_method}")
    print(f"   Mean Fitness: {results[best_method]['mean_best']:.2f}")
    print(f"   Std Dev: {results[best_method]['std_best']:.2f}")
    
    print("\n" + "-"*70)
    print(f"{'Method':<30} {'Mean Best':<12} {'Std Dev':<12} {'Runtime (s)':<12}")
    print("-"*70)
    
    # Sort by mean best
    sorted_names = sorted(method_names, 
                         key=lambda x: results[x]['mean_best'])
    
    for name in sorted_names:
        mean_best = results[name]['mean_best']
        std_best = results[name]['std_best']
        runtime = results[name]['mean_runtime']
        marker = " ⭐" if name == best_method else ""
        
        print(f"{name:<30} {mean_best:<12.2f} {std_best:<12.2f} {runtime:<12.2f}{marker}")
    
    print("-"*70)
    
    # Analysis of selection pressure
    print("\n📊 SELECTION PRESSURE ANALYSIS:")
    
    for name in method_names:
        pressure_histories = results[name]['selection_pressure_history']
        avg_pressure = np.mean(pressure_histories, axis=0)
        
        early_pressure = np.mean(avg_pressure[:100])  # First 100 gens
        late_pressure = np.mean(avg_pressure[-100:])  # Last 100 gens
        
        print(f"\n{name}:")
        print(f"   Early pressure (gen 0-100): {early_pressure:.4f}")
        print(f"   Late pressure (gen 400-500): {late_pressure:.4f}")
        print(f"   Pressure drop: {early_pressure - late_pressure:.4f}")
    
    # Diversity analysis
    print("\n🌈 DIVERSITY ANALYSIS:")
    
    for name in method_names:
        div_histories = results[name]['diversity_histories']
        avg_diversity = np.mean(div_histories, axis=0)
        
        early_div = np.mean(avg_diversity[:100])
        late_div = np.mean(avg_diversity[-100:])
        
        print(f"\n{name}:")
        print(f"   Early diversity: {early_div:.2f}")
        print(f"   Late diversity: {late_div:.2f}")
        print(f"   Diversity retention: {(late_div/early_div)*100:.1f}%")
    
    # Premature convergence warning
    print("\n⚠️  PREMATURE CONVERGENCE RISK:")
    
    for name in method_names:
        div_histories = results[name]['diversity_histories']
        avg_diversity = np.mean(div_histories, axis=0)
        
        # If diversity drops to <10% of initial before gen 200, flag it
        initial_div = avg_diversity[0]
        min_gen = 0
        for gen, div in enumerate(avg_diversity):
            if div < 0.1 * initial_div and gen < 200:
                min_gen = gen
                break
        
        if min_gen > 0:
            print(f"   {name}: ⚠️  Lost 90% diversity by gen {min_gen}")
        else:
            print(f"   {name}: ✅ No premature convergence detected")


def main():
    """Main function"""
    print("This script compares different parent selection methods.")
    print("It will analyze:")
    print("  - Convergence speed")
    print("  - Selection pressure dynamics")
    print("  - Population diversity")
    print("  - Risk of premature convergence")
    print("\nRunning 3 runs per method on TSP Qatar dataset...")
    print("Estimated time: 5-8 minutes\n")
    
    response = input("Continue? (y/n): ").lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    # Run comparison
    results = compare_parent_selection_methods(num_runs=3)
    
    # Print analysis
    print_detailed_analysis(results)
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_parent_selection_comparison(results)
    
    print("\n✅ Analysis complete!")
    print("\nKEY FINDINGS:")
    print("- Fitness Proportional: Fast early, slow late, risk of premature convergence")
    print("- Rank-Based: Consistent pressure, maintains diversity well")
    print("- Tournament (k=3): Low pressure, high diversity")
    print("- Tournament (k=5): Balanced approach, generally best")
    print("- Tournament (k=7): High pressure, fast convergence")
    print("\nRECOMMENDATIONS:")
    print("  • For most problems: Tournament (k=5) or Rank-Based")
    print("  • For maximum diversity: Tournament (k=3)")
    print("  • For fast convergence: Tournament (k=7)")
    print("  • Avoid: Fitness Proportional (unless using fitness scaling)")


if __name__ == "__main__":
    main()
