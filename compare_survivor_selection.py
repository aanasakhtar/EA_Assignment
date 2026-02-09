"""
Compare Different Survivor Selection Strategies
Demonstrates the impact of survivor selection on EA performance
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from TSP.tsp_solver import TSPSolver, load_qatar_dataset


def compare_survivor_selection_strategies(num_runs=3):
    """
    Compare different survivor selection strategies on TSP
    
    Args:
        num_runs: Number of runs per strategy
        
    Returns:
        Results dictionary
    """
    print("="*70)
    print("SURVIVOR SELECTION STRATEGY COMPARISON")
    print("="*70)
    
    # Load dataset
    print("\nLoading Qatar TSP dataset...")
    cities, distance_matrix = load_qatar_dataset()
    
    # Define strategies to compare
    strategies = {
        'Generational (Elitism=5)': {
            'survivor_selection': 'generational',
            'population_size': 100,
            'offspring_size': 100,
            'elitism_count': 5,
            'generations': 500
        },
        '(μ+λ): (100+100)': {
            'survivor_selection': 'mu_plus_lambda',
            'population_size': 100,
            'offspring_size': 100,
            'generations': 500
        },
        '(μ+λ): (100+200)': {
            'survivor_selection': 'mu_plus_lambda',
            'population_size': 100,
            'offspring_size': 200,
            'generations': 500
        },
        '(μ,λ): (100,200)': {
            'survivor_selection': 'mu_comma_lambda',
            'population_size': 100,
            'offspring_size': 200,
            'mutation_rate': 0.2,  # Higher mutation for comma selection
            'generations': 500
        },
        '(μ,λ): (100,500)': {
            'survivor_selection': 'mu_comma_lambda',
            'population_size': 100,
            'offspring_size': 500,
            'mutation_rate': 0.25,
            'generations': 500
        },
        'Tournament Survivor': {
            'survivor_selection': 'tournament',
            'population_size': 100,
            'offspring_size': 150,
            'generations': 500
        }
    }
    
    results = {}
    
    for strategy_name, params in strategies.items():
        print(f"\n{'='*70}")
        print(f"Testing: {strategy_name}")
        print(f"{'='*70}")
        print(f"Parameters: {params}")
        
        strategy_results = {
            'best_fitnesses': [],
            'convergence_histories': [],
            'diversity_histories': [],
            'runtimes': []
        }
        
        for run in range(num_runs):
            print(f"\n  Run {run + 1}/{num_runs}...", end=' ')
            
            # Create solver
            solver = TSPSolver(
                cities=cities,
                distance_matrix=distance_matrix,
                crossover_rate=0.85,
                **params
            )
            
            # Track diversity
            diversity_history = []
            
            # Custom evolve to track diversity
            import time
            start_time = time.time()
            
            population = solver.initialize_population()
            
            for generation in range(solver.generations):
                fitnesses = [solver.calculate_fitness(ind) for ind in population]
                
                # Track diversity (fitness variance)
                diversity_history.append(np.std(fitnesses))
                
                # Track statistics
                best_idx = np.argmin(fitnesses)
                solver.best_fitness_history.append(fitnesses[best_idx])
                solver.avg_fitness_history.append(np.mean(fitnesses))
                solver.worst_fitness_history.append(np.max(fitnesses))
                
                if fitnesses[best_idx] < solver.best_fitness:
                    solver.best_fitness = fitnesses[best_idx]
                    solver.best_solution = population[best_idx].copy()
                
                # Generate offspring
                offspring = []
                while len(offspring) < solver.offspring_size:
                    parent1 = solver.tournament_selection(population, fitnesses)
                    parent2 = solver.tournament_selection(population, fitnesses)
                    
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
            strategy_results['best_fitnesses'].append(solver.best_fitness)
            strategy_results['convergence_histories'].append(solver.best_fitness_history)
            strategy_results['diversity_histories'].append(diversity_history)
            strategy_results['runtimes'].append(runtime)
            
            print(f"Best: {solver.best_fitness:.2f}, Time: {runtime:.2f}s")
        
        # Calculate statistics
        results[strategy_name] = {
            'mean_best': np.mean(strategy_results['best_fitnesses']),
            'std_best': np.std(strategy_results['best_fitnesses']),
            'mean_runtime': np.mean(strategy_results['runtimes']),
            'convergence_histories': strategy_results['convergence_histories'],
            'diversity_histories': strategy_results['diversity_histories'],
            'all_best': strategy_results['best_fitnesses']
        }
    
    return results


def plot_survivor_selection_comparison(results):
    """
    Create comprehensive comparison plots
    
    Args:
        results: Results from comparison
    """
    fig = plt.figure(figsize=(18, 12))
    
    strategy_names = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategy_names)))
    
    # Plot 1: Convergence comparison
    ax1 = plt.subplot(2, 3, 1)
    for i, name in enumerate(strategy_names):
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
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Diversity over time
    ax2 = plt.subplot(2, 3, 2)
    for i, name in enumerate(strategy_names):
        div_histories = results[name]['diversity_histories']
        avg_diversity = np.mean(div_histories, axis=0)
        
        generations = range(len(avg_diversity))
        ax2.plot(generations, avg_diversity, label=name, color=colors[i], linewidth=2)
    
    ax2.set_xlabel('Generation', fontsize=10)
    ax2.set_ylabel('Fitness Std Dev (Diversity)', fontsize=10)
    ax2.set_title('Population Diversity Over Time', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Box plot of final results
    ax3 = plt.subplot(2, 3, 3)
    best_fitnesses = [results[name]['all_best'] for name in strategy_names]
    bp = ax3.boxplot(best_fitnesses, labels=strategy_names, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax3.set_ylabel('Best Fitness', fontsize=10)
    ax3.set_title('Distribution of Best Solutions', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45, labelsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Mean best fitness with error bars
    ax4 = plt.subplot(2, 3, 4)
    means = [results[name]['mean_best'] for name in strategy_names]
    stds = [results[name]['std_best'] for name in strategy_names]
    x_pos = np.arange(len(strategy_names))
    
    bars = ax4.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=colors)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(strategy_names, rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('Mean Best Fitness', fontsize=10)
    ax4.set_title('Mean Performance (± std)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Convergence speed (generations to 95% of final)
    ax5 = plt.subplot(2, 3, 5)
    convergence_gens = []
    
    for name in strategy_names:
        histories = results[name]['convergence_histories']
        avg_history = np.mean(histories, axis=0)
        
        final_best = avg_history[-1]
        target = avg_history[0] - 0.95 * (avg_history[0] - final_best)
        
        # Find generation where target is reached
        converged_gen = len(avg_history)
        for gen, fitness in enumerate(avg_history):
            if fitness <= target:
                converged_gen = gen
                break
        
        convergence_gens.append(converged_gen)
    
    bars = ax5.bar(x_pos, convergence_gens, alpha=0.7, color=colors)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(strategy_names, rotation=45, ha='right', fontsize=8)
    ax5.set_ylabel('Generations', fontsize=10)
    ax5.set_title('Convergence Speed (to 95% of final)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Computational efficiency
    ax6 = plt.subplot(2, 3, 6)
    runtimes = [results[name]['mean_runtime'] for name in strategy_names]
    
    bars = ax6.bar(x_pos, runtimes, alpha=0.7, color=colors)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(strategy_names, rotation=45, ha='right', fontsize=8)
    ax6.set_ylabel('Runtime (seconds)', fontsize=10)
    ax6.set_title('Computational Cost', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Survivor Selection Strategy Comparison - TSP', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save plot
    save_path = 'survivor_selection_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to {save_path}")
    
    plt.show()


def print_detailed_report(results):
    """Print detailed analysis report"""
    print("\n" + "="*70)
    print("DETAILED ANALYSIS REPORT")
    print("="*70)
    
    strategy_names = list(results.keys())
    
    # Find best strategy
    best_mean = min(results[name]['mean_best'] for name in strategy_names)
    best_strategy = [name for name in strategy_names 
                    if results[name]['mean_best'] == best_mean][0]
    
    print(f"\n🏆 BEST STRATEGY: {best_strategy}")
    print(f"   Mean Fitness: {results[best_strategy]['mean_best']:.2f}")
    print(f"   Std Dev: {results[best_strategy]['std_best']:.2f}")
    print(f"   Runtime: {results[best_strategy]['mean_runtime']:.2f}s")
    
    print("\n" + "-"*70)
    print(f"{'Strategy':<30} {'Mean Best':<12} {'Std Dev':<12} {'Runtime (s)':<12}")
    print("-"*70)
    
    # Sort by mean best
    sorted_names = sorted(strategy_names, 
                         key=lambda x: results[x]['mean_best'])
    
    for name in sorted_names:
        mean_best = results[name]['mean_best']
        std_best = results[name]['std_best']
        runtime = results[name]['mean_runtime']
        marker = " ⭐" if name == best_strategy else ""
        
        print(f"{name:<30} {mean_best:<12.2f} {std_best:<12.2f} {runtime:<12.2f}{marker}")
    
    print("-"*70)
    
    # Analysis
    print("\n📊 KEY INSIGHTS:")
    print("\n1. Solution Quality:")
    for i, name in enumerate(sorted_names[:3]):
        print(f"   {i+1}. {name}: {results[name]['mean_best']:.2f}")
    
    print("\n2. Reliability (Low Std Dev = More Consistent):")
    reliability_sorted = sorted(strategy_names, 
                               key=lambda x: results[x]['std_best'])
    for i, name in enumerate(reliability_sorted[:3]):
        print(f"   {i+1}. {name}: {results[name]['std_best']:.2f}")
    
    print("\n3. Computational Efficiency (Fastest):")
    speed_sorted = sorted(strategy_names, 
                         key=lambda x: results[x]['mean_runtime'])
    for i, name in enumerate(speed_sorted[:3]):
        print(f"   {i+1}. {name}: {results[name]['mean_runtime']:.2f}s")
    
    # Trade-off analysis
    print("\n4. Trade-offs:")
    for name in strategy_names:
        quality_rank = sorted_names.index(name) + 1
        speed_rank = speed_sorted.index(name) + 1
        print(f"   {name}:")
        print(f"      Quality Rank: {quality_rank}/{len(strategy_names)}")
        print(f"      Speed Rank: {speed_rank}/{len(strategy_names)}")


def main():
    """Main function"""
    print("This script compares different survivor selection strategies.")
    print("It will run multiple experiments and generate comprehensive analysis.")
    print("\nRunning 3 runs per strategy on TSP Qatar dataset...")
    print("Estimated time: 5-10 minutes\n")
    
    response = input("Continue? (y/n): ").lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    # Run comparison
    results = compare_survivor_selection_strategies(num_runs=3)
    
    # Print report
    print_detailed_report(results)
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_survivor_selection_comparison(results)
    
    print("\n✅ Analysis complete!")
    print("\nCONCLUSIONS:")
    print("- (μ+λ) strategies tend to be most reliable (elitist)")
    print("- (μ,λ) maintains more diversity but can be slower to converge")
    print("- Generational with elitism is computationally efficient")
    print("- Tournament survivor selection provides good balance")
    print("\nChoose based on your priorities:")
    print("  • Best solution quality → (μ+λ) with larger λ")
    print("  • Fastest convergence → Generational with elitism")
    print("  • Best diversity → (μ,λ) with large λ")
    print("  • Balanced approach → Tournament survivor selection")


if __name__ == "__main__":
    main()
