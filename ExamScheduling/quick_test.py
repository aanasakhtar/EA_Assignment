"""
Quick Test Script for Exam Scheduling Problem
This runs a reduced version for faster demonstration
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exam_scheduler import (
    ExamSchedulingSolver, 
    parse_purdue_data,
    plot_comparison,
    print_detailed_analysis,
    save_results_to_file
)
import numpy as np
import time

def quick_test(k_runs=3, generations=20):
    """
    Run a quick test with reduced parameters for demonstration
    
    Args:
        k_runs: Number of runs per scheme (default 3 instead of 10)
        generations: Number of generations (default 20 instead of 50)
    """
    print("="*80)
    print(" "*20 + "QUICK TEST MODE")
    print(" "*15 + "Reduced parameters for faster execution")
    print("="*80)
    print(f"\nParameters: K={k_runs} runs, {generations} generations per run")
    print("(Full assignment uses K=10 runs, 50 generations)\n")
    
    # Load data
    xml_path = os.path.join(os.path.dirname(__file__), 'pu-exam-spr12.xml')
    exams, students, periods = parse_purdue_data(xml_path)
    
    print(f"\nDataset: {len(exams)} exams, {len(students)} students, {periods} timeslots")
    
    # Test schemes
    schemes = [
        (('fitness_proportional', 'generational'), 'FPS + Generational'),
        (('tournament', 'generational'), 'Tournament + Generational'),
        (('random', 'generational'), 'Random + Generational')
    ]
    
    results = {}
    
    for (p_scheme, s_scheme), label in schemes:
        print(f"\n{'='*70}")
        print(f"Testing: {label}")
        print(f"{'='*70}")
        
        all_bsf = []
        all_final = []
        all_times = []
        
        for run in range(k_runs):
            print(f"  Run {run+1}/{k_runs}...", end=' ', flush=True)
            
            start = time.time()
            solver = ExamSchedulingSolver(
                exams=exams,
                students=students,
                timeslots=periods,
                constraints={},
                population_size=30,
                offspring_size=10,
                generations=generations,  # Reduced
                mutation_rate=0.5,
                crossover_rate=0.8,
                elitism_count=2,
                tournament_size=2,
                parent_selection=p_scheme,
                survivor_selection=s_scheme
            )
            solver.evolve(verbose=False)
            elapsed = time.time() - start
            
            all_bsf.append(solver.best_fitness_history)
            all_final.append(solver.best_fitness)
            all_times.append(elapsed)
            
            print(f"Fitness={solver.best_fitness:.2f}, Time={elapsed:.1f}s")
        
        # Store results
        results[label] = {
            'all_bsf_histories': all_bsf,
            'avg_bsf_history': np.mean(all_bsf, axis=0),
            'std_bsf_history': np.std(all_bsf, axis=0),
            'all_final_fitness': all_final,
            'mean_final_fitness': np.mean(all_final),
            'std_final_fitness': np.std(all_final),
            'best_final_fitness': np.min(all_final),
            'worst_final_fitness': np.max(all_final),
            'mean_runtime': np.mean(all_times),
            'total_runtime': np.sum(all_times)
        }
        
        print(f"  Mean: {results[label]['mean_final_fitness']:.2f} ± {results[label]['std_final_fitness']:.2f}")
    
    # Analysis
    print("\n" + "="*80)
    print("QUICK TEST RESULTS")
    print("="*80)
    
    print_detailed_analysis(results)
    plot_comparison(results, 'exam_scheduling_quick_test.png')
    save_results_to_file(results, 'exam_scheduling_quick_test_results.txt')
    
    print("\n✅ Quick test complete!")
    print(f"\nTotal time: {sum(r['total_runtime'] for r in results.values()):.1f}s")
    print("\nTo run full assignment version (K=10, 50 generations):")
    print("  python ExamScheduling/exam_scheduler.py")
    

if __name__ == "__main__":
    quick_test(k_runs=3, generations=20)
