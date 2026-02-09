"""
Master script to run both EA problems
Runs TSP and Exam Scheduling solvers with default parameters
"""

import os
import sys

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")

def run_tsp():
    """Run TSP solver"""
    print_header("PROBLEM 1: TRAVELLING SALESMAN PROBLEM")
    
    os.chdir("TSP")
    exec(open("tsp_solver.py").read())
    os.chdir("..")

def run_exam_scheduling():
    """Run Exam Scheduling solver"""
    print_header("PROBLEM 2: EXAM SCHEDULING")
    
    os.chdir("ExamScheduling")
    exec(open("exam_scheduler.py").read())
    os.chdir("..")

def main():
    """Main function to run both problems"""
    print_header("EVOLUTIONARY ALGORITHMS ASSIGNMENT")
    print("This script will run both problems sequentially:")
    print("  1. TSP - Qatar Dataset (194 cities)")
    print("  2. Exam Scheduling - Sample Dataset")
    print("\nEach problem will:")
    print("  - Run the EA optimization")
    print("  - Generate convergence plots")
    print("  - Generate problem-specific visualizations")
    print("  - Save detailed results to text files")
    
    print("\n" + "-"*70)
    print("Estimated total runtime: 5-10 minutes")
    print("-"*70)
    
    response = input("\nContinue? (y/n): ").lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    try:
        # Run TSP
        run_tsp()
        
        # Run Exam Scheduling
        run_exam_scheduling()
        
        # Summary
        print_header("ALL TASKS COMPLETED")
        print("Generated files:")
        print("\nTSP/")
        print("  ├── TSP_convergence.png")
        print("  ├── TSP_best_tour.png")
        print("  └── TSP_results.txt")
        print("\nExamScheduling/")
        print("  ├── ExamScheduling_convergence.png")
        print("  ├── ExamScheduling_distribution.png")
        print("  └── ExamScheduling_results.txt")
        
        print("\nCheck the output files for detailed results!")
        print("Review the plots to analyze convergence and solution quality.")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("Please check the error message and try running problems individually.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
