"""
Q2: EXAM SCHEDULING PROBLEM - MAIN RUNNER
Run this file to execute the complete assignment solution.

This will generate:
1. Generation tables (printed to console + CSV files)
2. Side-by-side plots for each combination (Avg BSF vs Avg ASF)
3. Combined comparison plot (all schemes)
4. Detailed results file
"""

import sys
import os

# Change to ExamScheduling directory and run the main solver
exam_scheduling_dir = os.path.join(os.path.dirname(__file__), 'ExamScheduling')
os.chdir(exam_scheduling_dir)
sys.path.insert(0, exam_scheduling_dir)

from exam_scheduler import main

if __name__ == "__main__":
    main()
