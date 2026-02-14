"""
Exam Scheduling Problem Solver using Evolutionary Algorithm
Based on Purdue University benchmark dataset (pu-exam-spr12.xml)

Q2: Exam Timetabling Problem Implementation

CHROMOSOME REPRESENTATION:
    - Integer array where index = exam ID (0 to n_exams-1)
    - Value = timeslot assignment (0 to n_timeslots-1)
    - Example: [2, 5, 1, 2, ...] means exam 0 at timeslot 2, exam 1 at timeslot 5, etc.

FITNESS FUNCTION:
    Minimization problem with two types of constraints:
    
    1. HARD CONSTRAINTS (must be satisfied):
       - No student can have two exams at the same timeslot
       - Penalty: 10,000 per violation
    
    2. SOFT CONSTRAINTS (should be minimized):
       - Spread penalty: Exams too close together for same student
         * Adjacent slots (gap=1): penalty 5 per occurrence
         * Two slots apart (gap=2): penalty 2 per occurrence
       - Balance penalty: Uneven distribution of exams across timeslots
         * Variance in exam counts per timeslot
       - Consecutive penalty: Students having back-to-back exams
         * Penalty: 200 per consecutive exam pair
    
    Total Fitness = (hard_violations x 10000) + (spread x 100) + (balance x 50) + (consecutive x 200)

GENETIC OPERATORS:
    - Crossover: Uniform crossover with conflict repair
    - Mutation: Random timeslot reassignment with conflict repair
    - Repair: Heuristic to eliminate hard constraint violations

SELECTION SCHEMES (as per assignment):
    Three combinations tested with K=10 runs each:
    1. Fitness Proportional Selection + Generational (with elitism)
    2. Tournament Selection + Generational (with elitism)
    3. Random Selection + Random Survivor Selection

FIXED PARAMETERS (as per assignment):
    - Population size (μ): 30
    - Offspring size (λ): 10
    - Generations: 50
    - Mutation rate: 0.5
"""

import numpy as np
import sys
import os

# Add parent directory to path to find common modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.ea_framework import EAFramework
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import time
from collections import defaultdict


class ExamSchedulingSolver(EAFramework):
    """
    Exam Scheduling solver using Evolutionary Algorithm
    
    Chromosome representation: Array where index is exam ID and value is timeslot
    """
    
    def __init__(self, exams, students, timeslots, constraints, **kwargs):
        """
        Initialize exam scheduling solver
        
        Args:
            exams: List of exam IDs
            students: Dict mapping student_id to list of exam_ids they're enrolled in
            timeslots: Number of available timeslots
            constraints: Dict with constraint parameters
            **kwargs: Additional EA parameters
        """
        super().__init__(**kwargs)
        self.exams = exams
        self.students = students
        self.n_exams = len(exams)
        self.n_timeslots = timeslots
        self.constraints = constraints
        
        # Build conflict matrix (exams that can't be at same time)
        self.conflict_matrix = self._build_conflict_matrix()
        
        # Note: student_exam_matrix not built for memory efficiency
        # All calculations use sparse structures instead
        
    def _build_conflict_matrix(self):
        """
        Build sparse conflict structure (exams that can't be at same time)
        Uses a dictionary for memory efficiency instead of full matrix
        
        Returns:
            Dictionary mapping exam_id to set of conflicting exam_ids
        """
        conflict_dict = defaultdict(set)
        
        for student_exams in self.students.values():
            # Each pair of exams taken by same student creates a conflict
            for i, exam1 in enumerate(student_exams):
                for exam2 in student_exams[i+1:]:
                    if exam1 < self.n_exams and exam2 < self.n_exams:
                        conflict_dict[exam1].add(exam2)
                        conflict_dict[exam2].add(exam1)
        
        # Convert defaultdict to regular dict for cleaner access
        return dict(conflict_dict)
    
    def initialize_population(self):
        """Initialize population with random valid schedules"""
        population = []
        
        for _ in range(self.population_size):
            # Random assignment of timeslots to exams
            schedule = np.random.randint(0, self.n_timeslots, self.n_exams)
            
            # Try to repair conflicts
            schedule = self._repair_conflicts(schedule)
            
            population.append(schedule)
        
        return population
    
    def _repair_conflicts(self, schedule):
        """
        Repair hard constraint violations (exam conflicts)
        Uses sparse conflict structure for efficiency
        
        Args:
            schedule: Exam schedule array
            
        Returns:
            Repaired schedule
        """
        schedule = schedule.copy()
        max_repairs = 50  # Reduced for large datasets
        repairs = 0
        
        while repairs < max_repairs:
            conflicts_found = False
            
            # Check conflicts using sparse structure
            for exam_id, conflicting_exams in self.conflict_matrix.items():
                exam_slot = schedule[exam_id]
                
                for conflicting_exam in conflicting_exams:
                    # If exams conflict and are in same timeslot
                    if schedule[conflicting_exam] == exam_slot:
                        # Move conflicting exam to a different timeslot
                        # Try to find a slot that doesn't conflict with this exam's other conflicts
                        available_slots = list(range(self.n_timeslots))
                        np.random.shuffle(available_slots)
                        
                        # Pick a random available slot
                        for slot in available_slots:
                            if slot != exam_slot:
                                schedule[conflicting_exam] = slot
                                conflicts_found = True
                                break
                        break  # Move to next exam after one repair
                
                if conflicts_found:
                    break
            
            if not conflicts_found:
                break
            
            repairs += 1
        
        return schedule
    
    def calculate_fitness(self, schedule):
        """
        Calculate fitness (penalty) of a schedule
        Lower is better
        
        Fitness = hard_constraint_penalty + soft_constraint_penalty
        
        Hard constraints (high penalty):
        - Exam conflicts (same student, same timeslot)
        
        Soft constraints (lower penalty):
        - Exams too close together for same student
        - Uneven distribution of exams across timeslots
        - Student preference violations
        """
        penalty = 0
        
        # Hard constraint: Exam conflicts
        hard_penalty = self._calculate_hard_constraints(schedule)
        penalty += hard_penalty * 10000  # Very high penalty
        
        # Soft constraint: Exams spread
        spread_penalty = self._calculate_spread_penalty(schedule)
        penalty += spread_penalty * 100
        
        # Soft constraint: Load balancing
        balance_penalty = self._calculate_balance_penalty(schedule)
        penalty += balance_penalty * 50
        
        # Soft constraint: Consecutive exams
        consecutive_penalty = self._calculate_consecutive_penalty(schedule)
        penalty += consecutive_penalty * 200
        
        return penalty
    
    def _calculate_hard_constraints(self, schedule):
        """
        Calculate hard constraint violations (conflicts)
        Uses sparse conflict structure for efficiency
        """
        violations = 0
        
        # Check each exam's conflicts
        for exam_id, conflicting_exams in self.conflict_matrix.items():
            exam_slot = schedule[exam_id]
            
            # Count how many conflicting exams are in the same timeslot
            for conflicting_exam in conflicting_exams:
                if conflicting_exam > exam_id:  # Only count each pair once
                    if schedule[conflicting_exam] == exam_slot:
                        violations += 1
        
        return violations
    
    def _calculate_spread_penalty(self, schedule):
        """
        Calculate penalty for exams that are too close together
        Students should have exams well-spaced
        """
        penalty = 0
        proximity_weights = {
            0: 10,  # Same slot (should be caught by hard constraint)
            1: 5,   # Adjacent slots
            2: 2,   # Two slots apart
        }
        
        for student_exams in self.students.values():
            exam_timeslots = [schedule[exam] for exam in student_exams 
                            if exam < self.n_exams]
            exam_timeslots.sort()
            
            for i in range(len(exam_timeslots)-1):
                gap = exam_timeslots[i+1] - exam_timeslots[i]
                if gap in proximity_weights:
                    penalty += proximity_weights[gap]
        
        return penalty
    
    def _calculate_balance_penalty(self, schedule):
        """
        Calculate penalty for uneven distribution of exams
        Exams should be relatively evenly distributed across timeslots
        """
        # Count exams per timeslot
        exams_per_slot = np.bincount(schedule, minlength=self.n_timeslots)
        
        # Calculate variance (penalize uneven distribution)
        mean_exams = self.n_exams / self.n_timeslots
        variance = np.var(exams_per_slot)
        
        return variance
    
    def _calculate_consecutive_penalty(self, schedule):
        """
        Calculate penalty for students having consecutive exams
        """
        penalty = 0
        
        for student_exams in self.students.values():
            if len(student_exams) < 2:
                continue
                
            exam_timeslots = [schedule[exam] for exam in student_exams 
                            if exam < self.n_exams]
            exam_timeslots.sort()
            
            # Count consecutive exams
            for i in range(len(exam_timeslots)-1):
                if exam_timeslots[i+1] - exam_timeslots[i] == 1:
                    penalty += 1
        
        return penalty
    
    def crossover(self, parent1, parent2):
        """
        Uniform crossover for exam scheduling
        
        Args:
            parent1, parent2: Parent schedules
            
        Returns:
            Two offspring schedules
        """
        size = len(parent1)
        
        # Create mask for uniform crossover
        mask = np.random.randint(0, 2, size)
        
        offspring1 = np.where(mask, parent1, parent2)
        offspring2 = np.where(mask, parent2, parent1)
        
        # Repair conflicts
        offspring1 = self._repair_conflicts(offspring1)
        offspring2 = self._repair_conflicts(offspring2)
        
        return offspring1, offspring2
    
    def mutate(self, schedule):
        """
        Mutate schedule by changing timeslot of random exams
        
        Strategy: Randomly select exams and assign them to new random timeslots,
        then repair any conflicts that arise.
        
        Args:
            schedule: Schedule to mutate
            
        Returns:
            Mutated schedule
        """
        schedule = schedule.copy()
        
        # Number of mutations (proportional to problem size)
        n_mutations = max(1, int(self.n_exams * 0.05))  # Mutate ~5% of exams
        
        for _ in range(n_mutations):
            # Select random exam
            exam_idx = np.random.randint(0, self.n_exams)
            
            # Assign new random timeslot
            schedule[exam_idx] = np.random.randint(0, self.n_timeslots)
        
        # Repair any conflicts introduced by mutation
        schedule = self._repair_conflicts(schedule)
        
        return schedule
    
    def analyze_schedule(self, schedule=None):
        """
        Analyze and print schedule quality metrics
        
        Args:
            schedule: Schedule to analyze (uses best if None)
        """
        if schedule is None:
            schedule = self.best_solution
        
        if schedule is None:
            print("No schedule to analyze!")
            return
        
        print("\n" + "="*60)
        print("SCHEDULE ANALYSIS")
        print("="*60)
        
        # Hard constraints
        hard_violations = self._calculate_hard_constraints(schedule)
        print(f"\nHard Constraint Violations: {hard_violations}")
        if hard_violations == 0:
            print("✓ All hard constraints satisfied!")
        else:
            print("✗ Schedule has conflicts that need resolution")
        
        # Soft constraints
        print(f"\nSoft Constraint Metrics:")
        print(f"  Spread Penalty: {self._calculate_spread_penalty(schedule):.2f}")
        print(f"  Balance Penalty: {self._calculate_balance_penalty(schedule):.2f}")
        print(f"  Consecutive Exams Penalty: {self._calculate_consecutive_penalty(schedule):.2f}")
        
        # Distribution
        exams_per_slot = np.bincount(schedule, minlength=self.n_timeslots)
        print(f"\nExam Distribution:")
        print(f"  Total Timeslots: {self.n_timeslots}")
        print(f"  Average exams per slot: {self.n_exams/self.n_timeslots:.2f}")
        print(f"  Min exams in a slot: {exams_per_slot.min()}")
        print(f"  Max exams in a slot: {exams_per_slot.max()}")
        
        # Student impact
        max_consecutive = 0
        total_consecutive = 0
        
        for student_exams in self.students.values():
            exam_timeslots = sorted([schedule[exam] for exam in student_exams 
                                   if exam < self.n_exams])
            
            consecutive = 0
            for i in range(len(exam_timeslots)-1):
                if exam_timeslots[i+1] - exam_timeslots[i] == 1:
                    consecutive += 1
            
            total_consecutive += consecutive
            max_consecutive = max(max_consecutive, consecutive)
        
        print(f"\nStudent Impact:")
        print(f"  Students with consecutive exams: {total_consecutive}")
        print(f"  Max consecutive exams for a student: {max_consecutive}")
        
    def plot_schedule_distribution(self, schedule=None, save_path=None):
        """Plot distribution of exams across timeslots"""
        if schedule is None:
            schedule = self.best_solution
        
        if schedule is None:
            print("No schedule to plot!")
            return
        
        exams_per_slot = np.bincount(schedule, minlength=self.n_timeslots)
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(self.n_timeslots), exams_per_slot, color='steelblue', alpha=0.7)
        plt.axhline(y=self.n_exams/self.n_timeslots, color='red', 
                   linestyle='--', label=f'Average ({self.n_exams/self.n_timeslots:.1f})')
        plt.xlabel('Timeslot', fontsize=12)
        plt.ylabel('Number of Exams', fontsize=12)
        plt.title('Exam Distribution Across Timeslots', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution plot saved to {save_path}")
        
        plt.show()


import xml.etree.ElementTree as ET

def parse_purdue_data(file_path):
    """
    Parse Purdue University exam scheduling XML dataset
    
    Args:
        file_path: Path to the XML file
        
    Returns:
        exams: List of exam IDs (0, 1, 2, ..., n-1)
        students: Dict mapping student_id to list of exam IDs they're enrolled in
        periods: Number of available timeslots
    """
    print(f"Parsing dataset: {file_path}")
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # 1. Get Periods (Timeslots) FIRST - from <periods> section
    period_elements = root.findall('./periods/period')
    periods = len(period_elements)
    print(f"  Found {periods} timeslots (periods)")
    
    # 2. Map Exams to simple IDs (0, 1, 2, ...) - from <exams> section
    exams = []
    exam_id_map = {}
    exam_elements = root.findall('./exams/exam')
    
    for i, exam in enumerate(exam_elements):
        ext_id = exam.get('id')
        exams.append(i)
        exam_id_map[ext_id] = i
    
    print(f"  Found {len(exams)} exams")
    
    # 3. Map Students to the list of Exams they take - from <students> section
    students = {}
    student_elements = root.findall('./students/student')
    
    for student in student_elements:
        s_id = student.get('id')
        student_exams = []
        for ref in student.findall('exam'):
            ext_exam_id = ref.get('id')
            if ext_exam_id in exam_id_map:
                student_exams.append(exam_id_map[ext_exam_id])
        if student_exams:  # Only add students with valid exams
            students[s_id] = student_exams
    
    print(f"  Found {len(students)} students")
    
    return exams, students, periods


def run_k_runs(parent_scheme, survivor_scheme, exams, students, periods, k=10):
    """
    Execute the EA K times as required by Assignment to get statistical results
    
    Args:
        parent_scheme: Parent selection method ('fitness_proportional', 'tournament', 'random')
        survivor_scheme: Survivor selection method ('generational', 'random')
        exams: List of exam IDs
        students: Student-exam enrollment dict
        periods: Number of timeslots
        k: Number of runs (default 10 as per assignment)
        
    Returns:
        Dictionary with results including average BSF history and statistics
    """
    all_runs_bsf = []
    all_final_fitness = []
    all_runtimes = []
    
    print(f"\n{'='*70}")
    print(f"Running: Parent={parent_scheme} + Survivor={survivor_scheme}")
    print(f"{'='*70}")
    
    for run in range(k):
        print(f"  Run {run+1}/{k}...", end=' ', flush=True)
        
        start_time = time.time()
        
        # Create solver with assignment-specific parameters
        solver = ExamSchedulingSolver(
            exams=exams, 
            students=students, 
            timeslots=periods,
            constraints={},
            population_size=30,          
            offspring_size=10,           
            generations=50,              
            mutation_rate=0.5,           
            crossover_rate=0.8,           # Standard value
            elitism_count=2,              # Small elitism for generational
            tournament_size=2,           
            parent_selection=parent_scheme,
            survivor_selection=survivor_scheme
        )
        
        # Run evolution
        solver.evolve(verbose=False)
        
        runtime = time.time() - start_time
        
        # Store results
        all_runs_bsf.append(solver.best_fitness_history)
        all_final_fitness.append(solver.best_fitness)
        all_runtimes.append(runtime)
        
        print(f"Best Fitness = {solver.best_fitness:.2f}, Time = {runtime:.2f}s")
    
    # Calculate statistics
    avg_bsf_history = np.mean(all_runs_bsf, axis=0)
    std_bsf_history = np.std(all_runs_bsf, axis=0)
    
    results = {
        'all_bsf_histories': all_runs_bsf,
        'avg_bsf_history': avg_bsf_history,
        'std_bsf_history': std_bsf_history,
        'all_final_fitness': all_final_fitness,
        'mean_final_fitness': np.mean(all_final_fitness),
        'std_final_fitness': np.std(all_final_fitness),
        'best_final_fitness': np.min(all_final_fitness),
        'worst_final_fitness': np.max(all_final_fitness),
        'mean_runtime': np.mean(all_runtimes),
        'total_runtime': np.sum(all_runtimes)
    }
    
    print(f"\n  Summary Statistics:")
    print(f"    Mean Final Fitness: {results['mean_final_fitness']:.2f} ± {results['std_final_fitness']:.2f}")
    print(f"    Best Final Fitness: {results['best_final_fitness']:.2f}")
    print(f"    Worst Final Fitness: {results['worst_final_fitness']:.2f}")
    print(f"    Mean Runtime: {results['mean_runtime']:.2f}s")
    
    return results


def plot_comparison(results_dict, save_path='exam_scheduling_comparison.png'):
    """
    Create comprehensive comparison plots for different selection schemes
    
    Args:
        results_dict: Dictionary with results from different schemes
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    schemes = list(results_dict.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(schemes)))
    
    # Plot 1: Average BSF Convergence Curves
    ax1 = axes[0, 0]
    for i, (scheme, results) in enumerate(results_dict.items()):
        avg_history = results['avg_bsf_history']
        std_history = results['std_bsf_history']
        generations = range(len(avg_history))
        
        ax1.plot(generations, avg_history, label=scheme, color=colors[i], linewidth=2)
        ax1.fill_between(generations, 
                         avg_history - std_history,
                         avg_history + std_history,
                         color=colors[i], alpha=0.2)
    
    ax1.set_xlabel('Generation', fontsize=11)
    ax1.set_ylabel('Average Best-So-Far (BSF) Fitness', fontsize=11)
    ax1.set_title('Convergence Curves (Mean ± Std Dev)\nLower is Better', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box Plot of Final Fitness Values
    ax2 = axes[0, 1]
    final_fitness_data = [results['all_final_fitness'] for results in results_dict.values()]
    bp = ax2.boxplot(final_fitness_data, labels=schemes, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax2.set_ylabel('Final Best Fitness', fontsize=11)
    ax2.set_title('Distribution of Final Solutions\nLower is Better', 
                  fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15, labelsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Mean Final Fitness Comparison
    ax3 = axes[1, 0]
    means = [results['mean_final_fitness'] for results in results_dict.values()]
    stds = [results['std_final_fitness'] for results in results_dict.values()]
    x_pos = np.arange(len(schemes))
    
    bars = ax3.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=colors)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(schemes, rotation=15, ha='right', fontsize=9)
    ax3.set_ylabel('Mean Final Fitness (± Std Dev)', fontsize=11)
    ax3.set_title('Mean Performance Comparison\nLower is Better', 
                  fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Computational Efficiency
    ax4 = axes[1, 1]
    runtimes = [results['mean_runtime'] for results in results_dict.values()]
    
    bars = ax4.bar(x_pos, runtimes, alpha=0.7, color=colors)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(schemes, rotation=15, ha='right', fontsize=9)
    ax4.set_ylabel('Mean Runtime per Run (seconds)', fontsize=11)
    ax4.set_title('Computational Efficiency', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Exam Scheduling: Selection Schemes Comparison (Purdue Spr12 Dataset)\n' +
                f'Parameters: μ=30, λ=10, Generations=50, K=10 runs per scheme',
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {save_path}")
    
    plt.show()


def print_detailed_analysis(results_dict):
    """
    Print detailed statistical analysis of all selection schemes
    
    Args:
        results_dict: Dictionary with results from different schemes
    """
    print("\n" + "="*80)
    print(" "*20 + "DETAILED ANALYSIS REPORT")
    print("="*80)
    
    schemes = list(results_dict.keys())
    
    # Find best scheme
    best_mean_fitness = min(results['mean_final_fitness'] for results in results_dict.values())
    best_scheme = [name for name, results in results_dict.items() 
                   if results['mean_final_fitness'] == best_mean_fitness][0]
    
    print(f"\n🏆 BEST SCHEME (by mean fitness): {best_scheme}")
    print(f"   Mean Final Fitness: {results_dict[best_scheme]['mean_final_fitness']:.2f}")
    print(f"   Std Deviation: {results_dict[best_scheme]['std_final_fitness']:.2f}")
    print(f"   Best Run: {results_dict[best_scheme]['best_final_fitness']:.2f}")
    print(f"   Worst Run: {results_dict[best_scheme]['worst_final_fitness']:.2f}")
    
    print("\n" + "-"*80)
    print(f"{'Selection Scheme':<40} {'Mean':<12} {'Std Dev':<12} {'Best':<12} {'Runtime(s)':<12}")
    print("-"*80)
    
    # Sort by mean final fitness (lower is better)
    sorted_schemes = sorted(schemes, 
                           key=lambda x: results_dict[x]['mean_final_fitness'])
    
    for i, scheme in enumerate(sorted_schemes):
        results = results_dict[scheme]
        marker = " ⭐" if scheme == best_scheme else ""
        rank = f"#{i+1}"
        
        print(f"{rank:<5} {scheme:<35} "
              f"{results['mean_final_fitness']:<12.2f} "
              f"{results['std_final_fitness']:<12.2f} "
              f"{results['best_final_fitness']:<12.2f} "
              f"{results['mean_runtime']:<12.2f}{marker}")
    
    print("-"*80)
    
    # Statistical comparisons
    print("\n PERFORMANCE ANALYSIS:")
    
    for i, scheme in enumerate(sorted_schemes):
        if i == 0:
            print(f"\n  1. {scheme} (BEST)")
            print(f"     - Baseline for comparison")
        else:
            baseline = results_dict[sorted_schemes[0]]['mean_final_fitness']
            current = results_dict[scheme]['mean_final_fitness']
            difference = current - baseline
            percent_worse = (difference / baseline) * 100
            
            print(f"\n  {i+1}. {scheme}")
            print(f"     - {difference:.2f} worse than best ({percent_worse:.2f}% higher)")
    
    # Convergence analysis
    print("\n CONVERGENCE ANALYSIS:")
    
    for scheme in schemes:
        results = results_dict[scheme]
        avg_history = results['avg_bsf_history']
        
        initial_fitness = avg_history[0]
        final_fitness = avg_history[-1]
        improvement = initial_fitness - final_fitness
        improvement_pct = (improvement / initial_fitness) * 100
        
        print(f"\n  {scheme}:")
        print(f"    Initial Avg Fitness: {initial_fitness:.2f}")
        print(f"    Final Avg Fitness: {final_fitness:.2f}")
        print(f"    Total Improvement: {improvement:.2f} ({improvement_pct:.2f}%)")
    
    # Reliability analysis
    print("\n RELIABILITY ANALYSIS (consistency across runs):")
    
    reliability_sorted = sorted(schemes, 
                               key=lambda x: results_dict[x]['std_final_fitness'])
    
    for i, scheme in enumerate(reliability_sorted):
        results = results_dict[scheme]
        cv = (results['std_final_fitness'] / results['mean_final_fitness']) * 100  # Coefficient of variation
        
        marker = "✓ Most Reliable" if i == 0 else ""
        
        print(f"\n  {i+1}. {scheme} {marker}")
        print(f"     Std Dev: {results['std_final_fitness']:.2f}")
        print(f"     Coefficient of Variation: {cv:.2f}%")
        print(f"     Range: {results['best_final_fitness']:.2f} - {results['worst_final_fitness']:.2f}")


def save_results_to_file(results_dict, filename='exam_scheduling_results.txt'):
    """
    Save detailed results to a text file
    
    Args:
        results_dict: Dictionary with results from different schemes
        filename: Output filename
    """
    with open(filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write(" "*15 + "EXAM SCHEDULING PROBLEM - RESULTS REPORT\n")
        f.write(" "*20 + "Purdue University Spring 2012 Dataset\n")
        f.write("="*80 + "\n\n")
        
        f.write("PROBLEM DESCRIPTION:\n")
        f.write("-"*80 + "\n")
        f.write("Task: Schedule exams to timeslots minimizing conflicts and penalties\n")
        f.write("Chromosome: Integer array [exam_id] -> timeslot_id\n")
        f.write("Fitness: Minimization (lower is better)\n\n")
        
        f.write("ALGORITHM PARAMETERS (Fixed by Assignment):\n")
        f.write("-"*80 + "\n")
        f.write("  Population Size (μ): 30\n")
        f.write("  Offspring Size (λ): 10\n")
        f.write("  Generations: 50\n")
        f.write("  Mutation Rate: 0.5\n")
        f.write("  Crossover Rate: 0.8\n")
        f.write("  Number of Runs (K): 10\n\n")
        
        f.write("SELECTION SCHEMES TESTED:\n")
        f.write("-"*80 + "\n")
        for i, scheme in enumerate(results_dict.keys(), 1):
            f.write(f"  {i}. {scheme}\n")
        f.write("\n")
        
        f.write("RESULTS SUMMARY:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Scheme':<40} {'Mean Fitness':<15} {'Std Dev':<15} {'Best':<15}\n")
        f.write("-"*80 + "\n")
        
        sorted_schemes = sorted(results_dict.keys(), 
                               key=lambda x: results_dict[x]['mean_final_fitness'])
        
        for scheme in sorted_schemes:
            results = results_dict[scheme]
            f.write(f"{scheme:<40} "
                   f"{results['mean_final_fitness']:<15.2f} "
                   f"{results['std_final_fitness']:<15.2f} "
                   f"{results['best_final_fitness']:<15.2f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED RESULTS BY SCHEME:\n")
        f.write("="*80 + "\n\n")
        
        for scheme in results_dict.keys():
            results = results_dict[scheme]
            f.write(f"\n{scheme}\n")
            f.write("-"*80 + "\n")
            f.write(f"  Mean Final Fitness: {results['mean_final_fitness']:.2f} ± {results['std_final_fitness']:.2f}\n")
            f.write(f"  Best Final Fitness: {results['best_final_fitness']:.2f}\n")
            f.write(f"  Worst Final Fitness: {results['worst_final_fitness']:.2f}\n")
            f.write(f"  Mean Runtime: {results['mean_runtime']:.2f}s\n")
            f.write(f"  Total Runtime: {results['total_runtime']:.2f}s\n")
            f.write(f"\n  Individual Run Results:\n")
            for i, fitness in enumerate(results['all_final_fitness'], 1):
                f.write(f"    Run {i:2d}: {fitness:.2f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Detailed results saved to: {filename}")


def main():
    """
    Main function to run Q2: Exam Scheduling Problem
    
    This implements the assignment requirements:
    - Load Purdue University Spring 2012 dataset
    - Test 3 selection scheme combinations
    - Run each combination K=10 times
    - Generate comparison plots and statistical analysis
    """
    print("="*80)
    print(" "*20 + "Q2: EXAM SCHEDULING PROBLEM")
    print(" "*15 + "Purdue University Spring 2012 Dataset")
    print("="*80)
    
    # 1. Load Real Data from XML
    xml_path = os.path.join(os.path.dirname(__file__), 'pu-exam-spr12.xml')
    
    if not os.path.exists(xml_path):
        print(f"\n ERROR: Dataset file not found: {xml_path}")
        print("Please ensure 'pu-exam-spr12.xml' is in the ExamScheduling directory.")
        return
    
    exams, students, periods = parse_purdue_data(xml_path)
    
    print(f"\nDataset Summary:")
    print(f"  Total Exams: {len(exams)}")
    print(f"  Total Students: {len(students)}")
    print(f"  Total Timeslots: {periods}")
    
    # 2. Define the three selection scheme combinations (as per assignment requirements)
    schemes_to_test = [
        ('fitness_proportional', 'generational'),  # FPS + Truncation (Generational with elitism)
        ('tournament', 'generational'),            # Binary Tournament + Truncation
        ('random', 'generational')                 # Random + Generational (baseline)
    ]
    
    # Create readable labels
    scheme_labels = [
        'FPS + Generational',
        'Tournament + Generational',
        'Random + Generational'
    ]
    
    print("\n" + "="*80)
    print("TESTING SELECTION SCHEMES")
    print("="*80)
    print("\nThe following combinations will be tested (K=10 runs each):")
    for i, label in enumerate(scheme_labels, 1):
        print(f"  {i}. {label}")
    
    # 3. Run experiments
    results = {}
    
    for (p_scheme, s_scheme), label in zip(schemes_to_test, scheme_labels):
        results[label] = run_k_runs(p_scheme, s_scheme, exams, students, periods, k=10)
    
    # 4. Generate comprehensive analysis
    print("\n" + "="*80)
    print("GENERATING ANALYSIS AND REPORTS")
    print("="*80)
    
    # Print detailed analysis to console
    print_detailed_analysis(results)
    
    # Save results to file
    save_results_to_file(results, 'exam_scheduling_results.txt')
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_comparison(results, 'exam_scheduling_comparison.png')
    
    print("\n" + "="*80)
    print("✅ Q2 COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print("  📊 exam_scheduling_comparison.png - Visual comparison of selection schemes")
    print("  📄 exam_scheduling_results.txt - Detailed statistical results")
    print("\nKey Findings:")
    
    # Quick summary
    best_scheme = min(results.keys(), key=lambda x: results[x]['mean_final_fitness'])
    print(f"  • Best Scheme: {best_scheme}")
    print(f"  • Best Mean Fitness: {results[best_scheme]['mean_final_fitness']:.2f}")
    print(f"  • Most Reliable: {min(results.keys(), key=lambda x: results[x]['std_final_fitness'])}")
    

if __name__ == "__main__":
    main()
