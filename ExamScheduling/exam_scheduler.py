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
    Four combinations tested with K=10 runs each:
    1. Fitness Proportional Selection (FPS) + Truncation
    2. Binary Tournament Selection + Truncation
    3. Truncation + Truncation
    4. Random + Generational

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
    Execute the EA K times as required by Assignment to get statistical results.
    Collects both BSF (Best-So-Far) and ASF (Average-So-Far / Average Fitness)
    per generation for each run.
    
    Args:
        parent_scheme: Parent selection method ('fitness_proportional', 'tournament', 'random')
        survivor_scheme: Survivor selection method ('generational', 'random')
        exams: List of exam IDs
        students: Student-exam enrollment dict
        periods: Number of timeslots
        k: Number of runs (default 10 as per assignment)
        
    Returns:
        Dictionary with BSF and ASF histories per run, plus averages
    """
    all_runs_bsf = []  # BSF history per run
    all_runs_asf = []  # ASF (avg fitness) history per run
    all_final_fitness = []
    all_runtimes = []
    
    print(f"\n{'='*70}")
    print(f"Running: Parent={parent_scheme} + Survivor={survivor_scheme}")
    print(f"{'='*70}")
    
    for run in range(k):
        print(f"  Run {run+1}/{k}...", flush=True)
        
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
            crossover_rate=0.8,
            elitism_count=2,
            tournament_size=2,           
            parent_selection=parent_scheme,
            survivor_selection=survivor_scheme
        )
        
        # Run evolution with progress dots
        for gen in range(solver.generations):
            # Manual single generation step
            if gen == 0:
                solver.population = solver.initialize_population()
            
            fitnesses = [solver.calculate_fitness(ind) for ind in solver.population]
            
            best_idx = np.argmin(fitnesses)
            solver.best_fitness_history.append(fitnesses[best_idx])
            solver.avg_fitness_history.append(np.mean(fitnesses))
            
            if fitnesses[best_idx] < solver.best_fitness:
                solver.best_fitness = fitnesses[best_idx]
                solver.best_solution = solver.population[best_idx].copy()
            
            # Generate offspring
            offspring = []
            while len(offspring) < solver.offspring_size:
                parent1 = solver.select_parent(solver.population, fitnesses)
                parent2 = solver.select_parent(solver.population, fitnesses)
                
                if np.random.random() < solver.crossover_rate:
                    o1, o2 = solver.crossover(parent1, parent2)
                else:
                    o1, o2 = parent1.copy(), parent2.copy()
                
                if np.random.random() < solver.mutation_rate:
                    o1 = solver.mutate(o1)
                if np.random.random() < solver.mutation_rate:
                    o2 = solver.mutate(o2)
                
                offspring.append(o1)
                if len(offspring) < solver.offspring_size:
                    offspring.append(o2)
            
            offspring_fitnesses = [solver.calculate_fitness(ind) for ind in offspring]
            
            solver.population = solver.perform_survivor_selection(
                solver.population, offspring, fitnesses, offspring_fitnesses
            )
            
            # Progress dot every 10 generations
            if gen % 10 == 9:
                print(".", end="", flush=True)
        
        runtime = time.time() - start_time
        
        # Store BSF and ASF histories from this run
        all_runs_bsf.append(solver.best_fitness_history)
        all_runs_asf.append(solver.avg_fitness_history)
        all_final_fitness.append(solver.best_fitness)
        all_runtimes.append(runtime)
        
        print(f" BSF={solver.best_fitness:.0f}, {runtime:.1f}s")
    
    # Calculate averages across K runs
    avg_bsf_history = np.mean(all_runs_bsf, axis=0)
    std_bsf_history = np.std(all_runs_bsf, axis=0)
    avg_asf_history = np.mean(all_runs_asf, axis=0)
    std_asf_history = np.std(all_runs_asf, axis=0)
    
    results = {
        'all_bsf_histories': all_runs_bsf,
        'all_asf_histories': all_runs_asf,
        'avg_bsf_history': avg_bsf_history,
        'std_bsf_history': std_bsf_history,
        'avg_asf_history': avg_asf_history,
        'std_asf_history': std_asf_history,
        'all_final_fitness': all_final_fitness,
        'mean_final_fitness': np.mean(all_final_fitness),
        'std_final_fitness': np.std(all_final_fitness),
        'best_final_fitness': np.min(all_final_fitness),
        'worst_final_fitness': np.max(all_final_fitness),
        'mean_runtime': np.mean(all_runtimes),
        'total_runtime': np.sum(all_runtimes),
        'k': k
    }
    
    print(f"\n  Summary Statistics:")
    print(f"    Mean Final BSF: {results['mean_final_fitness']:.2f} +/- {results['std_final_fitness']:.2f}")
    print(f"    Best Final BSF: {results['best_final_fitness']:.2f}")
    print(f"    Worst Final BSF: {results['worst_final_fitness']:.2f}")
    print(f"    Mean Runtime: {results['mean_runtime']:.2f}s")
    
    return results


def print_generation_table(label, results):
    """
    Print the generation-by-generation table for a single combination.
    Columns: Generation | Run#1 BSF | Run#1 ASF | ... | Run#K BSF | Run#K ASF | Avg BSF | Avg ASF
    
    Args:
        label: Name of the scheme combination
        results: Results dict from run_k_runs
    """
    k = results['k']
    n_gens = len(results['avg_bsf_history'])
    
    # Column widths
    gen_w = 5
    val_w = 10
    
    # Calculate total width
    total_cols = 1 + (k * 2) + 2  # Gen + (K runs × 2) + Avg BSF + Avg ASF
    total_w = gen_w + (total_cols - 1) * val_w + (total_cols + 1)  # +1 for each | separator
    
    print(f"\n┌{'─' * (total_w - 2)}┐")
    title = f"TABLE: {label}"
    print(f"│{title:^{total_w - 2}}│")
    
    # Build header row
    print(f"├{'─' * gen_w}┬", end="")
    for r in range(1, k + 1):
        print(f"{'─' * val_w}┬{'─' * val_w}┬", end="")
    print(f"{'─' * val_w}┬{'─' * val_w}┤")
    
    # Header labels
    header = f"│{'Gen':^{gen_w}}│"
    for r in range(1, k + 1):
        header += f"{'R'+str(r)+' BSF':^{val_w}}│{'R'+str(r)+' ASF':^{val_w}}│"
    header += f"{'Avg BSF':^{val_w}}│{'Avg ASF':^{val_w}}│"
    print(header)
    
    # Separator after header
    print(f"├{'─' * gen_w}┼", end="")
    for r in range(1, k + 1):
        print(f"{'─' * val_w}┼{'─' * val_w}┼", end="")
    print(f"{'─' * val_w}┼{'─' * val_w}┤")
    
    # Print each generation row
    for gen in range(n_gens):
        row = f"│{gen:^{gen_w}}│"
        for r in range(k):
            bsf_val = results['all_bsf_histories'][r][gen]
            asf_val = results['all_asf_histories'][r][gen]
            row += f"{bsf_val:>{val_w}.1f}│{asf_val:>{val_w}.1f}│"
        row += f"{results['avg_bsf_history'][gen]:>{val_w}.2f}│{results['avg_asf_history'][gen]:>{val_w}.2f}│"
        print(row)
    
    # Bottom border
    print(f"└{'─' * gen_w}┴", end="")
    for r in range(1, k + 1):
        print(f"{'─' * val_w}┴{'─' * val_w}┴", end="")
    print(f"{'─' * val_w}┴{'─' * val_w}┘")


def save_generation_table_csv(label, results, filename):
    """
    Save the generation-by-generation table as a CSV file.
    
    Args:
        label: Name of the scheme combination
        results: Results dict from run_k_runs
        filename: Output CSV path
    """
    k = results['k']
    n_gens = len(results['avg_bsf_history'])
    
    # Ensure absolute path in main directory
    filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', filename))
    
    with open(filepath, 'w') as f:
        # Header
        cols = ['Generation']
        for r in range(1, k + 1):
            cols.append(f'Run{r}_BSF')
            cols.append(f'Run{r}_ASF')
        cols.append('Average_BSF')
        cols.append('Average_ASF')
        f.write(','.join(cols) + '\n')
        
        # Data rows
        for gen in range(n_gens):
            vals = [str(gen)]
            for r in range(k):
                vals.append(f"{results['all_bsf_histories'][r][gen]:.2f}")
                vals.append(f"{results['all_asf_histories'][r][gen]:.2f}")
            vals.append(f"{results['avg_bsf_history'][gen]:.2f}")
            vals.append(f"{results['avg_asf_history'][gen]:.2f}")
            f.write(','.join(vals) + '\n')
    
    print(f"  Table saved to: {filepath}")


def save_generation_table_image(label, results, filename):
    """
    Save the generation-by-generation table as a figure image.
    
    Args:
        label: Name of the scheme combination
        results: Results dict from run_k_runs
        filename: Output image path (.png)
    """
    k = results['k']
    n_gens = len(results['avg_bsf_history'])
    
    # Build column headers
    col_labels = ['Gen']
    for r in range(1, k + 1):
        col_labels.append(f'R{r} BSF')
        col_labels.append(f'R{r} ASF')
    col_labels.append('Avg BSF')
    col_labels.append('Avg ASF')
    
    # Build table data
    table_data = []
    for gen in range(n_gens):
        row = [str(gen)]
        for r in range(k):
            row.append(f"{results['all_bsf_histories'][r][gen]/1e6:.2f}M")
            row.append(f"{results['all_asf_histories'][r][gen]/1e6:.2f}M")
        row.append(f"{results['avg_bsf_history'][gen]/1e6:.2f}M")
        row.append(f"{results['avg_asf_history'][gen]/1e6:.2f}M")
        table_data.append(row)
    
    # Create figure
    n_cols = len(col_labels)
    fig_width = min(24, 1.2 * n_cols)  # Cap width
    fig_height = max(8, 0.3 * n_gens)  # Scale with generations
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    ax.set_title(f'Generation Table: {label}\n(K={k} runs, {n_gens} generations)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.2, 1.5)
    
    # Color header row
    for j in range(n_cols):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors for readability
    for i in range(1, n_gens + 1):
        for j in range(n_cols):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D6DCE5')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    # Highlight Avg columns
    for i in range(1, n_gens + 1):
        table[(i, n_cols - 2)].set_facecolor('#E2EFDA')  # Avg BSF - light green
        table[(i, n_cols - 1)].set_facecolor('#FCE4D6')  # Avg ASF - light orange
    
    plt.tight_layout()
    
    # Save
    filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', filename))
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Table image saved to: {filepath}")


def plot_single_combination(label, results, save_path=None):
    """
    Plot Avg BSF and Avg ASF side by side for a single combination.
    
    Args:
        label: Name of the scheme combination
        results: Results dict from run_k_runs
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    generations = range(len(results['avg_bsf_history']))
    
    # Left plot: Average BSF
    ax1.plot(generations, results['avg_bsf_history'], 'b-', linewidth=2)
    ax1.set_xlabel('Generation', fontsize=11)
    ax1.set_ylabel('Average Best-So-Far (BSF)', fontsize=11)
    ax1.set_title('Avg. BSF vs Generation', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Average ASF
    ax2.plot(generations, results['avg_asf_history'], 'r-', linewidth=2)
    ax2.set_xlabel('Generation', fontsize=11)
    ax2.set_ylabel('Average Average-Fitness (ASF)', fontsize=11)
    ax2.set_title('Avg. ASF vs Generation', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'{label}\n(K={results["k"]} runs, 50 generations)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        # Ensure absolute path in main directory
        filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', save_path))
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Plot saved to: {filepath}")
    
    plt.close()  # Close figure to free memory and avoid blocking


def plot_all_combinations(results_dict, save_path='exam_scheduling_comparison.png'):
    """
    Plot Avg BSF and Avg ASF for ALL combinations side by side for comparison.
    Left subplot: Avg BSF curves for all schemes overlaid
    Right subplot: Avg ASF curves for all schemes overlaid
    
    Args:
        results_dict: Dictionary with results from different schemes
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    schemes = list(results_dict.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(schemes)))
    
    # Left: Avg BSF for all combinations
    for i, (scheme, results) in enumerate(results_dict.items()):
        generations = range(len(results['avg_bsf_history']))
        ax1.plot(generations, results['avg_bsf_history'], label=scheme,
                 color=colors[i], linewidth=2)
    
    ax1.set_xlabel('Generation', fontsize=11)
    ax1.set_ylabel('Average Best-So-Far (BSF)', fontsize=11)
    ax1.set_title('Avg. BSF vs Generation (All Schemes)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Right: Avg ASF for all combinations
    for i, (scheme, results) in enumerate(results_dict.items()):
        generations = range(len(results['avg_asf_history']))
        ax2.plot(generations, results['avg_asf_history'], label=scheme,
                 color=colors[i], linewidth=2)
    
    ax2.set_xlabel('Generation', fontsize=11)
    ax2.set_ylabel('Average Average-Fitness (ASF)', fontsize=11)
    ax2.set_title('Avg. ASF vs Generation (All Schemes)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Exam Scheduling: Selection Schemes Comparison (Purdue Spr12)\n'
                 'Parameters: μ=30, λ=10, Generations=50, K=10 runs',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Ensure absolute path in main directory
    filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', save_path))
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {filepath}")
    
    plt.close()  # Close figure to free memory and avoid blocking


def print_summary(results_dict):
    """
    Print a concise summary of all selection schemes.
    
    Args:
        results_dict: Dictionary with results from different schemes
    """
    print("\n" + "="*80)
    print(" "*20 + "RESULTS SUMMARY")
    print("="*80)
    
    schemes = list(results_dict.keys())
    
    # Find best scheme
    best_mean_fitness = min(results_dict[s]['mean_final_fitness'] for s in schemes)
    best_scheme = [s for s in schemes if results_dict[s]['mean_final_fitness'] == best_mean_fitness][0]
    
    print(f"\nBest Scheme (by mean BSF): {best_scheme}")
    print(f"  Mean Final BSF: {results_dict[best_scheme]['mean_final_fitness']:.2f}")
    print(f"  Std Dev:        {results_dict[best_scheme]['std_final_fitness']:.2f}")
    
    print("\n" + "-"*80)
    print(f"{'#':<4} {'Selection Scheme':<35} {'Mean BSF':<12} {'Std Dev':<12} {'Best BSF':<12} {'Runtime(s)':<12}")
    print("-"*80)
    
    sorted_schemes = sorted(schemes, key=lambda x: results_dict[x]['mean_final_fitness'])
    
    for i, scheme in enumerate(sorted_schemes):
        r = results_dict[scheme]
        marker = " <-- best" if scheme == best_scheme else ""
        print(f"{i+1:<4} {scheme:<35} {r['mean_final_fitness']:<12.2f} "
              f"{r['std_final_fitness']:<12.2f} {r['best_final_fitness']:<12.2f} "
              f"{r['mean_runtime']:<12.2f}{marker}")
    
    print("-"*80)


def save_results_to_file(results_dict, filename='exam_scheduling_results.txt'):
    """
    Save detailed results to a text file
    
    Args:
        results_dict: Dictionary with results from different schemes
        filename: Output filename
    """
    # Ensure absolute path in main directory
    filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', filename))
    
    with open(filepath, 'w') as f:
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
            k = results['k']
            f.write(f"\n{scheme}\n")
            f.write("-"*80 + "\n")
            f.write(f"  Mean Final BSF: {results['mean_final_fitness']:.2f} +/- {results['std_final_fitness']:.2f}\n")
            f.write(f"  Best Final BSF: {results['best_final_fitness']:.2f}\n")
            f.write(f"  Worst Final BSF: {results['worst_final_fitness']:.2f}\n")
            f.write(f"  Mean Runtime: {results['mean_runtime']:.2f}s\n")
            f.write(f"  Total Runtime: {results['total_runtime']:.2f}s\n")
            f.write(f"\n  Individual Run Results:\n")
            for i, fitness in enumerate(results['all_final_fitness'], 1):
                f.write(f"    Run {i:2d}: BSF = {fitness:.2f}\n")
            
            # Generation-by-generation table
            n_gens = len(results['avg_bsf_history'])
            f.write(f"\n  Generation Table (BSF and ASF per run):\n")
            header = f"    {'Gen':<6}"
            for r in range(1, k + 1):
                header += f" {'R'+str(r)+' BSF':>10} {'R'+str(r)+' ASF':>10}"
            header += f" {'Avg BSF':>10} {'Avg ASF':>10}"
            f.write(header + "\n")
            f.write("    " + "-" * (len(header) - 4) + "\n")
            for gen in range(n_gens):
                row = f"    {gen:<6}"
                for r in range(k):
                    row += f" {results['all_bsf_histories'][r][gen]:>10.2f}"
                    row += f" {results['all_asf_histories'][r][gen]:>10.2f}"
                row += f" {results['avg_bsf_history'][gen]:>10.2f}"
                row += f" {results['avg_asf_history'][gen]:>10.2f}"
                f.write(row + "\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Detailed results saved to: {filepath}")


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
    
    # Show output directory
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print(f"\nOutput directory: {output_dir}")
    
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
    
    # 2. Define the four selection scheme combinations (as per assignment requirements)
    # Format: (Parent Selection, Survivor Selection)
    # Truncation for survivors = mu_plus_lambda (keep best μ from parents+offspring)
    schemes_to_test = [
        ('fitness_proportional', 'mu_plus_lambda'),  # FPS + Truncation
        ('tournament', 'mu_plus_lambda'),            # Binary Tournament + Truncation
        ('truncation', 'mu_plus_lambda'),            # Truncation + Truncation
        ('random', 'generational')                   # Random + Generational
    ]
    
    # Create readable labels
    scheme_labels = [
        'FPS + Truncation',
        'Tournament + Truncation',
        'Truncation + Truncation',
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
    
    # 4. For each combination: print table + side-by-side BSF/ASF plot
    print("\n" + "="*80)
    print("GENERATION-BY-GENERATION TABLES & PLOTS")
    print("="*80)
    
    for label in scheme_labels:
        # (i) Print the table: Gen | R1 BSF | R1 ASF | ... | Avg BSF | Avg ASF
        print_generation_table(label, results[label])
        
        # Save table as CSV
        csv_name = label.replace(' ', '_').replace('+', '_') + '_table.csv'
        save_generation_table_csv(label, results[label], csv_name)
        
        # Save table as image
        table_img_name = label.replace(' ', '_').replace('+', '_') + '_table.png'
        save_generation_table_image(label, results[label], table_img_name)
        
        # (ii) Side-by-side plot of Avg BSF and Avg ASF for this combination
        plot_name = label.replace(' ', '_').replace('+', '_') + '_plot.png'
        plot_single_combination(label, results[label], save_path=plot_name)
    
    # 5. Comparison plot: all combinations' Avg BSF and Avg ASF side by side
    print("\nGenerating combined comparison plot...")
    plot_all_combinations(results, 'exam_scheduling_comparison.png')
    
    print("\n" + "="*80)
    print("Q2 COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    for label in scheme_labels:
        tag = label.replace(' ', '_').replace('+', '_')
        print(f"  - {tag}_table.csv (data)")
        print(f"  - {tag}_table.png (table image)")
        print(f"  - {tag}_plot.png (Avg BSF & ASF plot)")
    print(f"  - exam_scheduling_comparison.png (all schemes side by side)")
    

if __name__ == "__main__":
    main()
