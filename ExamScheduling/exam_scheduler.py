"""
Exam Scheduling Problem Solver using Evolutionary Algorithm
Based on Purdue University benchmark dataset
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))

from ea_framework import EAFramework
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json


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
        
        # Build student exam matrix for soft constraint evaluation
        self.student_exam_matrix = self._build_student_exam_matrix()
        
    def _build_conflict_matrix(self):
        """Build matrix of exam conflicts (same student enrolled)"""
        conflict_matrix = np.zeros((self.n_exams, self.n_exams), dtype=int)
        
        for student_exams in self.students.values():
            for i, exam1 in enumerate(student_exams):
                for exam2 in student_exams[i+1:]:
                    if exam1 < self.n_exams and exam2 < self.n_exams:
                        conflict_matrix[exam1][exam2] = 1
                        conflict_matrix[exam2][exam1] = 1
        
        return conflict_matrix
    
    def _build_student_exam_matrix(self):
        """Build matrix of which students take which exams"""
        matrix = []
        for student_exams in self.students.values():
            student_vector = [0] * self.n_exams
            for exam in student_exams:
                if exam < self.n_exams:
                    student_vector[exam] = 1
            matrix.append(student_vector)
        return np.array(matrix)
    
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
        
        Args:
            schedule: Exam schedule array
            
        Returns:
            Repaired schedule
        """
        schedule = schedule.copy()
        max_repairs = 100
        repairs = 0
        
        while repairs < max_repairs:
            conflicts_found = False
            
            for i in range(self.n_exams):
                for j in range(i+1, self.n_exams):
                    # If exams conflict and are in same timeslot
                    if self.conflict_matrix[i][j] == 1 and schedule[i] == schedule[j]:
                        # Move exam j to a different timeslot
                        available_slots = [s for s in range(self.n_timeslots) 
                                         if s != schedule[i]]
                        if available_slots:
                            schedule[j] = np.random.choice(available_slots)
                            conflicts_found = True
            
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
        """Calculate hard constraint violations (conflicts)"""
        violations = 0
        
        for i in range(self.n_exams):
            for j in range(i+1, self.n_exams):
                # If exams conflict and scheduled at same time
                if self.conflict_matrix[i][j] == 1 and schedule[i] == schedule[j]:
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
        
        Args:
            schedule: Schedule to mutate
            
        Returns:
            Mutated schedule
        """
        schedule = schedule.copy()
        
        # Number of mutations
        n_mutations = np.random.randint(1, max(2, self.n_exams // 20))
        
        for _ in range(n_mutations):
            # Select random exam
            exam_idx = np.random.randint(0, self.n_exams)
            
            # Assign new timeslot
            schedule[exam_idx] = np.random.randint(0, self.n_timeslots)
        
        # Repair conflicts
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


def generate_sample_data(n_exams=100, n_students=500, n_timeslots=20, avg_exams_per_student=5):
    """
    Generate sample exam scheduling data
    
    Args:
        n_exams: Number of exams
        n_students: Number of students
        n_timeslots: Number of available timeslots
        avg_exams_per_student: Average exams per student
        
    Returns:
        exams, students, timeslots, constraints
    """
    print(f"Generating sample dataset...")
    print(f"  Exams: {n_exams}")
    print(f"  Students: {n_students}")
    print(f"  Timeslots: {n_timeslots}")
    print(f"  Avg exams/student: {avg_exams_per_student}")
    
    exams = list(range(n_exams))
    students = {}
    
    # Generate student enrollments
    for student_id in range(n_students):
        # Random number of exams per student (around average)
        n_student_exams = max(2, int(np.random.normal(avg_exams_per_student, 2)))
        n_student_exams = min(n_student_exams, n_exams)
        
        # Random exam selection
        student_exams = list(np.random.choice(n_exams, n_student_exams, replace=False))
        students[student_id] = student_exams
    
    constraints = {
        'hard': {
            'no_conflicts': True,
        },
        'soft': {
            'spread_exams': True,
            'balance_load': True,
            'avoid_consecutive': True,
        }
    }
    
    return exams, students, n_timeslots, constraints


def main():
    """Main function to run exam scheduling solver"""
    print("="*60)
    print("Exam Scheduling Solver using Evolutionary Algorithm")
    print("="*60)
    
    # Generate sample data (simulating Purdue Spring 2012 characteristics)
    exams, students, timeslots, constraints = generate_sample_data(
        n_exams=100,
        n_students=400,
        n_timeslots=15,
        avg_exams_per_student=5
    )
    
    # Create solver
    solver = ExamSchedulingSolver(
        exams=exams,
        students=students,
        timeslots=timeslots,
        constraints=constraints,
        population_size=150,
        generations=500,
        crossover_rate=0.8,
        mutation_rate=0.2,
        elitism_count=5,
        tournament_size=5
    )
    
    print("\nEvolutionary Algorithm Parameters:")
    print(f"  Population Size: {solver.population_size}")
    print(f"  Generations: {solver.generations}")
    print(f"  Crossover Rate: {solver.crossover_rate}")
    print(f"  Mutation Rate: {solver.mutation_rate}")
    print(f"  Elitism Count: {solver.elitism_count}")
    print(f"  Tournament Size: {solver.tournament_size}")
    
    print("\nChromosome Design:")
    print("  Representation: Array where index=exam_id, value=timeslot")
    print("  Example: [2, 5, 1, 5, 0, ...] means exam0→slot2, exam1→slot5, etc.")
    
    print("\nFitness Function:")
    print("  Penalty-based (lower is better)")
    print("  Hard constraints (×10000 penalty): Exam conflicts")
    print("  Soft constraints: Spread, balance, consecutive exams")
    
    print("\nEvolutionary Operators:")
    print("  Selection: Tournament selection")
    print("  Crossover: Uniform crossover with conflict repair")
    print("  Mutation: Random timeslot reassignment with repair")
    print("  Elitism: Best schedules preserved")
    
    print("\nStarting evolution...\n")
    
    # Run evolution
    best_schedule = solver.evolve(verbose=True)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    stats = solver.get_statistics()
    print(f"Best Fitness (Penalty): {stats['best_fitness']:.2f}")
    print(f"Initial Best Penalty: {solver.best_fitness_history[0]:.2f}")
    print(f"Improvement: {stats['improvement']:.2f} ({stats['improvement_percent']:.2f}%)")
    
    # Analyze best schedule
    solver.analyze_schedule()
    
    # Plot convergence
    print("\nGenerating convergence plot...")
    solver.plot_convergence(
        title="Exam Scheduling Convergence",
        save_path="ExamScheduling_convergence.png"
    )
    
    # Plot distribution
    print("\nGenerating distribution plot...")
    solver.plot_schedule_distribution(
        save_path="ExamScheduling_distribution.png"
    )
    
    # Save results
    with open("ExamScheduling_results.txt", "w") as f:
        f.write("EXAM SCHEDULING SOLVER RESULTS\n")
        f.write("="*60 + "\n\n")
        
        f.write("Problem Instance:\n")
        f.write(f"  Number of Exams: {solver.n_exams}\n")
        f.write(f"  Number of Students: {len(solver.students)}\n")
        f.write(f"  Number of Timeslots: {solver.n_timeslots}\n\n")
        
        f.write("Algorithm Parameters:\n")
        f.write(f"  Population Size: {solver.population_size}\n")
        f.write(f"  Generations: {solver.generations}\n")
        f.write(f"  Crossover Rate: {solver.crossover_rate}\n")
        f.write(f"  Mutation Rate: {solver.mutation_rate}\n")
        f.write(f"  Elitism Count: {solver.elitism_count}\n")
        f.write(f"  Tournament Size: {solver.tournament_size}\n\n")
        
        f.write("Chromosome Design:\n")
        f.write("  Representation: Integer array\n")
        f.write("  Length: Number of exams\n")
        f.write("  Gene values: Timeslot assignment (0 to n_timeslots-1)\n\n")
        
        f.write("Fitness Function:\n")
        f.write("  Type: Penalty-based (minimization)\n")
        f.write("  Hard constraints: Exam conflicts (×10000 penalty)\n")
        f.write("  Soft constraints: Spread (×100), Balance (×50), Consecutive (×200)\n\n")
        
        f.write("Results:\n")
        f.write(f"  Best Fitness: {stats['best_fitness']:.2f}\n")
        f.write(f"  Initial Best Fitness: {solver.best_fitness_history[0]:.2f}\n")
        f.write(f"  Improvement: {stats['improvement']:.2f} ({stats['improvement_percent']:.2f}%)\n\n")
        
        f.write("Hard Constraint Violations:\n")
        hard_violations = solver._calculate_hard_constraints(best_schedule)
        f.write(f"  Exam Conflicts: {hard_violations}\n")
        if hard_violations == 0:
            f.write("  Status: ✓ All hard constraints satisfied\n\n")
        else:
            f.write("  Status: ✗ Has conflicts\n\n")
        
        f.write("Best Schedule:\n")
        f.write(str(best_schedule.tolist()))
    
    print("\nResults saved to ExamScheduling_results.txt")
    print("\nDone!")


if __name__ == "__main__":
    main()
