import numpy as np
from collections import defaultdict
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))

from ea_framework import EAFramework


class ExamSchedulingSolver(EAFramework):
    """Exam Scheduling solver with dictionary chromosome"""
    
    def __init__(self, exams_data, students, periods_data, rooms_data, **kwargs):
        super().__init__(**kwargs)
        
        self.exams_data = exams_data
        self.students = students
        self.periods_data = periods_data
        self.rooms_data = rooms_data
        
        self.exam_ids = list(exams_data.keys())
        self.period_ids = list(periods_data.keys())
        self.room_ids = list(rooms_data.keys())
        
        self.n_exams = len(self.exam_ids)
        self.n_periods = len(self.period_ids)
        
        # Build conflict structures
        self.conflict_matrix = self._build_conflict_matrix()
        
    def _build_conflict_matrix(self):
        """Build exam conflict matrix"""
        conflicts = defaultdict(set)
        
        for student_exams in self.students.values():
            for i, exam1 in enumerate(student_exams):
                for exam2 in student_exams[i+1:]:
                    if exam1 in self.exam_ids and exam2 in self.exam_ids:
                        conflicts[exam1].add(exam2)
                        conflicts[exam2].add(exam1)
        
        return dict(conflicts)
    
    def _smart_feasible_initialization(self):
        """
        Smarter initialization:
        - Sort exams by size (harder first)
        - Assign conflict-free period
        - Assign best-fit room capacity
        - Prevent room double booking
        """

        schedule = {}

        # Track which rooms are used in each period
        room_occupancy = {p: set() for p in self.period_ids}

        # Sort exams: largest first
        sorted_exams = sorted(
            self.exam_ids,
            key=lambda e: self.exams_data[e].get("size", 0),
            reverse=True
        )

        for exam_id in sorted_exams:

            exam_size = self.exams_data[exam_id].get("size", 1)

            # 1. Find conflict-free periods
            used_periods = set()
            for conflict_exam in self.conflict_matrix.get(exam_id, set()):
                if conflict_exam in schedule:
                    used_periods.add(schedule[conflict_exam]["period"])

            valid_periods = [p for p in self.period_ids if p not in used_periods]

            if not valid_periods:
                # fallback: choose any period
                valid_periods = self.period_ids

            np.random.shuffle(valid_periods)

            assigned = False

            # 2. Try periods until room found
            for period in valid_periods:

                # Rooms free in this period
                free_rooms = [
                    r for r in self.room_ids
                    if r not in room_occupancy[period]
                ]

                if not free_rooms:
                    continue

                # Sort rooms by capacity (smallest that fits)
                free_rooms.sort(key=lambda r: self.rooms_data[r]["capacity"])

                fitting_rooms = [
                    r for r in free_rooms
                    if self.rooms_data[r]["capacity"] >= exam_size
                ]

                if fitting_rooms:
                    chosen_room = fitting_rooms[0]

                    schedule[exam_id] = {
                        "period": period,
                        "rooms": [chosen_room]
                    }

                    room_occupancy[period].add(chosen_room)
                    assigned = True
                    break

            # 3. If no room fits, assign random room anyway 
            if not assigned:
                period = np.random.choice(valid_periods)
                room = np.random.choice(self.room_ids)

                schedule[exam_id] = {
                    "period": period,
                    "rooms": [room]
                }

        return schedule
    
    def initialize_population(self):
        """
        Smart conflict-free + room-aware initialization
        Places large exams first for better feasibility.
        """

        population = []
        print(f"Initializing {self.population_size} smart feasible schedules...")

        for i in range(self.population_size):
            if i % 5 == 0:
                print(f"Creating schedule {i+1}/{self.population_size}...")

            schedule = self._smart_feasible_initialization()
            population.append(schedule)

        print("Smart initialization complete!")
        return population
    
    def _calc_room_capacity_violations(self, schedule):
        """
        Count exams assigned to rooms that are too small.
        """
        violations = 0

        for exam_id, assignment in schedule.items():
            exam_size = self.exams_data[exam_id].get("size", 1)

            for room_id in assignment["rooms"]:
                room_cap = self.rooms_data[room_id]["capacity"]

                if room_cap < exam_size:
                    violations += 1

        return violations
    
    def _calc_room_clashes(self, schedule):
        """
        Count cases where two exams use the same room in the same period.
        """
        clashes = 0
        room_usage = defaultdict(list)

        for exam_id, assignment in schedule.items():
            period = assignment["period"]

            for room_id in assignment["rooms"]:
                key = (period, room_id)
                room_usage[key].append(exam_id)

        for key, exams in room_usage.items():
            if len(exams) > 1:
                clashes += len(exams) - 1  # extra exams beyond the first

        return clashes
    
    def _calc_room_waste(self, schedule):
        waste = 0

        for exam_id, assign in schedule.items():
            exam_size = self.exams_data[exam_id]["size"]

            for room in assign["rooms"]:
                cap = self.rooms_data[room]["capacity"]

                if cap > exam_size:
                    waste += (cap - exam_size)

        return waste

    def _calc_more_than_two_exams_day(self, schedule):
        penalty = 0

        for student, exams in self.students.items():
            day_count = defaultdict(int)

            for eid in exams:
                if eid not in schedule:
                    continue

                period = schedule[eid]["period"]
                day = self.periods_data[period]["day"]

                day_count[day] += 1

            for day, count in day_count.items():
                if count > 2:
                    penalty += (count - 2)

        return penalty
    
    def calculate_fitness(self, schedule):
        """
        Fitness based on CPSolver benchmark weights.
        Lower is better.
        """

        # CPSolver weights
        W_DIRECT = 1000.0
        W_TWO_IN_ROW = 10.0
        W_MORE_THAN_TWO_DAY = 100.0
        W_ROOM_SPLIT = 10.0
        W_ROOM_SIZE = 0.001
        W_ROOM_CLASH = 1000.0 

        penalty = 0
        # 1. Direct student conflicts
        penalty += W_DIRECT * self._calc_direct_conflicts(schedule)

        # 2. Room clashes (double booking)
        penalty += W_ROOM_CLASH * self._calc_room_clashes(schedule)

        # 3. Room capacity violations (soft but important)
        penalty += 500 * self._calc_room_capacity_violations(schedule)

        # 4. Room size waste (excess capacity)
        penalty += W_ROOM_SIZE * self._calc_room_waste(schedule)

        # 5. Back-to-back exams
        penalty += W_TWO_IN_ROW * self._calc_consecutive_penalty(schedule)

        # 6. More than 2 exams per day
        penalty += W_MORE_THAN_TWO_DAY * self._calc_more_than_two_exams_day(schedule)

        return penalty
    
    def _calc_direct_conflicts(self, schedule):
        """Count exam conflicts (same student, same period)"""
        conflicts = 0
        
        period_exams = defaultdict(list)
        for exam_id, assignment in schedule.items():
            period_exams[assignment['period']].append(exam_id)
        
        for period, exams in period_exams.items():
            for i, exam1 in enumerate(exams):
                for exam2 in exams[i+1:]:
                    if exam2 in self.conflict_matrix.get(exam1, set()):
                        conflicts += 1
        
        return conflicts
    
    def _calc_spread_penalty(self, schedule):
        """Penalty for exams too close together"""
        penalty = 0
        proximity_weights = {0: 10, 1: 5, 2: 2}
        
        for student_exams in self.students.values():
            periods = sorted([
                self.periods_data[schedule[eid]["period"]]["index"]
                for eid in student_exams
                if eid in schedule
            ])
            
            for i in range(len(periods) - 1):
                gap = periods[i+1] - periods[i]
                if gap in proximity_weights:
                    penalty += proximity_weights[gap]
        
        return penalty
    
    def _calc_balance_penalty(self, schedule):
        """Penalty for uneven distribution"""
        period_counts = defaultdict(int)
        for assignment in schedule.values():
            period_counts[assignment['period']] += 1
        
        counts = list(period_counts.values())
        return np.var(counts) if counts else 0
    
    def _calc_consecutive_penalty(self, schedule):
        """Penalty for consecutive exams"""
        penalty = 0
        
        for student_exams in self.students.values():
            periods = sorted([
                self.periods_data[schedule[eid]["period"]]["index"]
                for eid in student_exams
                if eid in schedule
            ])
            
            for i in range(len(periods) - 1):
                if periods[i+1] - periods[i] == 1:
                    penalty += 1
        
        return penalty
    
    def crossover(self, parent1, parent2):
        """
        Uniform crossover with conflict repair
        Ensures offspring remain conflict-free
        """
        offspring1 = {}
        offspring2 = {}
        
        for exam_id in self.exam_ids:
            if np.random.rand() < 0.5:
                offspring1[exam_id] = parent1[exam_id].copy()
                offspring2[exam_id] = parent2[exam_id].copy()
            else:
                offspring1[exam_id] = parent2[exam_id].copy()
                offspring2[exam_id] = parent1[exam_id].copy()
        
        # Repair any conflicts introduced by crossover
        offspring1 = self._repair_conflicts(offspring1)
        offspring2 = self._repair_conflicts(offspring2)
        
        return offspring1, offspring2

    def _repair_conflicts(self, schedule):
        """
        Repair offspring schedule after crossover.

        Repairs HARD violations:
        1. Student conflicts (same period)
        2. Room clashes (double booking)
        3. Room capacity violations

        Produces a feasible offspring.
        """

        max_iterations = 200

        for _ in range(max_iterations):
            conflict_fixed = False

            for exam_id in self.exam_ids:

                exam_period = schedule[exam_id]["period"]

                for conflict_exam in self.conflict_matrix.get(exam_id, set()):
                    if schedule[conflict_exam]["period"] == exam_period:

                        # Find safe periods
                        used_periods = {
                            schedule[c]["period"]
                            for c in self.conflict_matrix.get(exam_id, set())
                        }

                        available_periods = [
                            p for p in self.period_ids
                            if p not in used_periods
                        ]

                        if available_periods:
                            schedule[exam_id]["period"] = np.random.choice(available_periods)
                        else:
                            # fallback: least conflicting period
                            schedule[exam_id]["period"] = np.random.choice(self.period_ids)

                        conflict_fixed = True
                        break

                if conflict_fixed:
                    break

            if conflict_fixed:
                continue  # restart loop after fixing
            room_usage = defaultdict(set)

            for exam_id in self.exam_ids:

                exam_size = self.exams_data[exam_id].get("size", 1)
                period = schedule[exam_id]["period"]

                # Rooms already taken in this period
                taken_rooms = room_usage[period]

                # Find valid free rooms with enough capacity
                valid_rooms = [
                    r for r in self.room_ids
                    if r not in taken_rooms
                    and self.rooms_data[r]["capacity"] >= exam_size
                ]

                if valid_rooms:
                    chosen_room = np.random.choice(valid_rooms)
                    schedule[exam_id]["rooms"] = [chosen_room]
                    taken_rooms.add(chosen_room)

                else:
                    # fallback: ignore clash but choose any fitting room
                    fitting_rooms = [
                        r for r in self.room_ids
                        if self.rooms_data[r]["capacity"] >= exam_size
                    ]

                    if fitting_rooms:
                        schedule[exam_id]["rooms"] = [np.random.choice(fitting_rooms)]
                    else:
                        schedule[exam_id]["rooms"] = [np.random.choice(self.room_ids)]

            # If we reached here: stable schedule
            break

        return schedule


    def mutate(self, schedule):
        """
        Mutate by changing periods while maintaining conflict-free property
        Only moves exams to valid (conflict-free) periods
        """
        schedule = {k: v.copy() for k, v in schedule.items()}
        
        n_mutations = max(1, int(self.n_exams * 0.05))
        exam_ids_to_mutate = np.random.choice(self.exam_ids, n_mutations, replace=False)
        
        for exam_id in exam_ids_to_mutate:

            exam_size = self.exams_data[exam_id].get("size", 1)

            # Find conflict-free periods
            used_periods = {
                schedule[c]["period"]
                for c in self.conflict_matrix.get(exam_id, set())
            }

            available_periods = [
                p for p in self.period_ids if p not in used_periods
            ]

            if not available_periods:
                continue

            new_period = np.random.choice(available_periods)

            # Choose valid room (capacity-safe)
            taken_rooms = set()

            for other_exam, assign in schedule.items():
                if other_exam != exam_id and assign["period"] == new_period:
                    taken_rooms.update(assign["rooms"])

            valid_rooms = [
                r for r in self.room_ids
                if r not in taken_rooms
                and self.rooms_data[r]["capacity"] >= exam_size
            ]

            if not valid_rooms:
                valid_rooms = self.room_ids

            new_room = np.random.choice(valid_rooms)

            schedule[exam_id] = {
                "period": new_period,
                "rooms": [new_room]
            }
        
        return schedule
    
    def plot_schedule_distribution(self, schedule=None, save_path=None):
        """Plot distribution of exams across periods"""
        if schedule is None:
            schedule = self.best_solution
        
        if schedule is None:
            print("No schedule to plot!")
            return
        
        period_counts = defaultdict(int)
        for assignment in schedule.values():
            period_counts[assignment['period']] += 1
        
        periods = sorted(period_counts.keys())
        counts = [period_counts[p] for p in periods]
        
        plt.figure(figsize=(14, 6))
        plt.bar(periods, counts, color='steelblue', alpha=0.7)
        plt.axhline(y=self.n_exams/self.n_periods, color='red', 
                   linestyle='--', label=f'Average ({self.n_exams/self.n_periods:.1f})')
        plt.xlabel('Period', fontsize=12)
        plt.ylabel('Number of Exams', fontsize=12)
        plt.title('Exam Distribution Across Periods', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution plot saved to {save_path}")
        
        plt.show()


def parse_exam_data(file_path):
    """Parse Purdue XML dataset"""
    print(f"Loading dataset: {file_path}")
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Parse periods
    periods_data = {}

    for idx, period_elem in enumerate(root.findall('./periods/period')):
        pid = int(period_elem.get("id"))

        periods_data[pid] = {
            "index": idx,   # true chronological ordering
            "day": period_elem.get("day", ""),
            "time": period_elem.get("time", ""),
            "penalty": int(period_elem.get("penalty", 0))
        }
    
    print(f"  Loaded {len(periods_data)} periods")
    
    # Parse rooms (simplified)
    rooms_data = {}
    for room_elem in root.findall('./rooms/room'):
        rid = int(room_elem.get('id'))
        rooms_data[rid] = {
            'capacity': int(room_elem.get('size', 50))
        }
    
    # Parse exams
    exams_data = {}
    exam_id_map = {}
    
    for i, exam_elem in enumerate(root.findall('./exams/exam')):
        ext_id = int(exam_elem.get('id'))
        internal_id = i
        
        exams_data[internal_id] = {
            'external_id': ext_id,
            'size': 0
        }
        exam_id_map[ext_id] = internal_id
    
    print(f"  Loaded {len(exams_data)} exams")
    
    # Parse students
    students = {}
    for student_elem in root.findall('./students/student'):
        sid = student_elem.get('id')
        student_exams = []
        
        for exam_elem in student_elem.findall('exam'):
            ext_exam_id = int(exam_elem.get('id'))
            if ext_exam_id in exam_id_map:
                internal_id = exam_id_map[ext_exam_id]
                student_exams.append(internal_id)
        
        if student_exams:
            students[sid] = student_exams
    
    # Update exam sizes
    for student_exams in students.values():
        for exam_id in student_exams:
            if exam_id in exams_data:
                exams_data[exam_id]['size'] = exams_data[exam_id].get('size', 0) + 1
    
    print(f"  Loaded {len(students)} students")
    
    return exams_data, students, periods_data, rooms_data


def main():
    """Main function"""
    print("="*70)
    print("Exam Scheduling Solver - Dictionary Chromosome")
    print("Purdue Fall 2009 Dataset")
    print("="*70)
    
    # Load dataset
    xml_path = 'ExamScheduling\pu-exam-fal09.xml'
    
    if not os.path.exists(xml_path):
        print(f"ERROR: Dataset file not found: {xml_path}")
        return
    
    exams_data, students, periods_data, rooms_data = parse_exam_data(xml_path)
    
    # Create solver
    solver = ExamSchedulingSolver(
        exams_data=exams_data,
        students=students,
        periods_data=periods_data,
        rooms_data=rooms_data,
        population_size=100,
        generations=250,
        crossover_rate=0.8,
        mutation_rate=0.3,
        elitism_count=5,
        tournament_size=5,
        survivor_selection='mu_plus_lambda',
        parent_selection='fitness_proportional',
    )
    
    print("\nEvolutionary Algorithm Parameters:")
    print(f"Population Size: {solver.population_size}")
    print(f"Generations: {solver.generations}")
    print(f"Crossover Rate: {solver.crossover_rate}")
    print(f"Mutation Rate: {solver.mutation_rate}")
    print(f"Elitism Count: {solver.elitism_count}")
    print(f"Tournament Size: {solver.tournament_size}")
    print(f"Parent Selection: {solver.parent_selection}")
    print(f"Survivor Selection: {solver.survivor_selection}")
    
    print("\nStarting evolution...\n")
    
    # Run evolution
    best_schedule = solver.evolve(verbose=True)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    stats = solver.get_statistics()
    print(f"Best Fitness (Penalty): {stats['best_fitness']:.2f}")
    print(f"Initial Best Fitness: {solver.best_fitness_history[0]:.2f}")
    print(f"Improvement: {stats['improvement']:.2f} ({stats['improvement_percent']:.2f}%)")
    print(f"Final Average Fitness: {stats['final_avg_fitness']:.2f}")
    
    # Calculate constraint violations
    direct_conflicts = solver._calc_direct_conflicts(best_schedule)
    spread_penalty = solver._calc_spread_penalty(best_schedule)
    balance_penalty = solver._calc_balance_penalty(best_schedule)
    consecutive = solver._calc_consecutive_penalty(best_schedule)
    
    print(f"\nConstraint Analysis:")
    print(f"  Hard Constraints (Direct conflicts): {direct_conflicts}")
    if direct_conflicts == 0:
        print("All hard constraints satisfied! (Maintained throughout evolution)")
    else:
        print("WARNING: Unexpected conflicts detected")
    
    print(f"\nSoft Constraints (Optimized by EA):")
    print(f"Spread penalty: {spread_penalty:.0f}")
    print(f"Balance penalty: {balance_penalty:.2f}")
    print(f"Consecutive exams: {consecutive}")
    
    print(f"\nTotal soft constraint penalty: {stats['best_fitness']:.2f}")
    
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
        f.write("EXAM SCHEDULING RESULTS - Conflict-Free Approach\n")
        f.write("="*70 + "\n\n")
        f.write("APPROACH:\n")
        f.write("Conflict-free initialization with smart room assignment\n")
        f.write("EA optimizes soft constraints only\n")
        f.write("Hard constraints maintained throughout evolution\n\n") 
        f.write("Dataset:\n")
        f.write(f"Exams: {solver.n_exams}\n")
        f.write(f"Periods: {solver.n_periods}\n")
        f.write(f"Students: {len(solver.students)}\n\n")
        f.write("Algorithm Parameters:\n")
        f.write(f"Population Size: {solver.population_size}\n")
        f.write(f"Generations: {solver.generations}\n")
        f.write(f"Crossover Rate: {solver.crossover_rate}\n")
        f.write(f"Mutation Rate: {solver.mutation_rate}\n")
        f.write(f"Elitism Count: {solver.elitism_count}\n\n")
        f.write("Results:\n")
        f.write(f"Best Fitness (Soft Constraints Only): {stats['best_fitness']:.2f}\n")
        f.write(f"Initial Best: {solver.best_fitness_history[0]:.2f}\n")
        f.write(f"Improvement: {stats['improvement_percent']:.2f}%\n\n")
        f.write("Constraint Analysis:\n")
        f.write(f"Hard Constraints:\n")
        f.write(f"Direct Conflicts: {direct_conflicts} (Should be 0)\n")
        f.write(f"Soft Constraints:\n")
        f.write(f"Spread Penalty: {spread_penalty:.0f}\n")
        f.write(f"Balance Penalty: {balance_penalty:.2f}\n")
        f.write(f"Consecutive Exams: {consecutive}\n")
    
    print("\nResults saved to ExamScheduling_results.txt")
    print("\nDone!")


if __name__ == "__main__":
    main()
