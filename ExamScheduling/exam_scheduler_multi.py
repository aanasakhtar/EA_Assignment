import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import time
from collections import defaultdict
import os

from ea_framework import EAFramework


class ExamSchedulingSolver(EAFramework):
    """Improved Exam Scheduling solver with smart initialization"""
    
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
        self.n_rooms = len(self.room_ids)
        
        # Build conflict structures
        self.conflict_matrix = self._build_conflict_matrix()
        self.exam_degrees = self._calculate_exam_degrees()
        
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
    
    def _calculate_exam_degrees(self):
        """Calculate conflict degree for each exam"""
        degrees = {}
        for exam_id in self.exam_ids:
            degrees[exam_id] = len(self.conflict_matrix.get(exam_id, set()))
        return degrees
    
    def _smart_feasible_initialization(self):
        """
        SMART initialization:
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
            
            # Find conflict-free periods
            used_periods = set()
            for conflict_exam in self.conflict_matrix.get(exam_id, set()):
                if conflict_exam in schedule:
                    used_periods.add(schedule[conflict_exam]["period"])
            
            valid_periods = [p for p in self.period_ids if p not in used_periods]
            
            if not valid_periods:
                valid_periods = self.period_ids
            
            np.random.shuffle(valid_periods)
            
            assigned = False
            
            # Try periods until room found
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
            
            # If no room fits, assign random room anyway
            if not assigned:
                period = np.random.choice(valid_periods)
                room = np.random.choice(self.room_ids)
                
                schedule[exam_id] = {
                    "period": period,
                    "rooms": [room]
                }
        
        return schedule
    
    def initialize_population(self):
        """Smart conflict-free + room-aware initialization"""
        population = []
        
        for i in range(self.population_size):
            schedule = self._smart_feasible_initialization()
            population.append(schedule)
        
        return population
    
    def _calc_room_capacity_violations(self, schedule):
        """Count exams assigned to rooms that are too small"""
        violations = 0
        
        for exam_id, assignment in schedule.items():
            exam_size = self.exams_data[exam_id].get("size", 1)
            
            for room_id in assignment["rooms"]:
                room_cap = self.rooms_data[room_id]["capacity"]
                
                if room_cap < exam_size:
                    violations += 1
        
        return violations
    
    def _calc_room_clashes(self, schedule):
        """Count cases where two exams use the same room in the same period"""
        clashes = 0
        room_usage = defaultdict(list)
        
        for exam_id, assignment in schedule.items():
            period = assignment["period"]
            
            for room_id in assignment["rooms"]:
                key = (period, room_id)
                room_usage[key].append(exam_id)
        
        for key, exams in room_usage.items():
            if len(exams) > 1:
                clashes += len(exams) - 1
        
        return clashes
    
    def _calc_room_waste(self, schedule):
        """Calculate wasted room capacity"""
        waste = 0
        
        for exam_id, assign in schedule.items():
            exam_size = self.exams_data[exam_id]["size"]
            
            for room in assign["rooms"]:
                cap = self.rooms_data[room]["capacity"]
                
                if cap > exam_size:
                    waste += (cap - exam_size)
        
        return waste
    
    def _calc_more_than_two_exams_day(self, schedule):
        """Penalty for students with >2 exams per day"""
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
        """Fitness based on CPSolver benchmark weights"""
        # CPSolver weights
        W_DIRECT = 1000.0
        W_TWO_IN_ROW = 10.0
        W_MORE_THAN_TWO_DAY = 100.0
        W_ROOM_SIZE = 0.001
        W_ROOM_CLASH = 1000.0
        
        penalty = 0
        
        # Hard-like penalties
        penalty += W_DIRECT * self._calc_direct_conflicts(schedule)
        penalty += W_ROOM_CLASH * self._calc_room_clashes(schedule)
        
        # Room quality penalties
        penalty += 500 * self._calc_room_capacity_violations(schedule)
        penalty += W_ROOM_SIZE * self._calc_room_waste(schedule)
        
        # Soft constraints
        penalty += W_TWO_IN_ROW * self._calc_consecutive_penalty(schedule)
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
        """Uniform crossover with conflict repair"""
        offspring1 = {}
        offspring2 = {}
        
        for exam_id in self.exam_ids:
            if np.random.rand() < 0.5:
                offspring1[exam_id] = parent1[exam_id].copy()
                offspring2[exam_id] = parent2[exam_id].copy()
            else:
                offspring1[exam_id] = parent2[exam_id].copy()
                offspring2[exam_id] = parent1[exam_id].copy()
        
        offspring1 = self._repair_conflicts(offspring1)
        offspring2 = self._repair_conflicts(offspring2)
        
        return offspring1, offspring2
    
    def _repair_conflicts(self, schedule):
        """Repair offspring schedule after crossover"""
        max_iterations = 200
        
        for _ in range(max_iterations):
            # Fix student conflicts
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
                            schedule[exam_id]["period"] = np.random.choice(self.period_ids)
                        
                        conflict_fixed = True
                        break
                
                if conflict_fixed:
                    break
            
            if conflict_fixed:
                continue
            
            # Fix room clashes + capacity
            room_usage = defaultdict(set)
            
            for exam_id in self.exam_ids:
                exam_size = self.exams_data[exam_id].get("size", 1)
                period = schedule[exam_id]["period"]
                
                taken_rooms = room_usage[period]
                
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
                    fitting_rooms = [
                        r for r in self.room_ids
                        if self.rooms_data[r]["capacity"] >= exam_size
                    ]
                    
                    if fitting_rooms:
                        schedule[exam_id]["rooms"] = [np.random.choice(fitting_rooms)]
                    else:
                        schedule[exam_id]["rooms"] = [np.random.choice(self.room_ids)]
            
            break
        
        return schedule
    
    def mutate(self, schedule):
        """Mutate by changing periods while maintaining conflict-free property"""
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


def parse_purdue_data_enhanced(file_path):
    """Parse Purdue XML with enhanced information extraction"""
    print(f"Parsing dataset: {file_path}")
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    periods_data = {}
    for idx, period_elem in enumerate(root.findall('./periods/period')):
        pid = int(period_elem.get("id"))
        
        periods_data[pid] = {
            "index": idx,
            "day": period_elem.get("day", ""),
            "time": period_elem.get("time", ""),
            "penalty": int(period_elem.get("penalty", 0))
        }
    
    print(f"  Found {len(periods_data)} timeslots (periods)")
    
    rooms_data = {}
    for room_elem in root.findall('./rooms/room'):
        rid = int(room_elem.get('id'))
        
        available_periods = set(periods_data.keys())
        period_penalties = {}
        
        for period_elem in room_elem.findall('period'):
            pid = int(period_elem.get('id'))
            penalty = int(period_elem.get('penalty', 0))
            
            if penalty >= 4:
                available_periods.discard(pid)
            else:
                period_penalties[pid] = penalty
        
        rooms_data[rid] = {
            'capacity': int(room_elem.get('size', 50)),
            'available_periods': available_periods,
            'period_penalties': period_penalties
        }
    
    exams_data = {}
    exam_id_map = {}
    
    for i, exam_elem in enumerate(root.findall('./exams/exam')):
        ext_id = int(exam_elem.get('id'))
        internal_id = i
        
        exams_data[internal_id] = {
            'external_id': ext_id,
            'length': int(exam_elem.get('length', 120)),
            'size': 0
        }
        exam_id_map[ext_id] = internal_id
    
    print(f"  Found {len(exams_data)} exams")
    
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
    
    for student_exams in students.values():
        for exam_id in student_exams:
            if exam_id in exams_data:
                exams_data[exam_id]['size'] = exams_data[exam_id].get('size', 0) + 1
    
    print(f"  Found {len(students)} students")
    
    return exams_data, students, periods_data, rooms_data


def run_k_runs(parent_scheme, survivor_scheme, exams_data, students, periods_data, rooms_data, k=10):
    """Execute the EA K times to get statistical results"""
    all_runs_bsf = []
    all_runs_asf = []
    all_final_fitness = []
    all_runtimes = []
    
    print(f"\n{'='*70}")
    print(f"Running: Parent={parent_scheme} + Survivor={survivor_scheme}")
    print(f"{'='*70}")
    
    for run in range(k):
        print(f"  Run {run+1}/{k}...", end=' ', flush=True)
        
        start_time = time.time()
        
        solver = ExamSchedulingSolver(
            exams_data=exams_data,
            students=students,
            periods_data=periods_data,
            rooms_data=rooms_data,
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
        
        solver.evolve(verbose=False)
        
        runtime = time.time() - start_time
        
        all_runs_bsf.append(solver.best_fitness_history)
        all_runs_asf.append(solver.avg_fitness_history)
        all_final_fitness.append(solver.best_fitness)
        all_runtimes.append(runtime)
        
        print(f"BSF={solver.best_fitness:.0f}, {runtime:.1f}s")
    
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
    print(f"Mean Final BSF: {results['mean_final_fitness']:.2f} +/- {results['std_final_fitness']:.2f}")
    print(f"Best Final BSF: {results['best_final_fitness']:.2f}")
    print(f"Worst Final BSF: {results['worst_final_fitness']:.2f}")
    print(f"Mean Runtime: {results['mean_runtime']:.2f}s")
    
    return results


def plot_single_combination(label, results, save_path=None):
    """Plot Avg BSF and Avg ASF side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    generations = range(len(results['avg_bsf_history']))
    
    ax1.plot(generations, results['avg_bsf_history'], 'b-', linewidth=2)
    ax1.set_xlabel('Generation', fontsize=11)
    ax1.set_ylabel('Average Best-So-Far (BSF)', fontsize=11)
    ax1.set_title('Avg. BSF vs Generation', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(generations, results['avg_asf_history'], 'r-', linewidth=2)
    ax2.set_xlabel('Generation', fontsize=11)
    ax2.set_ylabel('Average Average-Fitness (ASF)', fontsize=11)
    ax2.set_title('Avg. ASF vs Generation', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'{label}\n(K={results["k"]} runs, 50 generations)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        filepath = os.path.abspath(os.path.join(os.getcwd(), '..', save_path))
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Plot saved to: {filepath}")
    
    plt.close()


def plot_all_combinations(results_dict, save_path='exam_scheduling_comparison.png'):
    """Plot all combinations side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    schemes = list(results_dict.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(schemes)))
    
    for i, (scheme, results) in enumerate(results_dict.items()):
        generations = range(len(results['avg_bsf_history']))
        ax1.plot(generations, results['avg_bsf_history'], label=scheme,
                 color=colors[i], linewidth=2)
    
    ax1.set_xlabel('Generation', fontsize=11)
    ax1.set_ylabel('Average Best-So-Far (BSF)', fontsize=11)
    ax1.set_title('Avg. BSF vs Generation (All Schemes)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    
    for i, (scheme, results) in enumerate(results_dict.items()):
        generations = range(len(results['avg_asf_history']))
        ax2.plot(generations, results['avg_asf_history'], label=scheme,
                 color=colors[i], linewidth=2)
    
    ax2.set_xlabel('Generation', fontsize=11)
    ax2.set_ylabel('Average Average-Fitness (ASF)', fontsize=11)
    ax2.set_title('Avg. ASF vs Generation (All Schemes)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Exam Scheduling: Selection Schemes Comparison (Smart Initialization)\n'
                 'Parameters: μ=30, λ=10, Generations=50, K=10 runs',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filepath = os.path.abspath(os.path.join(os.getcwd(), '..', save_path))
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {filepath}")
    
    plt.close()


def main():
    """Main function"""
    print("="*80)
    print(" "*15 + "EXAM SCHEDULING - SMART INITIALIZATION")
    print(" "*10 + "Dictionary Chromosome - Purdue Fall 2009 Dataset")
    print("="*80)
    
    output_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
    print(f"\nOutput directory: {output_dir}")
    
    xml_path = os.path.join(os.getcwd(), 'pu-exam-fal09.xml')
    
    if not os.path.exists(xml_path):
        print(f"\nERROR: Dataset file not found: {xml_path}")
        print("Please ensure 'pu-exam-fal09.xml' is in the ExamScheduling directory.")
        return
    
    exams_data, students, periods_data, rooms_data = parse_purdue_data_enhanced(xml_path)
    
    print(f"\nDataset Summary:")
    print(f"Total Exams: {len(exams_data)}")
    print(f"Total Students: {len(students)}")
    print(f"Total Timeslots: {len(periods_data)}")
    
    schemes_to_test = [
        ('fitness_proportional', 'mu_plus_lambda'),
        ('tournament', 'mu_plus_lambda'),
        ('random', 'generational')
    ]
    
    scheme_labels = [
        'FPS + Truncation',
        'Tournament + Truncation',
        'Random + Generational'
    ]
    
    print("\n" + "="*80)
    print("TESTING SELECTION SCHEMES")
    print("="*80)
    print("\nThe following combinations will be tested (K=10 runs each):")
    for i, label in enumerate(scheme_labels, 1):
        print(f"  {i}. {label}")
    
    results = {}
    
    for (p_scheme, s_scheme), label in zip(schemes_to_test, scheme_labels):
        results[label] = run_k_runs(p_scheme, s_scheme, exams_data, students, 
                                    periods_data, rooms_data, k=10)
    
    print("\n" + "="*80)
    print("GENERATION-BY-GENERATION TABLES & PLOTS")
    print("="*80)
    
    for label in scheme_labels:                
        plot_name = label.replace(' ', '_').replace('+', '_') + '_plot.png'
        plot_single_combination(label, results[label], save_path=plot_name)
    
    print("\nGenerating combined comparison plot...")
    plot_all_combinations(results, 'exam_scheduling_comparison_smart.png')
    
    print("\n" + "="*80)
    print("Q2 COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    for label in scheme_labels:
        tag = label.replace(' ', '_').replace('+', '_')
        print(f"{tag}_table.csv (data)")
        print(f"{tag}_table.png (table image)")
        print(f"{tag}_plot.png (Avg BSF & ASF plot)")
    print(f"exam_scheduling_comparison_smart.png (all schemes side by side)")


if __name__ == "__main__":
    main()
