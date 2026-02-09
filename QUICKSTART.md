# Quick Start Guide

## Installation

1. Ensure Python 3.7+ is installed:
```bash
python --version
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Running the Programs

### TSP Solver (Qatar Dataset - 194 Cities)

```bash
cd TSP
python tsp_solver.py
```

**What it does:**
- Downloads Qatar TSP dataset (or generates sample if download fails)
- Runs EA for 1000 generations with population of 200
- Outputs:
  - `TSP_convergence.png` - Shows fitness improvement over generations
  - `TSP_best_tour.png` - Visual map of best tour found
  - `TSP_results.txt` - Detailed statistics

**Expected runtime:** 2-5 minutes

---

### Exam Scheduling Solver

```bash
cd ExamScheduling
python exam_scheduler.py
```

**What it does:**
- Generates sample exam scheduling problem (100 exams, 400 students, 15 timeslots)
- Runs EA for 500 generations with population of 150
- Outputs:
  - `ExamScheduling_convergence.png` - Fitness evolution
  - `ExamScheduling_distribution.png` - Exam distribution across timeslots
  - `ExamScheduling_results.txt` - Comprehensive analysis

**Expected runtime:** 1-3 minutes

---

## Understanding the Output

### Convergence Plots
- **Green line (Best)**: Best solution found so far
- **Blue line (Average)**: Average fitness of population
- **Red line (Worst)**: Worst individual in population

**Good convergence**: Lines come together and flatten out

### TSP Tour Plot
- **Red dots**: Cities
- **Blue line**: Tour path
- **Green star**: Start/end city
- **Title**: Shows total distance

### Exam Distribution Plot
- **Blue bars**: Number of exams per timeslot
- **Red dashed line**: Ideal average
- **Good schedule**: Bars close to average height

---

## Customizing Parameters

### For Better TSP Solutions

Edit `tsp_solver.py` and modify:

```python
solver = TSPSolver(
    cities=cities,
    distance_matrix=distance_matrix,
    population_size=300,    # Try 300-500 for better results
    generations=2000,       # Try 1500-3000 for more time
    crossover_rate=0.9,     # Try 0.85-0.95
    mutation_rate=0.1,      # Try 0.05-0.2
    elitism_count=10,       # Try 5-20
    tournament_size=7       # Try 5-10
)
```

### For Better Exam Schedules

Edit `exam_scheduler.py` and modify:

```python
solver = ExamSchedulingSolver(
    exams=exams,
    students=students,
    timeslots=timeslots,
    constraints=constraints,
    population_size=200,    # Try 150-300
    generations=800,        # Try 500-1500
    crossover_rate=0.85,    # Try 0.75-0.9
    mutation_rate=0.25,     # Try 0.15-0.3
    elitism_count=10,       # Try 5-15
    tournament_size=5       # Try 3-7
)
```

---

## Troubleshooting

**Problem**: Module not found error
```
Solution: pip install numpy matplotlib
```

**Problem**: No output files generated
```
Solution: Check you're running from correct directory (TSP/ or ExamScheduling/)
```

**Problem**: Results not improving
```
Solution: 
- Increase population_size
- Increase generations
- Adjust mutation_rate
- Try multiple runs (EA is stochastic)
```

**Problem**: Takes too long to run
```
Solution:
- Decrease population_size (try 100)
- Decrease generations (try 300)
- Results will be less optimal but faster
```

---

## Interpreting Results

### TSP Results

**Good results:**
- Improvement > 40%
- Best fitness plateaus after 500-800 generations
- Tour looks visually reasonable (no excessive crossings)

**Typical values:**
- Qatar dataset: Total distance 8000-12000 (depends on scale)
- Convergence: Stable after 600-900 generations

### Exam Scheduling Results

**Good results:**
- Hard constraint violations = 0
- Improvement > 50%
- < 10% students with consecutive exams
- Relatively even distribution (max/min ratio < 2.0)

**Success criteria:**
1. Zero conflicts (MUST ACHIEVE)
2. Balanced load (SHOULD ACHIEVE)
3. Minimal consecutive exams (NICE TO HAVE)

---

## Next Steps

1. **Experiment**: Try different parameter combinations
2. **Analyze**: Study convergence patterns
3. **Compare**: Run multiple times and compare results
4. **Modify**: Add new mutation operators or selection schemes
5. **Extend**: Try different TSP datasets or exam scenarios

---

## File Structure After Running

```
EA_Assignment/
├── common/
│   └── ea_framework.py
├── TSP/
│   ├── tsp_solver.py
│   ├── TSP_convergence.png       ← Generated
│   ├── TSP_best_tour.png         ← Generated
│   └── TSP_results.txt           ← Generated
├── ExamScheduling/
│   ├── exam_scheduler.py
│   ├── ExamScheduling_convergence.png    ← Generated
│   ├── ExamScheduling_distribution.png   ← Generated
│   └── ExamScheduling_results.txt        ← Generated
├── README.md
├── QUICKSTART.md                 ← This file
└── requirements.txt
```

---

## Tips for Success

1. **Run multiple times**: EA is stochastic, results vary
2. **Monitor convergence**: If not improving, adjust parameters
3. **Start small**: Use default parameters first
4. **Document changes**: Note what parameters work best
5. **Understand the code**: Read comments and docstrings

---

**Happy Optimizing!**
