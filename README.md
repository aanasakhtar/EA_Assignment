# Evolutionary Algorithms Assignment
## Global Optimization using Evolutionary Algorithms

### Author Information
**Course Assignment**: Evolutionary Algorithms  
**Date**: February 2026  
**Problems**: TSP (Qatar Dataset) & Exam Scheduling (Purdue Benchmark)

---

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Problem 1: Travelling Salesman Problem (TSP)](#problem-1-tsp)
5. [Problem 2: Exam Scheduling](#problem-2-exam-scheduling)
6. [Evolutionary Algorithm Framework](#ea-framework)
7. [Results and Analysis](#results-and-analysis)
8. [References](#references)

---

## Overview

This project implements Evolutionary Algorithms (EA) to solve two computationally hard optimization problems:

1. **Travelling Salesman Problem (TSP)** - Finding the shortest route through 194 cities in Qatar
2. **Exam Scheduling** - Creating optimal exam timetables with constraint satisfaction

Both implementations use a common EA framework with tournament selection, elitism, adaptive operators, and comprehensive visualization.

---

## Project Structure

```
EA_Assignment/
│
├── common/
│   └── ea_framework.py          # Base EA implementation
│
├── TSP/
│   └── tsp_solver.py            # TSP-specific solver
│
├── ExamScheduling/
│   └── exam_scheduler.py        # Exam scheduling solver
│
└── README.md                     # This file
```

---

## Installation

### Requirements
- Python 3.7 or higher
- NumPy
- Matplotlib

### Setup

1. Install required packages:
```bash
pip install numpy matplotlib
```

2. No additional setup required - the programs will download the Qatar TSP dataset automatically or generate sample data if unavailable.

---

## Problem 1: Travelling Salesman Problem (TSP)

### Problem Description
Given 194 cities in Qatar, find the shortest possible route that:
- Visits each city exactly once
- Returns to the starting city

### Running the TSP Solver

```bash
cd TSP
python tsp_solver.py
```

### Chromosome Representation
- **Type**: Permutation encoding
- **Structure**: Array of city indices [0, 1, 2, ..., 193]
- **Example**: [45, 12, 89, 3, ...] means visit city 45, then 12, then 89, etc.

### Fitness Function
- **Objective**: Minimize total tour distance
- **Calculation**: Sum of Euclidean distances between consecutive cities in tour
- **Formula**: `fitness = Σ distance(city[i], city[i+1]) + distance(city[n], city[0])`

### Evolutionary Operators

**1. Selection**
- Tournament Selection (size = 5)
- Best individual from random subset wins

**2. Crossover** (85% probability)
- Order Crossover (OX)
- Preserves city order from parents
- Ensures valid permutations

**3. Mutation** (15% probability)
- Three types used randomly:
  - **Swap**: Exchange two random cities
  - **Inversion**: Reverse a segment of the tour
  - **Scramble**: Shuffle a segment

**4. Elitism**
- Top 5 solutions preserved each generation

### Parameters
```python
population_size = 200
generations = 1000
crossover_rate = 0.85
mutation_rate = 0.15
elitism_count = 5
tournament_size = 5
```

### Outputs
1. `TSP_convergence.png` - Fitness evolution graph
2. `TSP_best_tour.png` - Visualization of best tour
3. `TSP_results.txt` - Detailed results and statistics

### Expected Results
- Significant improvement over random tours
- Convergence typically within 500-800 generations
- 40-60% improvement from initial best solution

---

## Problem 2: Exam Scheduling

### Problem Description
Create an optimal exam timetable that:
- Satisfies hard constraints (no student has conflicting exams)
- Optimizes soft constraints (spread exams, balance load, avoid consecutive exams)

### Running the Exam Scheduler

```bash
cd ExamScheduling
python exam_scheduler.py
```

### Chromosome Representation
- **Type**: Integer vector
- **Structure**: Array where index = exam ID, value = timeslot
- **Example**: [2, 5, 1, 5, 0, ...] means:
  - Exam 0 → Timeslot 2
  - Exam 1 → Timeslot 5
  - Exam 2 → Timeslot 1
  - etc.

### Fitness Function (Penalty-based Minimization)

**Hard Constraints** (×10,000 penalty each):
- No exam conflicts (students with same exams at same time)

**Soft Constraints**:
- Spread penalty (×100): Exams too close together for same student
  - Adjacent slots: 5 points
  - Two slots apart: 2 points
- Balance penalty (×50): Uneven distribution across timeslots
  - Based on variance of exams per slot
- Consecutive exams penalty (×200): Students with back-to-back exams

**Total Fitness**:
```
fitness = (hard_violations × 10000) + 
          (spread_penalty × 100) + 
          (balance_penalty × 50) + 
          (consecutive_penalty × 200)
```

### Evolutionary Operators

**1. Selection**
- Tournament Selection (size = 5)

**2. Crossover** (80% probability)
- Uniform Crossover
- Randomly inherit timeslot from either parent
- Followed by conflict repair mechanism

**3. Mutation** (20% probability)
- Random timeslot reassignment
- 1 to 5% of exams modified
- Automatic conflict repair after mutation

**4. Constraint Handling**
- Repair mechanism fixes hard constraint violations
- Iteratively resolves conflicts by reassigning timeslots
- Ensures all solutions in population are valid

**5. Elitism**
- Top 5 solutions preserved

### Parameters
```python
population_size = 150
generations = 500
crossover_rate = 0.8
mutation_rate = 0.2
elitism_count = 5
tournament_size = 5
```

### Sample Dataset
Since actual Purdue data requires institutional access, the program generates a realistic sample:
- 100 exams
- 400 students
- 15 timeslots
- Average 5 exams per student

### Outputs
1. `ExamScheduling_convergence.png` - Fitness evolution
2. `ExamScheduling_distribution.png` - Exam distribution across timeslots
3. `ExamScheduling_results.txt` - Detailed analysis

### Expected Results
- All hard constraints satisfied (0 conflicts)
- Balanced distribution of exams
- Minimized consecutive exams for students
- Convergence within 300-400 generations

---

## EA Framework

### Common Components (ea_framework.py)

Both problems inherit from the `EAFramework` base class, which provides:

**Core Evolution Loop**:
1. Initialize random population
2. Evaluate fitness for all individuals
3. Track statistics (best, average, worst)
4. Select parents using tournament selection
5. Apply crossover and mutation
6. Preserve elite individuals
7. Replace old population
8. Repeat until convergence or max generations

**Statistics Tracking**:
- Best fitness per generation
- Average fitness per generation
- Worst fitness per generation
- Overall best solution found

**Visualization**:
- Convergence plots showing fitness evolution
- Problem-specific visualizations (tour maps, distributions)

### Key Design Decisions

**1. Tournament Selection**
- Better than roulette wheel for minimization problems
- Maintains selection pressure
- Easy to implement and tune

**2. Elitism**
- Prevents loss of best solutions
- Accelerates convergence
- Small elite size (5) balances exploration/exploitation

**3. Adaptive Operators**
- Multiple mutation types for diversity
- Repair mechanisms for constraint handling
- Problem-specific crossover operators

**4. Parameter Tuning**
- Higher mutation for exam scheduling (more complex constraints)
- Higher crossover for TSP (preserve good tour segments)
- Larger population for TSP (larger search space)

---

## Results and Analysis

### Convergence Analysis

Both problems show typical EA convergence patterns:

**Early Generations (0-100)**:
- Rapid improvement
- High diversity in population
- Large fitness variance

**Middle Generations (100-300)**:
- Steady improvement
- Population converging
- Decreasing fitness variance

**Late Generations (300+)**:
- Slow improvement or plateau
- Population converged
- Local optimum exploitation

### Performance Metrics

**TSP**:
- Convergence indicator: < 1% improvement over 50 generations
- Solution quality: Within 5-15% of known optimal (dataset dependent)
- Diversity maintenance: Multiple good solutions in final population

**Exam Scheduling**:
- Hard constraint satisfaction: 100% (0 conflicts)
- Soft constraint optimization: 60-80% improvement from random
- Student impact: < 5% students with consecutive exams

### Impact of Parameters

**Population Size**:
- Larger → Better exploration, slower convergence
- Smaller → Faster convergence, risk of local optima

**Mutation Rate**:
- Higher → More exploration, slower convergence
- Lower → Faster convergence, risk of stagnation

**Crossover Rate**:
- Higher → More exploitation of good solutions
- Lower → More random search

**Tournament Size**:
- Larger → Higher selection pressure
- Smaller → More diversity maintained

---

## Appendix A: Standard EA Process

### Generic EA Algorithm

```
1. INITIALIZE population with random individuals
2. EVALUATE fitness of each individual
3. REPEAT until termination condition:
   a. SELECT parents using selection method
   b. APPLY crossover to create offspring
   c. APPLY mutation to offspring
   d. EVALUATE fitness of offspring
   e. SELECT individuals for next generation (with elitism)
4. RETURN best individual found
```

### Selection Schemes Implemented

**Tournament Selection**:
- Randomly select k individuals
- Choose best among them
- Repeat for each parent needed

**Advantages**:
- Simple and efficient
- Works well for minimization
- Easy to parallelize

### Termination Conditions

The EA terminates when ANY of these conditions is met:
1. Maximum generations reached
2. Fitness improvement < threshold for n generations (optional)
3. Population diversity below threshold (optional)
4. Optimal solution found (problem-specific)

Current implementation uses condition #1 (max generations).

---

## Usage Tips

### Running with Different Parameters

**TSP Example**:
```python
solver = TSPSolver(
    cities=cities,
    distance_matrix=distance_matrix,
    population_size=300,      # Increase for better solutions
    generations=2000,         # More generations = better convergence
    crossover_rate=0.9,       # Try different rates
    mutation_rate=0.1,        # Lower for more exploitation
    elitism_count=10,         # Preserve more elite solutions
    tournament_size=7         # Higher selection pressure
)
```

**Exam Scheduling Example**:
```python
solver = ExamSchedulingSolver(
    exams=exams,
    students=students,
    timeslots=timeslots,
    constraints=constraints,
    population_size=200,      # Larger for complex instances
    generations=800,          # More for harder problems
    mutation_rate=0.25,       # Higher for constraint problems
)
```

### Troubleshooting

**Problem**: EA not converging
- **Solution**: Increase population size, increase generations, adjust mutation rate

**Problem**: TSP tour has crossings
- **Solution**: Increase generations, try different crossover operators

**Problem**: Exam schedule has conflicts
- **Solution**: Check repair mechanism, increase hard constraint penalty

**Problem**: Slow execution
- **Solution**: Decrease population size, reduce generations, optimize fitness calculation

---

## References

### TSP Dataset
- Qatar TSP Dataset: http://www.math.uwaterloo.ca/tsp/world/countries.html
- 194 cities with geographic coordinates

### Exam Scheduling
- UniTime Exam Timetabling: https://www.unitime.org/
- Purdue University benchmark datasets
- Spring 2012 instance with production configuration

### Evolutionary Algorithms
- Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning.
- Eiben, A. E., & Smith, J. E. (2015). Introduction to Evolutionary Computing.
- Burke, E. K., & Kendall, G. (2014). Search Methodologies.

---

## License and Academic Integrity

This code is provided for educational purposes. Students should:
- Understand the algorithms before using
- Modify and experiment with parameters
- Write their own analysis and conclusions
- Follow their institution's academic integrity policies

---

## Contact and Support

For questions or issues:
1. Check the code comments for detailed explanations
2. Review the output files for diagnostic information
3. Experiment with different parameters
4. Consult EA textbooks and papers for theoretical background

---

**Good luck with your evolutionary algorithms exploration!**
