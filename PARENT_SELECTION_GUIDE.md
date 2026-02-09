# Parent Selection Methods in Evolutionary Algorithms

## Overview

Parent selection determines which individuals from the current population get to reproduce and create offspring. This is a critical component that directly affects:
- **Selection pressure**: How strongly better individuals are favored
- **Diversity**: How varied the population remains
- **Convergence speed**: How quickly the algorithm improves
- **Risk of premature convergence**: Getting stuck in local optima

## The Three Main Selection Methods

### 1. Fitness Proportional Selection (Roulette Wheel)

**Proposed by**: John Holland (father of Genetic Algorithms)

**How it works**:
```
For minimization problems:
1. Invert fitness: better_fitness = max_fitness - fitness + ε
2. Calculate probabilities: P(i) = better_fitness(i) / Σ better_fitness
3. Select by sampling from this distribution
```

**Visualization**:
```
Roulette Wheel:
┌───────────────────────────────────────┐
│        Individual 1 (best)            │  40% of wheel
├───────────────────┬───────────────────┤
│   Individual 2    │  Individual 3     │  30% + 20%
├──────────┬────────┴───────────────────┤
│   Ind 4  │      Individual 5          │  5% + 5%
└──────────┴────────────────────────────┘

Spin the wheel → Selection probability ∝ fitness
```

**Characteristics**:

✅ **Advantages**:
- Simple and intuitive
- Gives all individuals a chance (no individual has 0 probability)
- Natural selection analogy

❌ **Disadvantages**:
- **Premature convergence**: When few individuals are much better, they dominate
- **Loss of selection pressure**: When all fitnesses are similar (late in run)
- **Scaling issues**: Very sensitive to fitness distribution

**Selection Pressure**:
- **Early generations**: HIGH (when fitness variance is high)
- **Late generations**: LOW (when population has converged)

**Example Scenario**:
```
Population fitnesses: [10, 20, 30, 100, 100]
Inverted (for min): [90, 80, 70, 0, 0]
Probabilities: [37.5%, 33.3%, 29.2%, 0%, 0%]

Result: Best individual (fitness=10) gets 37.5% chance
        Worst individuals (fitness=100) get 0% chance → Premature convergence!
```

**When to use**:
- ✅ Early exploration phase
- ✅ When fitness variance is moderate
- ❌ NOT for highly variable fitness landscapes
- ❌ NOT when risk of premature convergence is high

---

### 2. Rank-Based Selection

**How it works**:
```
1. Sort population by fitness
2. Assign ranks: best = n, worst = 1
3. Calculate probabilities: P(i) = rank(i) / Σ ranks
4. Select based on rank probabilities
```

**Example**:
```
Population fitnesses: [10, 20, 30, 100, 100]
Sorted by fitness:    [10, 20, 30, 100, 100]
Ranks assigned:       [5,  4,  3,  2,   1 ]
Probabilities:        [33%, 27%, 20%, 13%, 7%]

Result: More balanced selection, less domination by super-fit individuals
```

**Characteristics**:

✅ **Advantages**:
- **Consistent selection pressure** throughout the run
- Prevents premature convergence
- Not affected by fitness scaling
- Works well when fitnesses are very different

❌ **Disadvantages**:
- Slower convergence than fitness proportional
- Ignores magnitude of fitness differences
- More complex to implement

**Selection Pressure**:
- **Constant** throughout the run (doesn't depend on fitness values)

**Comparison with Fitness Proportional**:
```
Scenario 1: Large fitness differences
Fitnesses:  [10, 20, 30, 90, 100]

Fitness Proportional (inverted):
P = [47%, 37%, 29%, 5%, 0%]  ← Best dominates!

Rank-based:
P = [33%, 27%, 20%, 13%, 7%]  ← More balanced

---

Scenario 2: Similar fitnesses (late in run)
Fitnesses:  [49, 50, 50, 51, 51]

Fitness Proportional (inverted):
P = [20.4%, 20.0%, 20.0%, 19.8%, 19.8%]  ← Almost no selection pressure!

Rank-based:
P = [33%, 27%, 20%, 13%, 7%]  ← Still maintains pressure
```

**When to use**:
- ✅ When you want consistent performance
- ✅ Throughout entire run (early and late)
- ✅ When fitness values have extreme differences
- ✅ To prevent premature convergence

---

### 3. Tournament Selection

**How it works**:
```
1. Randomly select k individuals (tournament size)
2. Choose the best among them
3. Repeat for each parent needed
```

**Example** (k=3):
```
Population: [ind1(fit=50), ind2(fit=30), ind3(fit=70), ind4(fit=20), ind5(fit=60)]

Tournament 1: Randomly pick [ind2, ind4, ind5]
              Fitnesses: [30, 20, 60]
              Winner: ind4 (fitness=20, best)

Tournament 2: Randomly pick [ind1, ind3, ind5]
              Fitnesses: [50, 70, 60]
              Winner: ind1 (fitness=50, best)

Parents: ind4 and ind1
```

**Characteristics**:

✅ **Advantages**:
- **Tunable selection pressure** (via tournament size k)
- Efficient (no need to evaluate entire population)
- Works with any fitness scale
- Easy to implement and parallelize

❌ **Disadvantages**:
- Stochastic (same setup can give different results)
- Requires tuning of k

**Selection Pressure vs Tournament Size**:
```
k=2:  Low pressure   (50% chance best is selected)
k=3:  Medium         (75% chance)
k=5:  High           (93.75% chance)
k=10: Very high      (99.9% chance)
k=n:  Deterministic  (always selects best)
```

**When to use**:
- ✅ Default choice for most problems
- ✅ When you want tunable selection pressure
- ✅ For parallel implementations
- ✅ When fitness scaling is unpredictable

---

## Comparison Table

| Feature | Fitness Proportional | Rank-Based | Tournament |
|---------|---------------------|------------|------------|
| **Selection Pressure** | Variable (high→low) | Constant | Tunable (via k) |
| **Premature Convergence Risk** | High | Low | Medium |
| **Late-Run Performance** | Poor (low pressure) | Good | Good |
| **Computational Cost** | O(n) | O(n log n) | O(k) |
| **Fitness Scaling Sensitivity** | Very high | None | None |
| **Tuning Required** | None | None | Tournament size |
| **Guarantees Diversity** | No | Better | Medium |

## Selection Pressure Over Time

```
Selection Pressure
    │
    │  Fitness Proportional
    │  ╱╲
    │ ╱  ╲___________
    │╱               
    ├─────────────────────────────────
    │     Rank-Based
    │  ─────────────────────────
    │
    │     Tournament (k=5)
    │  ──────────────────────────
    │
    │     Tournament (k=3)
    │  ─────────────────────────
    │
    └─────────────────────────────────→
      Early         Mid          Late
              Generation
```

## Practical Examples

### Example 1: TSP with Fitness Proportional Selection

```python
from TSP.tsp_solver import TSPSolver, load_qatar_dataset

cities, distance_matrix = load_qatar_dataset()

solver = TSPSolver(
    cities=cities,
    distance_matrix=distance_matrix,
    population_size=100,
    generations=500,
    parent_selection='fitness_proportional',  # Roulette wheel
    mutation_rate=0.2  # Higher mutation to maintain diversity
)

best_tour = solver.evolve()
```

**Expected behavior**:
- Fast initial improvement
- Risk of premature convergence
- May need higher mutation rate

### Example 2: TSP with Rank Selection

```python
solver = TSPSolver(
    cities=cities,
    distance_matrix=distance_matrix,
    population_size=100,
    generations=500,
    parent_selection='rank',  # Rank-based
    mutation_rate=0.15
)

best_tour = solver.evolve()
```

**Expected behavior**:
- Steady improvement throughout
- Better late-run performance
- More consistent results

### Example 3: TSP with Tournament Selection

```python
solver = TSPSolver(
    cities=cities,
    distance_matrix=distance_matrix,
    population_size=100,
    generations=500,
    parent_selection='tournament',
    tournament_size=5,  # Tune this!
    mutation_rate=0.15
)

best_tour = solver.evolve()
```

**Expected behavior**:
- Balanced exploration/exploitation
- Tunable via tournament size
- Generally good performance

### Example 4: Comparing All Methods

```python
import numpy as np
import matplotlib.pyplot as plt

methods = ['fitness_proportional', 'rank', 'tournament']
results = {}

for method in methods:
    solver = TSPSolver(
        cities=cities,
        distance_matrix=distance_matrix,
        population_size=100,
        generations=500,
        parent_selection=method,
        tournament_size=5 if method == 'tournament' else 3
    )
    
    solver.evolve(verbose=False)
    results[method] = {
        'best_fitness': solver.best_fitness,
        'history': solver.best_fitness_history
    }

# Plot comparison
plt.figure(figsize=(12, 6))
for method, data in results.items():
    plt.plot(data['history'], label=method, linewidth=2)

plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Parent Selection Methods Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Theory: Why Selection Matters

### Selection Pressure vs Diversity Trade-off

```
High Selection Pressure:
├─ Fast convergence
├─ Risk of premature convergence
├─ Low diversity
└─ Gets stuck in local optima

Low Selection Pressure:
├─ Slow convergence
├─ Maintains diversity
├─ Explores more of search space
└─ May not converge in time

Optimal:
└─ High early (exploit good solutions)
   Low late (maintain diversity)
```

### The Schema Theorem (Holland, 1975)

Fitness proportional selection is theoretically grounded in the **Schema Theorem**:

> "Short, low-order, above-average schemata receive exponentially increasing trials in subsequent generations"

However, this assumes:
- Infinite population
- No sampling errors
- Fitness scaling issues don't occur

In practice, these assumptions don't hold, which is why other methods often work better!

## Common Mistakes

### ❌ Mistake 1: Using fitness proportional with extreme fitness values
```python
# DON'T DO THIS if fitnesses can vary wildly
solver = TSPSolver(
    parent_selection='fitness_proportional'  # Will cause premature convergence!
)
```

**Fix**: Use rank or tournament selection

### ❌ Mistake 2: Tournament size too large
```python
solver = TSPSolver(
    parent_selection='tournament',
    tournament_size=50  # In population of 100 → too high!
)
```

**Fix**: Use k = 3 to 7 for most problems

### ❌ Mistake 3: Not adapting mutation rate to selection method
```python
# High selection pressure + low mutation = premature convergence
solver = TSPSolver(
    parent_selection='fitness_proportional',  # High pressure
    mutation_rate=0.05  # Too low!
)
```

**Fix**: Higher selection pressure → Higher mutation rate

## Advanced: Hybrid Approaches

### Adaptive Selection

Change selection method during run:
```python
def adaptive_parent_selection(generation, max_generations):
    ratio = generation / max_generations
    
    if ratio < 0.3:
        return 'fitness_proportional'  # High pressure early
    elif ratio < 0.7:
        return 'tournament'  # Balanced middle
    else:
        return 'rank'  # Consistent pressure late
```

### Fitness Scaling

Improve fitness proportional selection:
```python
# Sigma scaling
scaled_fitness = max(fitness - (mean - 2*std), 0)

# Linear scaling
scaled_fitness = a * fitness + b  # where a, b chosen to set mean/max
```

## Recommendations by Problem Type

### TSP (Travelling Salesman)
- **Best**: Tournament (k=5) or Rank
- **Why**: Consistent pressure, prevents premature convergence

### Exam Scheduling
- **Best**: Rank or Tournament (k=3)
- **Why**: Maintains diversity, helps with constraint satisfaction

### Continuous Optimization
- **Best**: Tournament (k=3) or Fitness Proportional with scaling
- **Why**: Smooth fitness landscape allows proportional selection

### Multi-modal Problems
- **Best**: Rank or Tournament (k=3)
- **Why**: Maintains diversity to explore multiple peaks

## Summary

**Key Takeaways**:

1. **Fitness Proportional (Roulette Wheel)**:
   - Simple and classic (Holland's original method)
   - High risk of premature convergence
   - Use with caution or with fitness scaling

2. **Rank-Based**:
   - Most consistent performance
   - Safe default choice
   - Best for maintaining diversity

3. **Tournament**:
   - Most flexible (tunable pressure)
   - Generally best practical performance
   - Recommended for most problems

**Quick Decision Guide**:
- **Unsure? → Use Tournament (k=5)**
- **Need consistency? → Use Rank**
- **Studying theory? → Try Fitness Proportional**
- **Need speed? → Use Tournament (smallest O(k) cost)**

## References

- Holland, J. H. (1975). Adaptation in Natural and Artificial Systems
- Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization and Machine Learning
- Eiben & Smith (2015). Introduction to Evolutionary Computing
- Baker, J. E. (1985). Adaptive Selection Methods for Genetic Algorithms

---

**All three methods are now implemented in your EA framework!**
