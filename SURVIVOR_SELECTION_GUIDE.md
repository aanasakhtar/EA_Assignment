# Survivor Selection in Evolutionary Algorithms

## Overview

Survivor selection (also called replacement or environmental selection) determines which individuals survive from one generation to the next. This is a critical component that many implementations overlook.

## The Complete EA Cycle

```
┌─────────────────┐
│ Initialization  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Population    │◄──────────┐
└────────┬────────┘           │
         │                    │
         ▼                    │
┌─────────────────┐           │
│ Parent Selection│           │
└────────┬────────┘           │
         │                    │
         ▼                    │
┌─────────────────┐           │
│    Parents      │           │
└────────┬────────┘           │
         │                    │
         ▼                    │
┌─────────────────┐           │
│  Recombination  │           │
│   (Crossover)   │           │
└────────┬────────┘           │
         │                    │
         ▼                    │
┌─────────────────┐           │
│    Mutation     │           │
└────────┬────────┘           │
         │                    │
         ▼                    │
┌─────────────────┐           │
│   Offspring     │           │
└────────┬────────┘           │
         │                    │
         ▼                    │
┌─────────────────┐           │
│    Survivor     │───────────┘
│   Selection     │    (Next Generation)
└────────┬────────┘
         │
         ▼
    Termination?
```

## Key Distinction

### Parent Selection
- **Purpose**: Choose parents for reproduction
- **Input**: Current population
- **Output**: Selected parents
- **Methods**: Tournament, roulette wheel, rank selection, etc.

### Survivor Selection  
- **Purpose**: Choose next generation from parents + offspring
- **Input**: Parent population + Offspring population
- **Output**: New population
- **Methods**: Generational, (μ+λ), (μ,λ), tournament, etc.

## Survivor Selection Strategies

### 1. Generational Replacement (Default)

**Description**: Replace entire population with offspring, optionally keeping elite parents

**Notation**: Generational with elitism

**How it works**:
1. Generate `population_size` offspring
2. Keep `elitism_count` best parents
3. Fill remaining slots with best offspring

**Advantages**:
- Simple to implement
- Fast population turnover
- Elitism prevents loss of best solutions

**Disadvantages**:
- Can lose diversity quickly
- Good non-elite solutions discarded

**Best for**:
- Problems where rapid convergence is acceptable
- When combined with high mutation to maintain diversity

**Parameters**:
```python
solver = TSPSolver(
    ...,
    survivor_selection='generational',
    elitism_count=5  # Keep 5 best parents
)
```

---

### 2. (μ + λ) Selection

**Description**: Best μ individuals from combined parents (μ) and offspring (λ)

**Notation**: (μ+λ) or "plus" selection

**How it works**:
1. Generate λ offspring (can be any number)
2. Combine all μ parents + λ offspring
3. Select best μ individuals for next generation

**Advantages**:
- Elitist (best solutions never lost)
- Preserves good parents
- Flexible offspring size

**Disadvantages**:
- Can lead to stagnation
- Older individuals may dominate
- Slower diversity loss than generational

**Best for**:
- Steady-state evolution
- When you want guaranteed improvement
- Problems where elitism is critical

**Parameters**:
```python
solver = TSPSolver(
    ...,
    population_size=100,        # μ
    offspring_size=150,         # λ
    survivor_selection='mu_plus_lambda'
)
```

**Common configurations**:
- λ = μ (equal parents and offspring)
- λ = 2μ (twice as many offspring)
- λ = 7μ (ES standard for mutation-based search)

---

### 3. (μ , λ) Selection

**Description**: Best μ individuals from offspring λ only (parents die)

**Notation**: (μ,λ) or "comma" selection

**Constraint**: Requires λ ≥ μ

**How it works**:
1. Generate λ offspring (where λ ≥ μ)
2. Select best μ offspring for next generation
3. Discard ALL parents

**Advantages**:
- Forces exploration (no stagnation)
- Better for dynamic problems
- Prevents aging (all parents replaced each generation)

**Disadvantages**:
- Non-elitist (can lose best solution!)
- Requires more offspring
- Can be wasteful

**Best for**:
- Dynamic optimization problems
- When avoiding premature convergence is critical
- Evolution Strategies (ES)

**Parameters**:
```python
solver = TSPSolver(
    ...,
    population_size=100,        # μ
    offspring_size=200,         # λ (must be ≥ μ)
    survivor_selection='mu_comma_lambda'
)
```

**Note**: Often use λ = 7μ in Evolution Strategies

---

### 4. Tournament Survivor Selection

**Description**: Random tournaments between parents and offspring

**How it works**:
1. Generate offspring
2. For each slot in new population:
   - Randomly select 2 individuals from combined pool
   - Choose the better one

**Advantages**:
- Balanced selection pressure
- Stochastic (maintains diversity)
- Works with any offspring size

**Disadvantages**:
- No guaranteed elitism
- Less predictable than (μ+λ)

**Best for**:
- When you want stochastic survivor selection
- Balancing exploration and exploitation
- Maintaining population diversity

**Parameters**:
```python
solver = TSPSolver(
    ...,
    survivor_selection='tournament'
)
```

---

## Comparison Table

| Strategy | Elitist? | Parents in Selection? | Offspring Size | Diversity | Convergence Speed |
|----------|----------|---------------------|----------------|-----------|------------------|
| Generational (elitism) | Yes | Elite only | = μ | Low | Fast |
| (μ+λ) | Yes | All | Any | Medium | Medium |
| (μ,λ) | No | None | ≥ μ | High | Slow |
| Tournament | No | All | Any | Medium-High | Medium |

## When to Use Each Strategy

### Use **Generational** when:
- You want fast convergence
- Elitism is important
- Problem is static
- You have limited computational budget

### Use **(μ+λ)** when:
- You need guaranteed non-degradation
- Problem requires careful preservation of good solutions
- You want steady-state evolution
- TSP, routing problems, structural optimization

### Use **(μ,λ)** when:
- Problem is dynamic or changing
- You want to avoid premature convergence
- Diversity is more important than convergence speed
- Evolution Strategies applications
- You can afford larger offspring populations

### Use **Tournament** when:
- You want balanced selection pressure
- Diversity maintenance is important
- You want a middle ground between generational and (μ+λ)

## Practical Examples

### Example 1: TSP with (μ+λ)

```python
from TSP.tsp_solver import TSPSolver, load_qatar_dataset

cities, distance_matrix = load_qatar_dataset()

solver = TSPSolver(
    cities=cities,
    distance_matrix=distance_matrix,
    population_size=100,           # μ = 100
    offspring_size=200,            # λ = 200 (generate 2x offspring)
    generations=1000,
    survivor_selection='mu_plus_lambda',
    elitism_count=0  # Not needed, (μ+λ) is inherently elitist
)

best_tour = solver.evolve()
```

### Example 2: Exam Scheduling with (μ,λ)

```python
from ExamScheduling.exam_scheduler import ExamSchedulingSolver, generate_sample_data

exams, students, timeslots, constraints = generate_sample_data()

solver = ExamSchedulingSolver(
    exams=exams,
    students=students,
    timeslots=timeslots,
    constraints=constraints,
    population_size=100,           # μ = 100
    offspring_size=700,            # λ = 700 (7μ, ES standard)
    generations=500,
    mutation_rate=0.3,             # Higher mutation for (μ,λ)
    survivor_selection='mu_comma_lambda'
)

best_schedule = solver.evolve()
```

### Example 3: Comparing Strategies

```python
strategies = {
    'Generational': {
        'survivor_selection': 'generational',
        'elitism_count': 5,
        'offspring_size': 100
    },
    '(100+200)': {
        'survivor_selection': 'mu_plus_lambda',
        'offspring_size': 200
    },
    '(100,700)': {
        'survivor_selection': 'mu_comma_lambda',
        'offspring_size': 700,
        'mutation_rate': 0.25  # Higher for comma selection
    },
    'Tournament': {
        'survivor_selection': 'tournament',
        'offspring_size': 150
    }
}

results = {}
for name, params in strategies.items():
    solver = TSPSolver(
        cities=cities,
        distance_matrix=distance_matrix,
        population_size=100,
        generations=500,
        **params
    )
    solver.evolve(verbose=False)
    results[name] = solver.best_fitness

# Compare results
for name, fitness in results.items():
    print(f"{name}: {fitness:.2f}")
```

## Theory Behind Selection Pressure

### Selection Pressure
How strongly the algorithm favors better solutions

**High Selection Pressure** (fast convergence, low diversity):
- Generational with small elitism
- (μ+λ) with small λ

**Medium Selection Pressure**:
- (μ+λ) with λ = 2μ to 4μ
- Tournament survivor selection

**Low Selection Pressure** (slow convergence, high diversity):
- (μ,λ) with large λ
- Generational with high mutation

## Common Mistakes

### ❌ Mistake 1: No survivor selection
```python
# Just replacing population without considering fitness
population = offspring  # Wrong!
```

### ❌ Mistake 2: Confusing parent and survivor selection
```python
# Using same mechanism for both
parents = tournament_selection(population)
next_gen = tournament_selection(offspring)  # These are different!
```

### ❌ Mistake 3: (μ,λ) with λ < μ
```python
# This will crash!
solver = TSPSolver(
    population_size=100,
    offspring_size=50,  # Error: need offspring_size >= population_size
    survivor_selection='mu_comma_lambda'
)
```

### ❌ Mistake 4: Forgetting elitism is built into (μ+λ)
```python
# Redundant elitism parameter
solver = TSPSolver(
    survivor_selection='mu_plus_lambda',
    elitism_count=5  # Not needed, (μ+λ) already preserves best
)
```

## Advanced Considerations

### Adaptive Selection Pressure

Adjust strategy during run:
```python
def adaptive_survivor_selection(generation, max_generations):
    if generation < max_generations * 0.3:
        return 'mu_comma_lambda'  # Explore early
    elif generation < max_generations * 0.7:
        return 'mu_plus_lambda'   # Balance middle
    else:
        return 'generational'     # Exploit late
```

### Age-Based Selection

Add age tracking:
```python
# Penalize old individuals in (μ+λ)
adjusted_fitness = fitness + age_penalty * individual.age
```

### Multi-Objective Survivor Selection

For multi-objective problems, use Pareto dominance:
```python
# Select non-dominated individuals
next_gen = pareto_front(parents + offspring)
```

## Summary

**Key Takeaways**:

1. **Survivor selection ≠ Parent selection** - They serve different purposes
2. **Different strategies have different trade-offs**
3. **Choose based on your problem**:
   - Static problems → Generational or (μ+λ)
   - Dynamic problems → (μ,λ)
   - Balanced approach → Tournament
4. **(μ+λ) is generally safer** - Guarantees no degradation
5. **(μ,λ) explores more** - Better for avoiding local optima
6. **Always consider selection pressure** - Too high = premature convergence, too low = slow progress

## References

- Eiben & Smith (2015): Introduction to Evolutionary Computing, Chapter 6
- Back et al. (1997): Handbook of Evolutionary Computation
- Schwefel (1995): Evolution and Optimum Seeking (for ES and (μ,λ))

---

**Your implementation now includes all four strategies!** Try them and see which works best for your problems.
