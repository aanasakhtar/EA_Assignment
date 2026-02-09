# Frequently Asked Questions (FAQ)

## General Questions

### Q1: What is an Evolutionary Algorithm (EA)?
**A:** An Evolutionary Algorithm is a population-based optimization method inspired by biological evolution. It uses mechanisms like selection, crossover, and mutation to iteratively improve solutions to a problem.

### Q2: Why use EA for TSP and Exam Scheduling?
**A:** Both problems are NP-hard, meaning exact solutions are computationally infeasible for large instances. EA provides good approximate solutions in reasonable time.

### Q3: How do I know if my EA is working correctly?
**A:** Check for:
- Decreasing best fitness over generations
- Convergence (fitness improvement slowing down)
- Reasonable final solution quality
- No runtime errors

---

## Installation Issues

### Q4: "Module not found" error
**Problem:** `ModuleNotFoundError: No module named 'numpy'`

**Solution:**
```bash
pip install numpy matplotlib
```

Or with conda:
```bash
conda install numpy matplotlib
```

### Q5: "Permission denied" when running scripts
**Problem:** `Permission denied: ./run_all.py`

**Solution:**
```bash
python run_all.py
```
(Don't use `./` on Windows, use `python` explicitly)

---

## TSP-Specific Questions

### Q6: TSP not converging/improving
**Problem:** Best fitness plateaus early or doesn't improve

**Solutions:**
1. Increase population size (try 300-500)
2. Increase generations (try 1500-3000)
3. Increase mutation rate (try 0.2-0.3)
4. Check if dataset loaded correctly

### Q7: TSP tour has many crossing edges
**Problem:** Visual tour looks messy with many intersections

**Explanation:** This is normal for EA. Perfect solutions are rare. The distance is more important than visual appearance.

**Improvement strategies:**
1. Run for more generations
2. Increase population size
3. Try multiple runs and keep best
4. Implement 2-opt local search (advanced)

### Q8: Cannot download Qatar dataset
**Problem:** Network error or timeout when downloading

**Solution:** The program automatically generates sample data if download fails. This is fine for learning purposes.

To manually download:
1. Visit http://www.math.uwaterloo.ca/tsp/world/qa194.tsp
2. Save as `qa194.tsp` in TSP directory
3. Modify code to load from file

### Q9: What's a good TSP distance?
**A:** Depends on dataset scale. For sample data, look for:
- Consistent improvement (30-60%)
- Convergence within 500-1000 generations
- Relative comparison matters more than absolute value

---

## Exam Scheduling Questions

### Q10: Hard constraints not satisfied
**Problem:** Final schedule still has exam conflicts

**Solutions:**
1. Check repair mechanism is working
2. Increase hard constraint penalty (try 50000)
3. Run for more generations
4. Reduce mutation rate to preserve valid solutions

### Q11: Poor soft constraint performance
**Problem:** Too many consecutive exams or imbalanced distribution

**Solutions:**
1. Adjust soft constraint weights
2. Increase mutation rate for more exploration
3. Increase population diversity
4. Run for more generations

### Q12: How many timeslots should I use?
**A:** Rule of thumb:
- Minimum: `n_exams / avg_room_capacity`
- Recommended: `n_exams / 5` to `n_exams / 8`
- Too few: Hard to satisfy constraints
- Too many: Wastes resources, poor balance

### Q13: Real Purdue dataset not available
**Problem:** Cannot access institutional data

**Solution:** The sample data generator creates realistic scenarios:
- Adjust parameters to match your needs
- Use for learning and algorithm development
- Results are still valid for assignment

---

## Parameter Tuning

### Q14: How to choose population size?
**Guidelines:**
- Small problems (<50 variables): 50-100
- Medium problems (50-200 variables): 100-200
- Large problems (>200 variables): 200-500
- Trade-off: Larger = better solutions but slower

### Q15: How to choose number of generations?
**Guidelines:**
- Run until convergence (fitness plateaus)
- Typical: 500-2000 generations
- Monitor: If still improving at end, add more
- Early stopping: If no improvement for 100 generations

### Q16: How to balance crossover vs mutation?
**Typical ranges:**
- Crossover rate: 0.7-0.9 (high)
- Mutation rate: 0.05-0.2 (low to medium)

**Reasoning:**
- High crossover: Exploit good solutions
- Low mutation: Maintain exploration
- Constraint problems: Higher mutation (0.2-0.3)

### Q17: Tournament size impact?
**Guidelines:**
- Small (2-3): More diversity, slower convergence
- Medium (5-7): Balanced
- Large (10+): Faster convergence, risk premature convergence

**Recommendation:** Start with 5, adjust based on results

---

## Performance Issues

### Q18: Code running too slowly
**Solutions:**
1. Reduce population size
2. Reduce generations
3. Optimize fitness calculation (use numpy)
4. Use smaller test dataset first
5. Profile code to find bottlenecks

### Q19: Out of memory error
**Problem:** `MemoryError` during execution

**Solutions:**
1. Reduce population size
2. Reduce problem size (fewer cities/exams)
3. Close other applications
4. Use 64-bit Python

### Q20: How to speed up convergence?
**Strategies:**
1. Increase elitism count
2. Increase tournament size
3. Use adaptive mutation rates
4. Implement local search (hybrid EA)
5. Better initial population (greedy construction)

---

## Results and Analysis

### Q21: How to compare different configurations?
**Use the comparison script:**
```bash
python compare_parameters.py
```

Or manually:
1. Run same problem with different parameters
2. Do multiple runs (5-10) per configuration
3. Compare mean and variance of results
4. Use statistical tests if formal comparison needed

### Q22: What if results vary between runs?
**A:** This is normal! EA is stochastic.

**Best practices:**
1. Run multiple times (3-10 runs)
2. Report mean and standard deviation
3. Show best, average, and worst results
4. Discuss reliability in your analysis

### Q23: How to know if solution is "good enough"?
**Criteria:**
1. **TSP:**
   - Within 5-15% of known optimal (if available)
   - Consistent across multiple runs
   - Reasonable looking tour

2. **Exam Scheduling:**
   - Zero hard constraint violations (mandatory)
   - Soft constraints optimized (>50% improvement)
   - Practical feasibility

### Q24: Should I use the best ever or final population best?
**A:** Best ever (tracked by `self.best_solution`)

**Reasoning:**
- Elitism preserves it in population
- Can't get worse over time
- More reliable for reporting

---

## Advanced Topics

### Q25: How to implement adaptive parameters?
**Example - Adaptive mutation:**
```python
def adaptive_mutation_rate(self, generation):
    # High early, low later
    return 0.3 * (1 - generation / self.generations) + 0.05
```

Apply in evolve loop:
```python
self.mutation_rate = adaptive_mutation_rate(generation)
```

### Q26: Can I use different selection methods?
**Yes! Examples:**

**Roulette Wheel:**
```python
def roulette_selection(self, population, fitnesses):
    # Convert to maximization (invert fitness)
    max_fit = max(fitnesses)
    probs = [(max_fit - f + 1) for f in fitnesses]
    probs = np.array(probs) / sum(probs)
    return population[np.random.choice(len(population), p=probs)]
```

**Rank Selection:**
```python
def rank_selection(self, population, fitnesses):
    sorted_indices = np.argsort(fitnesses)
    ranks = np.arange(len(population), 0, -1)
    probs = ranks / ranks.sum()
    return population[np.random.choice(sorted_indices, p=probs)]
```

### Q27: How to implement multi-objective optimization?
**For exam scheduling with multiple objectives:**

```python
def calculate_fitness_multi(self, schedule):
    objectives = {
        'conflicts': self._calculate_hard_constraints(schedule),
        'spread': self._calculate_spread_penalty(schedule),
        'balance': self._calculate_balance_penalty(schedule),
    }
    
    # Weighted sum
    fitness = (objectives['conflicts'] * 10000 + 
               objectives['spread'] * 100 + 
               objectives['balance'] * 50)
    
    return fitness, objectives
```

Then track all objectives for analysis.

---

## Debugging

### Q28: How to debug fitness calculation?
**Add debug prints:**
```python
def calculate_fitness(self, individual):
    fitness = self._compute_fitness(individual)
    print(f"Individual: {individual[:5]}... Fitness: {fitness}")
    return fitness
```

**Verify with known cases:**
```python
# Test with known solution
test_solution = [...known good solution...]
print(f"Known solution fitness: {solver.calculate_fitness(test_solution)}")
```

### Q29: EA not finding feasible solutions
**For constrained problems:**

1. Check repair mechanism works:
```python
# Test repair
invalid = [... create invalid solution ...]
repaired = solver._repair_conflicts(invalid)
violations = solver._calculate_hard_constraints(repaired)
print(f"Violations after repair: {violations}")
```

2. Ensure initial population is valid
3. Increase constraint penalties
4. Verify constraint checking logic

### Q30: Visualizations not showing
**Solutions:**
1. Check matplotlib is installed
2. If using SSH/remote: Save plots instead of showing
3. Try different backend:
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

---

## Assignment-Specific

### Q31: What should I include in my report?
**Essential sections:**
1. Problem description
2. EA design (representation, operators, fitness)
3. Parameter settings with justification
4. Results with graphs
5. Analysis and discussion
6. Conclusions

See `REPORT_TEMPLATE.md` for detailed template.

### Q32: How much analysis is expected?
**Depth depends on assignment requirements, but include:**
- Convergence analysis
- Parameter sensitivity discussion
- Comparison with baselines (if applicable)
- Discussion of strengths and limitations
- Multiple runs for reliability

### Q33: Can I modify the code?
**Yes! Encouraged modifications:**
- Different operators (crossover/mutation)
- New selection schemes
- Adaptive parameters
- Hybrid approaches (EA + local search)
- Better visualization

**Document all changes in your report!**

---

## Getting Help

### Q34: Code isn't working, what should I do?
**Debugging checklist:**
1. Read error message carefully
2. Check file paths and imports
3. Verify input data format
4. Add print statements to trace execution
5. Test with smaller problem size
6. Check requirements are installed
7. Review relevant code sections

### Q35: Results don't match expected behavior
**Verification steps:**
1. Run provided tests: `python test_framework.py`
2. Check parameter values are reasonable
3. Verify fitness function logic
4. Compare with baseline (random solutions)
5. Try multiple random seeds

### Q36: Where to find more information?
**Resources:**
1. Code comments and docstrings
2. README.md - comprehensive overview
3. QUICKSTART.md - basic usage
4. REPORT_TEMPLATE.md - analysis guide
5. Online EA textbooks and tutorials
6. Original problem datasets and papers

---

## Tips for Success

### Best Practices

1. **Start simple:** Use default parameters first
2. **Test incrementally:** Verify each component works
3. **Document everything:** Parameters, changes, results
4. **Run multiple times:** EA is stochastic
5. **Analyze convergence:** Don't just report final values
6. **Compare variants:** Try different operators/parameters
7. **Visualize results:** Graphs are more informative than numbers
8. **Understand the theory:** Know why EA works, not just how

### Common Mistakes to Avoid

1. **Too few runs:** Single run isn't reliable
2. **Ignoring convergence:** Running too long wastes time
3. **Wrong fitness direction:** Minimizing when should maximize
4. **Breaking constraints:** Invalid solutions in population
5. **No diversity:** Population converges to local optimum
6. **Poor parameter choices:** Extreme values rarely work
7. **Not validating results:** Trust but verify
8. **Inadequate analysis:** Just showing numbers without interpretation

---

**Still have questions? Check the code comments or experiment!**
