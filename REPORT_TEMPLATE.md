# ASSIGNMENT REPORT TEMPLATE
## Evolutionary Algorithms for Global Optimization

**Student Name:** [Your Name]  
**Student ID:** [Your ID]  
**Date:** [Submission Date]  
**Course:** Evolutionary Algorithms

---

## Executive Summary

[Write a brief summary of your findings - 150-200 words]

Key findings:
- TSP best tour distance: [X] units
- TSP improvement from initial: [Y]%
- Exam scheduling achieved [0/X] hard constraint violations
- Exam scheduling achieved [Y]% soft constraint satisfaction

---

## Problem 1: Travelling Salesman Problem (TSP)

### 1.1 Problem Description

The Travelling Salesman Problem for Qatar dataset involves finding the shortest route through 194 cities. This is an NP-hard combinatorial optimization problem.

**Dataset Details:**
- Number of cities: 194
- Source: [URL or description]
- Coordinate system: [Geographic/Euclidean]

### 1.2 EA Design

**Chromosome Representation:**
- Type: Permutation encoding
- Length: 194 (number of cities)
- Example: [45, 12, 89, 3, ..., 156]
- Ensures: Each city visited exactly once

**Fitness Function:**
```
fitness(tour) = Σ distance(city[i], city[(i+1) mod n]) for i in 0..n-1
```
- Objective: Minimization
- Range: [Minimum observed] to [Maximum observed]

**Selection Method:**
- Tournament selection with size = [X]
- Rationale: [Explain why tournament selection is appropriate]

**Crossover Operator:**
- Order Crossover (OX)
- Probability: [X]
- Mechanism: [Brief description]
- Why it works: [Explain preservation of order]

**Mutation Operators:**
1. Swap mutation: [Description and probability]
2. Inversion mutation: [Description and probability]
3. Scramble mutation: [Description and probability]

**Elitism:**
- Count: [X] individuals
- Rationale: [Explain benefit]

### 1.3 Parameter Settings

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Population Size | [X] | [Why this size?] |
| Generations | [X] | [Why this number?] |
| Crossover Rate | [X] | [Why this rate?] |
| Mutation Rate | [X] | [Why this rate?] |
| Elitism Count | [X] | [Why this count?] |
| Tournament Size | [X] | [Why this size?] |

### 1.4 Results

**Best Solution Found:**
- Tour distance: [X] units
- Initial best distance: [Y] units
- Absolute improvement: [Z] units
- Percentage improvement: [W]%

**Convergence Analysis:**
- Generations to 90% of final solution: [X]
- Generations to 99% of final solution: [Y]
- Final improvement rate: [Z]% per 100 generations

**Statistical Summary:**
- Mean fitness (final generation): [X]
- Standard deviation (final generation): [Y]
- Best fitness variance across generations: [Z]

### 1.5 Convergence Graph Analysis

[Insert TSP_convergence.png here]

**Observations:**
1. [What patterns do you see in best/average/worst fitness?]
2. [When does the algorithm converge?]
3. [Any signs of premature convergence or stagnation?]
4. [How does population diversity change over time?]

### 1.6 Tour Visualization Analysis

[Insert TSP_best_tour.png here]

**Observations:**
1. [Does the tour look reasonable?]
2. [Are there obvious optimizations that could be made?]
3. [How many crossing edges are there?]
4. [Any clustering patterns?]

### 1.7 Comparison with Baseline

**Random Tour:**
- Average distance: [X]
- Our solution vs random: [Y]% improvement

**Greedy Nearest Neighbor:**
- Distance: [X] (if implemented)
- Our solution vs greedy: [Y]% improvement/difference

### 1.8 Discussion

**What worked well:**
1. [Point 1]
2. [Point 2]
3. [Point 3]

**What could be improved:**
1. [Point 1]
2. [Point 2]
3. [Point 3]

**Impact of parameters:**
- [Discuss which parameters had biggest impact]
- [What would you change for better results?]

---

## Problem 2: Exam Scheduling

### 2.1 Problem Description

The exam scheduling problem involves creating a timetable that satisfies hard constraints (no conflicts) while optimizing soft constraints (student preferences and resource balance).

**Problem Instance:**
- Number of exams: [X]
- Number of students: [Y]
- Number of timeslots: [Z]
- Average exams per student: [W]

### 2.2 EA Design

**Chromosome Representation:**
- Type: Integer vector
- Length: [Number of exams]
- Gene values: Timeslot assignment (0 to [n_timeslots-1])
- Example: [2, 5, 1, 5, 0, ...]
- Interpretation: exam[i] is scheduled in timeslot[value[i]]

**Fitness Function:**
```
fitness = (hard_violations × 10000) + 
          (spread_penalty × 100) + 
          (balance_penalty × 50) + 
          (consecutive_penalty × 200)
```

**Hard Constraints:**
1. No exam conflicts (same student, same timeslot)
   - Weight: 10,000 per violation
   - Must be satisfied: Yes

**Soft Constraints:**
1. Exam spread (weight: 100)
   - Penalizes exams too close together
   - Proximity weights: [List weights]

2. Load balance (weight: 50)
   - Based on variance of exams per timeslot
   - Formula: [Variance calculation]

3. Consecutive exams (weight: 200)
   - Penalizes back-to-back exams
   - Count per student

**Constraint Handling:**
- Repair mechanism: [Description]
- Applied: After crossover and mutation
- Effectiveness: [Analysis]

**Selection Method:**
- Tournament selection, size = [X]

**Crossover Operator:**
- Uniform crossover
- Probability: [X]
- Followed by repair

**Mutation Operator:**
- Random timeslot reassignment
- Probability: [X]
- Number of mutations: [Range]
- Followed by repair

**Elitism:**
- Count: [X]

### 2.3 Parameter Settings

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Population Size | [X] | [Why this size?] |
| Generations | [X] | [Why this number?] |
| Crossover Rate | [X] | [Why this rate?] |
| Mutation Rate | [X] | [Higher than TSP - why?] |
| Elitism Count | [X] | [Why this count?] |
| Tournament Size | [X] | [Why this size?] |

### 2.4 Results

**Best Solution Found:**
- Total fitness (penalty): [X]
- Initial fitness: [Y]
- Improvement: [Z]%

**Hard Constraint Satisfaction:**
- Exam conflicts: [X] (must be 0)
- Status: [✓ All satisfied / ✗ Has violations]

**Soft Constraint Performance:**
- Spread penalty: [X]
- Balance penalty: [Y]
- Consecutive penalty: [Z]

**Distribution Analysis:**
- Exams per timeslot range: [min] to [max]
- Mean: [X]
- Variance: [Y]
- Standard deviation: [Z]

**Student Impact:**
- Students with consecutive exams: [X]
- Percentage: [Y]%
- Maximum consecutive for any student: [Z]

### 2.5 Convergence Analysis

[Insert ExamScheduling_convergence.png here]

**Observations:**
1. [When were hard constraints satisfied?]
2. [How did soft constraints improve?]
3. [Any correlation between constraint types?]
4. [Convergence speed compared to TSP?]

### 2.6 Distribution Analysis

[Insert ExamScheduling_distribution.png here]

**Observations:**
1. [How balanced is the distribution?]
2. [Any timeslots over/under-utilized?]
3. [Does it meet practical requirements?]

### 2.7 Constraint Analysis

**Hard Constraints:**
- Initial violations: [X]
- Final violations: [Y]
- Generations to satisfaction: [Z]

**Soft Constraints Over Time:**
- [Graph or table showing progression]

### 2.8 Discussion

**Effectiveness of approach:**
1. [How well did repair mechanism work?]
2. [Trade-offs between soft constraints?]
3. [Real-world applicability?]

**Challenges encountered:**
1. [Challenge 1]
2. [Challenge 2]
3. [Challenge 3]

**Solutions implemented:**
1. [Solution 1]
2. [Solution 2]
3. [Solution 3]

---

## Comparative Analysis

### 3.1 EA Behavior Across Problems

**Convergence patterns:**
- TSP: [Description]
- Exam Scheduling: [Description]
- Comparison: [Why different?]

**Parameter sensitivity:**
| Parameter | TSP Impact | Exam Impact | Explanation |
|-----------|-----------|-------------|-------------|
| Population Size | [High/Med/Low] | [High/Med/Low] | [Why?] |
| Mutation Rate | [High/Med/Low] | [High/Med/Low] | [Why?] |
| [Others] | ... | ... | ... |

### 3.2 Computational Complexity

**Runtime comparison:**
- TSP: [X] seconds per generation
- Exam Scheduling: [Y] seconds per generation
- Ratio: [Z]
- Explanation: [Why the difference?]

**Scalability:**
- How would runtime scale with problem size?
- Bottlenecks identified:
  1. [Bottleneck 1]
  2. [Bottleneck 2]

### 3.3 Solution Quality

**Optimality:**
- TSP: [% from known optimal if available]
- Exam Scheduling: [Discussion - no known optimal]

**Consistency:**
- Multiple run variance: [Analysis]
- Reliability: [High/Medium/Low]

---

## Lessons Learned

### 4.1 Parameter Tuning Insights

**Most impactful parameters:**
1. [Parameter 1]: [Impact and explanation]
2. [Parameter 2]: [Impact and explanation]
3. [Parameter 3]: [Impact and explanation]

**Surprising findings:**
- [Finding 1]
- [Finding 2]

### 4.2 Algorithm Design Insights

**What worked well:**
1. [Success 1]
2. [Success 2]
3. [Success 3]

**What didn't work:**
1. [Failure 1 and lesson learned]
2. [Failure 2 and lesson learned]

### 4.3 Real-world Applicability

**TSP applications:**
- [Real-world scenario 1]
- [Real-world scenario 2]

**Exam Scheduling applications:**
- [Beyond academia]
- [Related scheduling problems]

---

## Conclusions

### 5.1 Summary of Achievements

1. **TSP:**
   - Achieved [X]% improvement over random
   - Solution quality: [Assessment]
   - Convergence: [Assessment]

2. **Exam Scheduling:**
   - [Hard constraint success]
   - [Soft constraint performance]
   - [Practical usability]

### 5.2 Challenges in Global Optimization

**Problem complexity:**
- [Discussion of NP-hardness]
- [No guarantee of global optimum]

**Trade-offs:**
- Exploration vs exploitation
- Solution quality vs computation time
- Simplicity vs sophistication

### 5.3 Future Work

**Potential improvements:**
1. [Improvement 1]
2. [Improvement 2]
3. [Improvement 3]

**Advanced techniques to explore:**
1. [Technique 1]
2. [Technique 2]
3. [Technique 3]

### 5.4 Final Thoughts

[Your concluding remarks about EA for global optimization - what you learned, what impressed you, what frustrated you, how you would apply this knowledge]

---

## References

1. [Your references - papers, books, websites]
2. [Include any sources you consulted]
3. [Cite the TSP dataset source]
4. [Cite the exam scheduling benchmark if used]

---

## Appendix A: Code Structure

**File organization:**
```
[List your files and their purposes]
```

**Key design decisions:**
1. [Decision 1 and rationale]
2. [Decision 2 and rationale]

---

## Appendix B: Experimental Data

**Additional runs:**
[Include data from multiple runs if you performed them]

**Parameter sweep results:**
[If you tested multiple parameter combinations]

---

## Appendix C: Visualizations

[Include any additional graphs or visualizations you generated]

---

**Declaration:**
I confirm that this work is my own and that I have properly cited all sources used.

Signature: ___________________  
Date: ___________________
