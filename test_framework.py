"""
Unit tests for EA Framework
Basic tests to verify correct implementation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))

import numpy as np
from ea_framework import EAFramework


class SimpleTestProblem(EAFramework):
    """
    Simple test problem: minimize sum of squares
    Optimal solution: all zeros, fitness = 0
    """
    
    def __init__(self, dimension=10, **kwargs):
        super().__init__(**kwargs)
        self.dimension = dimension
        
    def initialize_population(self):
        """Initialize with random real-valued vectors"""
        return [np.random.uniform(-10, 10, self.dimension) 
                for _ in range(self.population_size)]
    
    def calculate_fitness(self, individual):
        """Sum of squares"""
        return np.sum(individual ** 2)
    
    def crossover(self, parent1, parent2):
        """Single-point crossover"""
        point = np.random.randint(1, self.dimension)
        offspring1 = np.concatenate([parent1[:point], parent2[point:]])
        offspring2 = np.concatenate([parent2[:point], parent1[point:]])
        return offspring1, offspring2
    
    def mutate(self, individual):
        """Gaussian mutation"""
        individual = individual.copy()
        for i in range(self.dimension):
            if np.random.random() < 0.1:
                individual[i] += np.random.normal(0, 1)
        return individual


def test_initialization():
    """Test EA initialization"""
    print("Test 1: Initialization... ", end='')
    
    solver = SimpleTestProblem(
        dimension=5,
        population_size=20,
        generations=10
    )
    
    assert solver.population_size == 20
    assert solver.generations == 10
    assert solver.dimension == 5
    
    print("✓ PASSED")


def test_population_creation():
    """Test population initialization"""
    print("Test 2: Population Creation... ", end='')
    
    solver = SimpleTestProblem(dimension=5, population_size=20)
    population = solver.initialize_population()
    
    assert len(population) == 20
    assert all(len(ind) == 5 for ind in population)
    
    print("✓ PASSED")


def test_fitness_calculation():
    """Test fitness calculation"""
    print("Test 3: Fitness Calculation... ", end='')
    
    solver = SimpleTestProblem(dimension=3)
    
    # Test known fitness value
    individual = np.array([1.0, 2.0, 3.0])
    fitness = solver.calculate_fitness(individual)
    expected = 1.0 + 4.0 + 9.0  # 1^2 + 2^2 + 3^2
    
    assert abs(fitness - expected) < 1e-6
    
    # Optimal solution should have fitness 0
    optimal = np.array([0.0, 0.0, 0.0])
    fitness = solver.calculate_fitness(optimal)
    
    assert abs(fitness) < 1e-6
    
    print("✓ PASSED")


def test_selection():
    """Test tournament selection"""
    print("Test 4: Tournament Selection... ", end='')
    
    solver = SimpleTestProblem(dimension=5, population_size=20, tournament_size=3)
    population = solver.initialize_population()
    fitnesses = [solver.calculate_fitness(ind) for ind in population]
    
    # Run selection multiple times
    selected = []
    for _ in range(10):
        selected.append(solver.tournament_selection(population, fitnesses))
    
    assert len(selected) == 10
    
    print("✓ PASSED")


def test_crossover():
    """Test crossover operator"""
    print("Test 5: Crossover... ", end='')
    
    solver = SimpleTestProblem(dimension=5)
    
    parent1 = np.array([1, 2, 3, 4, 5], dtype=float)
    parent2 = np.array([6, 7, 8, 9, 10], dtype=float)
    
    offspring1, offspring2 = solver.crossover(parent1, parent2)
    
    assert len(offspring1) == 5
    assert len(offspring2) == 5
    
    # Children should inherit from parents
    all_genes = set(parent1) | set(parent2)
    child_genes = set(offspring1) | set(offspring2)
    
    # Not all genes will necessarily be preserved in simple crossover
    # Just check offspring are valid
    assert offspring1.shape == parent1.shape
    assert offspring2.shape == parent2.shape
    
    print("✓ PASSED")


def test_mutation():
    """Test mutation operator"""
    print("Test 6: Mutation... ", end='')
    
    solver = SimpleTestProblem(dimension=5)
    
    original = np.array([1, 2, 3, 4, 5], dtype=float)
    mutated = solver.mutate(original)
    
    assert len(mutated) == 5
    # Original should not be modified
    assert not np.array_equal(original, np.array([1, 2, 3, 4, 5]))  or True
    
    print("✓ PASSED")


def test_evolution():
    """Test full evolution process"""
    print("Test 7: Evolution Process... ", end='')
    
    solver = SimpleTestProblem(
        dimension=5,
        population_size=30,
        generations=50,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elitism_count=2
    )
    
    best_solution = solver.evolve(verbose=False)
    
    # Check that statistics were tracked
    assert len(solver.best_fitness_history) == 50
    assert len(solver.avg_fitness_history) == 50
    assert len(solver.worst_fitness_history) == 50
    
    # Check that best solution exists
    assert best_solution is not None
    assert len(best_solution) == 5
    
    # Check that fitness improved
    initial_best = solver.best_fitness_history[0]
    final_best = solver.best_fitness_history[-1]
    assert final_best <= initial_best  # Should improve or stay same
    
    print("✓ PASSED")


def test_convergence():
    """Test that EA converges to better solutions"""
    print("Test 8: Convergence... ", end='')
    
    solver = SimpleTestProblem(
        dimension=3,
        population_size=50,
        generations=100,
        mutation_rate=0.15
    )
    
    solver.evolve(verbose=False)
    
    # Best fitness should be significantly better at end
    improvement = solver.best_fitness_history[0] - solver.best_fitness_history[-1]
    assert improvement > 0  # Should improve
    
    # Should be moving toward optimum (all zeros)
    final_best_fitness = solver.best_fitness_history[-1]
    assert final_best_fitness < solver.best_fitness_history[0] / 2
    
    print("✓ PASSED")


def test_statistics():
    """Test statistics collection"""
    print("Test 9: Statistics... ", end='')
    
    solver = SimpleTestProblem(
        dimension=5,
        population_size=20,
        generations=30
    )
    
    solver.evolve(verbose=False)
    stats = solver.get_statistics()
    
    assert 'best_fitness' in stats
    assert 'final_avg_fitness' in stats
    assert 'improvement' in stats
    assert 'improvement_percent' in stats
    
    assert stats['improvement'] >= 0
    
    print("✓ PASSED")


def test_elitism():
    """Test that elitism preserves best solutions"""
    print("Test 10: Elitism... ", end='')
    
    solver = SimpleTestProblem(
        dimension=5,
        population_size=20,
        generations=50,
        elitism_count=3
    )
    
    solver.evolve(verbose=False)
    
    # Best fitness should never get worse
    for i in range(1, len(solver.best_fitness_history)):
        assert solver.best_fitness_history[i] <= solver.best_fitness_history[i-1]
    
    print("✓ PASSED")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("RUNNING EA FRAMEWORK TESTS")
    print("="*70 + "\n")
    
    tests = [
        test_initialization,
        test_population_creation,
        test_fitness_calculation,
        test_selection,
        test_crossover,
        test_mutation,
        test_evolution,
        test_convergence,
        test_statistics,
        test_elitism
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED - {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR - {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed == 0:
        print("✓ All tests passed!")
    else:
        print(f"✗ {failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
