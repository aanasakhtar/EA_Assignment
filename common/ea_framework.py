"""
Base Evolutionary Algorithm Framework
This module provides the core EA implementation structure
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import matplotlib.pyplot as plt


class EAFramework(ABC):
    """Abstract base class for Evolutionary Algorithms"""
    
    def __init__(self, population_size=100, generations=500, 
                 crossover_rate=0.8, mutation_rate=0.1,
                 elitism_count=2, tournament_size=3):
        """
        Initialize EA parameters
        
        Args:
            population_size: Number of individuals in population
            generations: Number of generations to evolve
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elitism_count: Number of best individuals to preserve
            tournament_size: Size of tournament for selection
        """
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
        
        # Statistics tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.worst_fitness_history = []
        self.best_solution = None
        self.best_fitness = float('inf')
        
    @abstractmethod
    def initialize_population(self) -> List[Any]:
        """Initialize the population with random individuals"""
        pass
    
    @abstractmethod
    def calculate_fitness(self, individual: Any) -> float:
        """Calculate fitness of an individual (lower is better)"""
        pass
    
    @abstractmethod
    def crossover(self, parent1: Any, parent2: Any) -> Tuple[Any, Any]:
        """Perform crossover between two parents"""
        pass
    
    @abstractmethod
    def mutate(self, individual: Any) -> Any:
        """Perform mutation on an individual"""
        pass
    
    def tournament_selection(self, population: List[Any], 
                           fitnesses: List[float]) -> Any:
        """
        Select an individual using tournament selection
        
        Args:
            population: Current population
            fitnesses: Fitness values for population
            
        Returns:
            Selected individual
        """
        tournament_indices = np.random.choice(
            len(population), self.tournament_size, replace=False
        )
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitnesses)]
        return population[winner_idx]
    
    def evolve(self, verbose=True):
        """
        Main evolution loop
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Best solution found
        """
        # Initialize population
        population = self.initialize_population()
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitnesses = [self.calculate_fitness(ind) for ind in population]
            
            # Track statistics
            best_idx = np.argmin(fitnesses)
            self.best_fitness_history.append(fitnesses[best_idx])
            self.avg_fitness_history.append(np.mean(fitnesses))
            self.worst_fitness_history.append(np.max(fitnesses))
            
            # Update best solution
            if fitnesses[best_idx] < self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_solution = population[best_idx].copy()
            
            if verbose and (generation % 50 == 0 or generation == self.generations - 1):
                print(f"Generation {generation}: Best={fitnesses[best_idx]:.2f}, "
                      f"Avg={np.mean(fitnesses):.2f}, "
                      f"Worst={np.max(fitnesses):.2f}")
            
            # Create new population
            new_population = []
            
            # Elitism - preserve best individuals
            elite_indices = np.argsort(fitnesses)[:self.elitism_count]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                # Crossover
                if np.random.random() < self.crossover_rate:
                    offspring1, offspring2 = self.crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if np.random.random() < self.mutation_rate:
                    offspring1 = self.mutate(offspring1)
                if np.random.random() < self.mutation_rate:
                    offspring2 = self.mutate(offspring2)
                
                new_population.append(offspring1)
                if len(new_population) < self.population_size:
                    new_population.append(offspring2)
            
            population = new_population
        
        return self.best_solution
    
    def plot_convergence(self, title="EA Convergence", save_path=None):
        """
        Plot convergence graph
        
        Args:
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(12, 6))
        
        generations = range(len(self.best_fitness_history))
        
        plt.plot(generations, self.best_fitness_history, 
                label='Best Fitness', linewidth=2, color='green')
        plt.plot(generations, self.avg_fitness_history, 
                label='Average Fitness', linewidth=2, color='blue', alpha=0.7)
        plt.plot(generations, self.worst_fitness_history, 
                label='Worst Fitness', linewidth=2, color='red', alpha=0.5)
        
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Fitness (Lower is Better)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
    def get_statistics(self):
        """Return evolution statistics"""
        return {
            'best_fitness': self.best_fitness,
            'final_avg_fitness': self.avg_fitness_history[-1],
            'improvement': self.best_fitness_history[0] - self.best_fitness,
            'improvement_percent': ((self.best_fitness_history[0] - self.best_fitness) / 
                                   self.best_fitness_history[0] * 100)
        }
