import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import matplotlib.pyplot as plt


class EAFramework(ABC):
    """Abstract base class for Evolutionary Algorithms"""
    
    def __init__(self, population_size=100, generations=500, 
                 crossover_rate=0.8, mutation_rate=0.1,
                 elitism_count=2, tournament_size=3,
                 survivor_selection='generational', offspring_size=None,
                 parent_selection='tournament', enable_hall_of_fame=True):
        """
        Initialize EA parameters
        
        Args:
            population_size: Number of individuals in population
            generations: Number of generations to evolve
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elitism_count: Number of best individuals to preserve
            tournament_size: Size of tournament for selection
            survivor_selection: Strategy for survivor selection
                - 'generational': Replace entire population (with elitism)
                - 'mu_plus_lambda': Best from parents + offspring
                - 'mu_comma_lambda': Best from offspring only
                - 'tournament': Tournament between parents and offspring
            offspring_size: Number of offspring to generate (default: population_size)
            parent_selection: Strategy for parent selection
                - 'tournament': Tournament selection
                - 'fitness_proportional': Roulette wheel selection (Holland)
                - 'rank': Rank-based selection
                - 'random': Random selection (preserves diversity, ignores fitness)
            enable_hall_of_fame: Whether to maintain a hall of fame archive of best individuals
        """
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
        self.survivor_selection = survivor_selection
        self.offspring_size = offspring_size or population_size
        self.parent_selection = parent_selection
        self.enable_hall_of_fame = enable_hall_of_fame
        
        # Statistics tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.worst_fitness_history = []
        self.best_solution = None
        self.best_fitness = float('inf')
        
        # Hall of Fame: archive of best individuals found during evolution
        self.hall_of_fame = []  # List of (individual, fitness) tuples
        self.hall_of_fame_fitness_history = []  # Best fitness added to HoF each generation
        
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
    
    def fitness_proportional_selection(self, population: List[Any],
                                      fitnesses: List[float]) -> Any:
        """
        Fitness Proportional Selection (Roulette Wheel Selection)
        
        Proposed by Holland, this method biases selection towards the most fit
        individuals by creating a probability distribution proportional to fitness.
        
        Characteristics:
        - High selective pressure when fitness variance is high
        - Risk of premature convergence (strong individuals dominate)
        - Low selective pressure when fitnesses are similar (late in run)
        - Selection probability ∝ fitness
        
        For MINIMIZATION problems, we need to invert the fitness values.
        
        Args:
            population: Current population
            fitnesses: Fitness values for population
            
        Returns:
            Selected individual
        """
        # Convert to numpy array for easier manipulation
        fitnesses_array = np.array(fitnesses)
        
        # For minimization: invert fitness values
        # We want lower fitness to have higher selection probability
        # Method: Use (max_fitness - fitness + epsilon) to avoid division by zero
        max_fitness = np.max(fitnesses_array)
        epsilon = 1e-10  # Small value to avoid zero probabilities
        
        # Inverted fitness: better solutions (lower fitness) get higher values
        inverted_fitness = max_fitness - fitnesses_array + epsilon
        
        # Handle case where all fitnesses are the same
        if np.all(inverted_fitness == inverted_fitness[0]):
            # Uniform selection if all fitnesses equal
            selected_idx = np.random.randint(0, len(population))
            return population[selected_idx]
        
        # Calculate selection probabilities (proportional to inverted fitness)
        total_inverted_fitness = np.sum(inverted_fitness)
        probabilities = inverted_fitness / total_inverted_fitness
        
        # Sample from the distribution (roulette wheel spin)
        selected_idx = np.random.choice(len(population), p=probabilities)
        
        return population[selected_idx]
    
    def rank_selection(self, population: List[Any],
                      fitnesses: List[float]) -> Any:
        """
        Rank-based Selection
        
        Selection probability is based on rank rather than raw fitness values.
        This addresses some issues with fitness proportional selection:
        - More consistent selection pressure throughout the run
        - Prevents premature convergence when fitness variance is high
        - Maintains selection pressure when fitnesses are similar
        
        Args:
            population: Current population
            fitnesses: Fitness values for population
            
        Returns:
            Selected individual
        """
        # Sort indices by fitness (best to worst for minimization)
        sorted_indices = np.argsort(fitnesses)
        
        # Assign ranks: best individual gets highest rank
        n = len(population)
        ranks = np.zeros(n)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = n - rank  # Best gets rank n, worst gets rank 1
        
        # Calculate selection probabilities based on rank
        total_rank = np.sum(ranks)
        probabilities = ranks / total_rank
        
        # Select based on rank probabilities
        selected_idx = np.random.choice(len(population), p=probabilities)
        
        return population[selected_idx]
    
    def random_selection(self, population: List[Any],
                        fitnesses: List[float] = None) -> Any:
        """
        Random Selection
        
        Selects individuals purely by chance, completely ignoring fitness values.
        This method:
        - Returns elements with equal probability regardless of fitness
        - Effectively performs a random walk through the search space
        - Maximally preserves population diversity
        - Useful in combination with separate environmental selection
        - Good for selecting from an optimal set where all are equally viable
        
        Args:
            population: Current population
            fitnesses: Fitness values (ignored)
            
        Returns:
            Randomly selected individual
        """
        # Select a random index with uniform probability
        selected_idx = np.random.randint(0, len(population))
        return population[selected_idx]
    
    def select_parent(self, population: List[Any], fitnesses: List[float]) -> Any:
        """
        Select a parent using the configured parent selection method
        
        Args:
            population: Current population
            fitnesses: Fitness values for population
            
        Returns:
            Selected parent
        """
        if self.parent_selection == 'tournament':
            return self.tournament_selection(population, fitnesses)
        elif self.parent_selection == 'fitness_proportional':
            return self.fitness_proportional_selection(population, fitnesses)
        elif self.parent_selection == 'rank':
            return self.rank_selection(population, fitnesses)
        elif self.parent_selection == 'random':
            return self.random_selection(population, fitnesses)
        else:
            raise ValueError(f"Unknown parent selection method: {self.parent_selection}")
    
    def survivor_selection_generational(self, population, offspring, 
                                       parent_fitnesses, offspring_fitnesses):
        """
        Generational replacement with elitism
        
        Args:
            population: Parent population
            offspring: Offspring population
            parent_fitnesses: Fitness values of parents
            offspring_fitnesses: Fitness values of offspring
            
        Returns:
            New population
        """
        # Combine elite parents with offspring
        new_population = []
        
        if self.elitism_count > 0:
            # Get elite individuals from parents
            elite_indices = np.argsort(parent_fitnesses)[:self.elitism_count]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
        
        # Fill rest with best offspring
        remaining_size = self.population_size - len(new_population)
        if remaining_size > 0:
            best_offspring_indices = np.argsort(offspring_fitnesses)[:remaining_size]
            for idx in best_offspring_indices:
                new_population.append(offspring[idx])
        
        return new_population
    
    def survivor_selection_mu_plus_lambda(self, population, offspring,
                                         parent_fitnesses, offspring_fitnesses):
        """
        (mu+lambda) selection: Best from parents AND offspring
        
        Args:
            population: Parent population (mu)
            offspring: Offspring population (lambda)
            parent_fitnesses: Fitness values of parents
            offspring_fitnesses: Fitness values of offspring
            
        Returns:
            New population of size mu
        """
        # Combine parents and offspring
        combined = population + offspring
        combined_fitnesses = list(parent_fitnesses) + list(offspring_fitnesses)
        
        # Select best mu individuals
        best_indices = np.argsort(combined_fitnesses)[:self.population_size]
        new_population = [combined[i] for i in best_indices]
        
        return new_population
    
    
    def survivor_selection_tournament(self, population, offspring,
                                     parent_fitnesses, offspring_fitnesses):
        """
        Tournament-based survivor selection
        Randomly pair parents with offspring and select better one
        
        Args:
            population: Parent population
            offspring: Offspring population
            parent_fitnesses: Fitness values of parents
            offspring_fitnesses: Fitness values of offspring
            
        Returns:
            New population
        """
        new_population = []
        
        # Ensure we have enough individuals
        combined = population + offspring
        combined_fitnesses = list(parent_fitnesses) + list(offspring_fitnesses)
        
        # Run tournaments
        for _ in range(self.population_size):
            # Select two random individuals
            idx1, idx2 = np.random.choice(len(combined), 2, replace=False)
            
            # Tournament: select better one
            if combined_fitnesses[idx1] <= combined_fitnesses[idx2]:
                new_population.append(combined[idx1].copy())
            else:
                new_population.append(combined[idx2].copy())
        
        return new_population
    
    def perform_survivor_selection(self, population, offspring,
                                   parent_fitnesses, offspring_fitnesses):
        """
        Perform survivor selection based on configured strategy
        
        Args:
            population: Parent population
            offspring: Offspring population  
            parent_fitnesses: Fitness values of parents
            offspring_fitnesses: Fitness values of offspring
            
        Returns:
            New population
        """
        if self.survivor_selection == 'generational':
            return self.survivor_selection_generational(
                population, offspring, parent_fitnesses, offspring_fitnesses
            )
        elif self.survivor_selection == 'mu_plus_lambda':
            return self.survivor_selection_mu_plus_lambda(
                population, offspring, parent_fitnesses, offspring_fitnesses
            )
        elif self.survivor_selection == 'tournament':
            return self.survivor_selection_tournament(
                population, offspring, parent_fitnesses, offspring_fitnesses
            )
        else:
            raise ValueError(f"Unknown survivor selection: {self.survivor_selection}")
    
    def update_hall_of_fame(self, individual: Any, fitness: float):
        """
        Update the hall of fame with a new individual.
        Maintains an archive of the best individuals found during evolution.
        
        Args:
            individual: The individual to potentially add to hall of fame
            fitness: The fitness value of the individual
        """
        if not self.enable_hall_of_fame:
            return
        
        # If hall of fame is empty, add the first individual
        if len(self.hall_of_fame) == 0:
            self.hall_of_fame.append((individual.copy(), fitness))
            self.hall_of_fame_fitness_history.append(fitness)
            return
        
        # Get the best fitness in hall of fame
        best_hof_fitness = min([f for _, f in self.hall_of_fame])
        
        # If new individual is better than worst in hall of fame, add it
        if fitness <= best_hof_fitness:
            # Check if this fitness already exists in hall of fame
            exists = any(f == fitness for _, f in self.hall_of_fame)
            if not exists:
                self.hall_of_fame.append((individual.copy(), fitness))
        
        # Track the best fitness added/updated in this generation
        self.hall_of_fame_fitness_history.append(best_hof_fitness)
    
    def get_best_from_hall_of_fame(self) -> Tuple[Any, float]:
        """
        Get the best individual from the hall of fame.
        Useful at the last generation or when using hall of fame as parent pool.
        
        Returns:
            Tuple of (best_individual, best_fitness) from hall of fame
            Returns (None, float('inf')) if hall of fame is empty
        """
        if not self.hall_of_fame:
            return None, float('inf')
        
        # Find the best individual in hall of fame
        best_individual, best_fitness = min(self.hall_of_fame, key=lambda x: x[1])
        return best_individual, best_fitness
    
    def get_hall_of_fame_size(self) -> int:
        """
        Get the number of unique individuals in the hall of fame.
        
        Returns:
            Size of hall of fame
        """
        return len(self.hall_of_fame)

    
    def evolve(self, verbose=True):
        """
        Main evolution loop with proper survivor selection
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Best solution found
        """
        # Initialize population
        population = self.initialize_population()
        
        for generation in range(self.generations):
            # Evaluate fitness of current population
            fitnesses = [self.calculate_fitness(ind) for ind in population]
            
            # Track statistics
            best_idx = np.argmin(fitnesses)
            self.best_fitness_history.append(fitnesses[best_idx])
            self.avg_fitness_history.append(np.mean(fitnesses))
            self.worst_fitness_history.append(np.max(fitnesses))
            
            # Update best solution ever found
            if fitnesses[best_idx] < self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_solution = population[best_idx].copy()
            
            # Update hall of fame with best individual in this generation
            if self.enable_hall_of_fame:
                self.update_hall_of_fame(population[best_idx], fitnesses[best_idx])
            
            if verbose and (generation % 50 == 0 or generation == self.generations - 1):
                print(f"Generation {generation}: Best={fitnesses[best_idx]:.2f}, "
                      f"Avg={np.mean(fitnesses):.2f}, "
                      f"Worst={np.max(fitnesses):.2f}")
            
            # Generate offspring through parent selection, crossover, and mutation
            offspring = []
            
            while len(offspring) < self.offspring_size:
                # Parent selection using configured method
                parent1 = self.select_parent(population, fitnesses)
                parent2 = self.select_parent(population, fitnesses)
                
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
                
                offspring.append(offspring1)
                if len(offspring) < self.offspring_size:
                    offspring.append(offspring2)
            
            # Evaluate offspring fitness
            offspring_fitnesses = [self.calculate_fitness(ind) for ind in offspring]
            
            # Survivor selection: combine parents and offspring
            population = self.perform_survivor_selection(
                population, offspring, fitnesses, offspring_fitnesses
            )
        
        # Return best solution (prefer hall of fame if enabled)
        if self.enable_hall_of_fame and self.hall_of_fame:
            best_hof_solution, best_hof_fitness = self.get_best_from_hall_of_fame()
            if best_hof_fitness < self.best_fitness:
                return best_hof_solution
        
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
