"""
Travelling Salesman Problem (TSP) Solver using Evolutionary Algorithm
Solves TSP for Qatar dataset (194 cities)
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))

from ea_framework import EAFramework
import matplotlib.pyplot as plt
import urllib.request
import re


class TSPSolver(EAFramework):
    """TSP solver using Evolutionary Algorithm"""
    
    def __init__(self, cities, distance_matrix, **kwargs):
        """
        Initialize TSP solver
        
        Args:
            cities: List of city names/coordinates
            distance_matrix: Distance matrix between cities
            **kwargs: Additional EA parameters
        """
        super().__init__(**kwargs)
        self.cities = cities
        self.distance_matrix = distance_matrix
        self.n_cities = len(cities)
        
    def initialize_population(self):
        """Initialize population with random tours"""
        population = []
        for _ in range(self.population_size):
            # Create random permutation of cities
            tour = np.random.permutation(self.n_cities)
            population.append(tour)
        return population
    
    def calculate_fitness(self, tour):
        """
        Calculate total distance of a tour
        
        Args:
            tour: Array of city indices representing the tour
            
        Returns:
            Total distance (fitness)
        """
        total_distance = 0
        for i in range(len(tour)):
            from_city = tour[i]
            to_city = tour[(i + 1) % len(tour)]
            total_distance += self.distance_matrix[from_city][to_city]
        return total_distance
    
    def crossover(self, parent1, parent2):
        """
        Order Crossover (OX) for TSP
        
        Args:
            parent1, parent2: Parent tours
            
        Returns:
            Two offspring tours
        """
        size = len(parent1)
        
        # Select two random crossover points
        cx_point1, cx_point2 = sorted(np.random.choice(size, 2, replace=False))
        
        # Create offspring
        offspring1 = self._order_crossover_single(parent1, parent2, cx_point1, cx_point2)
        offspring2 = self._order_crossover_single(parent2, parent1, cx_point1, cx_point2)
        
        return offspring1, offspring2
    
    def _order_crossover_single(self, parent1, parent2, start, end):
        """Helper function for order crossover"""
        size = len(parent1)
        offspring = np.full(size, -1)
        
        # Copy segment from parent1
        offspring[start:end] = parent1[start:end]
        
        # Fill remaining positions from parent2
        current_pos = end
        for city in np.concatenate([parent2[end:], parent2[:end]]):
            if city not in offspring:
                offspring[current_pos % size] = city
                current_pos += 1
        
        return offspring
    
    def mutate(self, tour):
        """
        Swap mutation for TSP
        
        Args:
            tour: Tour to mutate
            
        Returns:
            Mutated tour
        """
        tour = tour.copy()
        
        # Choose mutation type randomly
        mutation_type = np.random.choice(['swap', 'inversion', 'scramble'])
        
        if mutation_type == 'swap':
            # Swap two random cities
            idx1, idx2 = np.random.choice(len(tour), 2, replace=False)
            tour[idx1], tour[idx2] = tour[idx2], tour[idx1]
            
        elif mutation_type == 'inversion':
            # Reverse a segment
            idx1, idx2 = sorted(np.random.choice(len(tour), 2, replace=False))
            tour[idx1:idx2+1] = tour[idx1:idx2+1][::-1]
            
        else:  # scramble
            # Scramble a segment
            idx1, idx2 = sorted(np.random.choice(len(tour), 2, replace=False))
            segment = tour[idx1:idx2+1]
            np.random.shuffle(segment)
            tour[idx1:idx2+1] = segment
        
        return tour
    
    def plot_tour(self, tour=None, title="TSP Tour", save_path=None):
        """
        Plot the tour on a map
        
        Args:
            tour: Tour to plot (uses best solution if None)
            title: Plot title
            save_path: Path to save plot
        """
        if tour is None:
            tour = self.best_solution
        
        if tour is None:
            print("No solution to plot yet!")
            return
        
        # Extract coordinates
        coords = np.array(self.cities)
        
        plt.figure(figsize=(12, 10))
        
        # Plot cities
        plt.scatter(coords[:, 0], coords[:, 1], c='red', s=50, zorder=2, alpha=0.6)
        
        # Plot tour
        tour_coords = coords[tour]
        tour_coords = np.vstack([tour_coords, tour_coords[0]])  # Close the loop
        plt.plot(tour_coords[:, 0], tour_coords[:, 1], 'b-', linewidth=1.5, alpha=0.7, zorder=1)
        
        # Mark start city
        plt.scatter(coords[tour[0], 0], coords[tour[0], 1], 
                   c='green', s=200, marker='*', zorder=3, label='Start/End')
        
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.title(f"{title}\nTotal Distance: {self.calculate_fitness(tour):.2f}", 
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Tour plot saved to {save_path}")
        
        plt.show()


def load_qatar_dataset():
    """
    Load Qatar TSP dataset
    Returns city coordinates and distance matrix
    """
    print("Loading Qatar TSP dataset...")
    
    try:
        # Try to download the dataset
        url = "http://www.math.uwaterloo.ca/tsp/world/qa194.tsp"
        with urllib.request.urlopen(url, timeout=10) as response:
            content = response.read().decode('utf-8')
        
        # Parse TSP file
        lines = content.strip().split('\n')
        coords = []
        reading_coords = False
        
        for line in lines:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                reading_coords = True
                continue
            elif line == "EOF":
                break
            
            if reading_coords and line:
                parts = line.split()
                if len(parts) >= 3:
                    x, y = float(parts[1]), float(parts[2])
                    coords.append([x, y])
        
        coords = np.array(coords)
        print(f"Loaded {len(coords)} cities from online dataset")
        
    except Exception as e:
        print(f"Could not download dataset: {e}")
        print("Generating sample Qatar-like dataset...")
        
        # Generate sample data if download fails
        np.random.seed(42)
        n_cities = 194
        
        # Generate coordinates in a rectangular region (roughly Qatar shape)
        coords = []
        for _ in range(n_cities):
            x = np.random.uniform(50.7, 51.7)  # Longitude range
            y = np.random.uniform(24.5, 26.2)  # Latitude range
            coords.append([x, y])
        
        coords = np.array(coords)
        print(f"Generated {len(coords)} sample cities")
    
    # Calculate distance matrix
    n = len(coords)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Euclidean distance
                distance_matrix[i][j] = np.sqrt(
                    (coords[i][0] - coords[j][0])**2 + 
                    (coords[i][1] - coords[j][1])**2
                )
    
    return coords, distance_matrix


def main():
    """Main function to run TSP solver"""
    print("="*60)
    print("TSP Solver using Evolutionary Algorithm")
    print("Qatar Dataset (194 cities)")
    print("="*60)
    
    # Load dataset
    cities, distance_matrix = load_qatar_dataset()
    
    # Create solver with optimized parameters
    solver = TSPSolver(
        cities=cities,
        distance_matrix=distance_matrix,
        population_size=200,
        generations=1000,
        crossover_rate=0.85,
        mutation_rate=0.15,
        elitism_count=5,
        tournament_size=5
    )
    
    print("\nEvolutionary Algorithm Parameters:")
    print(f"  Population Size: {solver.population_size}")
    print(f"  Generations: {solver.generations}")
    print(f"  Crossover Rate: {solver.crossover_rate}")
    print(f"  Mutation Rate: {solver.mutation_rate}")
    print(f"  Elitism Count: {solver.elitism_count}")
    print(f"  Tournament Size: {solver.tournament_size}")
    print("\nStarting evolution...\n")
    
    # Run evolution
    best_tour = solver.evolve(verbose=True)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    stats = solver.get_statistics()
    print(f"Best Tour Distance: {stats['best_fitness']:.2f}")
    print(f"Initial Best Distance: {solver.best_fitness_history[0]:.2f}")
    print(f"Improvement: {stats['improvement']:.2f} ({stats['improvement_percent']:.2f}%)")
    print(f"Final Average Distance: {stats['final_avg_fitness']:.2f}")
    
    # Plot convergence
    print("\nGenerating convergence plot...")
    solver.plot_convergence(
        title="TSP Convergence - Qatar Dataset (194 cities)",
        save_path="TSP_convergence.png"
    )
    
    # Plot best tour
    print("\nGenerating tour visualization...")
    solver.plot_tour(
        title="Best TSP Tour - Qatar Dataset",
        save_path="TSP_best_tour.png"
    )
    
    # Save results to file
    with open("TSP_results.txt", "w") as f:
        f.write("TSP SOLVER RESULTS - Qatar Dataset\n")
        f.write("="*60 + "\n\n")
        f.write("Algorithm Parameters:\n")
        f.write(f"  Population Size: {solver.population_size}\n")
        f.write(f"  Generations: {solver.generations}\n")
        f.write(f"  Crossover Rate: {solver.crossover_rate}\n")
        f.write(f"  Mutation Rate: {solver.mutation_rate}\n")
        f.write(f"  Elitism Count: {solver.elitism_count}\n")
        f.write(f"  Tournament Size: {solver.tournament_size}\n\n")
        f.write("Results:\n")
        f.write(f"  Best Tour Distance: {stats['best_fitness']:.2f}\n")
        f.write(f"  Initial Best Distance: {solver.best_fitness_history[0]:.2f}\n")
        f.write(f"  Improvement: {stats['improvement']:.2f} ({stats['improvement_percent']:.2f}%)\n")
        f.write(f"  Final Average Distance: {stats['final_avg_fitness']:.2f}\n\n")
        f.write("Best Tour (city indices):\n")
        f.write(str(best_tour.tolist()))
    
    print("\nResults saved to TSP_results.txt")
    print("\nDone!")


if __name__ == "__main__":
    main()
