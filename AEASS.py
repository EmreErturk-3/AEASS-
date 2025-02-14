

import time
import numpy as np
import random
from typing import List, Tuple, Dict
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import defaultdict

class SudokuSolver:
    def __init__(self, sudoku: List[List[int]], population_size: int = 350, local_search_frequency: float = 0.8, 
                 tournament_size: int = 5, crossover_rate: float = 0.7,mutation_rate: float = 1,
                stagnation_threshold: int= 75,local_search_depth:int=30,early_ls_stop=10):
        self.sudoku = np.array(sudoku)
        self.population_size = population_size
        self.fixed_positions = np.where(self.sudoku != 0)
        self.fixed_values = self.sudoku[self.fixed_positions]
        self.best_solution = None
        self.best_fitness = 0
        self.local_search_frequency = local_search_frequency
        self.local_search_depth = local_search_depth
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.stagnation_threshold=stagnation_threshold
        self.mutation_rate = mutation_rate
        self.early_ls_stop=early_ls_stop
        
        
    def generate_individual(self) -> List[List[int]]:
        """Generate a random valid individual where each row contains 1-9"""
        grid = [[0]*9 for _ in range(9)]
        
        # Set fixed numbers
        for i, j in zip(*self.fixed_positions):
            grid[i][j] = self.sudoku[i][j]
            
        # Fill rows with valid permutations
        for row in range(9):
            fixed = set(grid[row][col] for col in range(9) if grid[row][col] != 0)
            numbers = list(set(range(1, 10)) - fixed)
            empty_cols = [col for col in range(9) if grid[row][col] == 0]
            random.shuffle(numbers)
            
            for col, num in zip(empty_cols, numbers):
                grid[row][col] = num
                
        return grid

    def find_duplicates(self, values: List[int]) -> Dict[int, List[int]]:
        """Find duplicate numbers and their positions"""
        value_positions = defaultdict(list)
        for pos, value in enumerate(values):
            value_positions[value].append(pos)
        return {val: pos for val, pos in value_positions.items() if len(pos) > 1}

    def count_errors(self, grid: List[List[int]]) -> Tuple[int, int, int, int, Dict]:
        """
        Count errors and track their locations
        Returns: total_errors, row_errors, col_errors, box_errors, error_locations
        """
        row_errors = 0
        col_errors = 0
        box_errors = 0
        error_locations = {
            'rows': {},
            'columns': {},
            'boxes': {}
        }
        
        # Row errors
        for row in range(9):
            row_vals = [grid[row][col] for col in range(9)]
            duplicates = self.find_duplicates(row_vals)
            if duplicates:
                error_locations['rows'][row] = duplicates
                row_errors += sum(len(pos) - 1 for pos in duplicates.values())
            
        # Column errors    
        for col in range(9):
            col_vals = [grid[row][col] for row in range(9)]
            duplicates = self.find_duplicates(col_vals)
            if duplicates:
                error_locations['columns'][col] = duplicates
                col_errors += sum(len(pos) - 1 for pos in duplicates.values())
            
        # Box errors
        for box_num in range(9):
            box_row, box_col = divmod(box_num, 3)
            box = []
            box_positions = []
            for i in range(3):
                for j in range(3):
                    row = box_row * 3 + i
                    col = box_col * 3 + j
                    box.append(grid[row][col])
                    box_positions.append((row, col))
                    
            duplicates = self.find_duplicates(box)
            if duplicates:
                box_duplicates = {}
                for value, positions in duplicates.items():
                    box_duplicates[value] = [box_positions[pos] for pos in positions]
                error_locations['boxes'][box_num] = box_duplicates
                box_errors += sum(len(pos) - 1 for pos in duplicates.values())
                
        total_errors = row_errors + col_errors + box_errors
        return total_errors, row_errors, col_errors, box_errors, error_locations
    

    def calculate_fitness(self, grid: List[List[int]]) -> float:
        """Calculate fitness score for a grid"""
        total_errors, _, _, _, _ = self.count_errors(grid)
        return 1 / (1 + total_errors)


    def tournament_selection(self, population: List[List[List[int]]]) -> List[List[int]]:
        """Select the best individual from a random tournament"""
        tournament = random.sample(population, self.tournament_size)
        return max(tournament, key=self.calculate_fitness)
    

    def crossover(self, parent1: List[List[int]], parent2: List[List[int]]) -> List[List[int]]:
        """Perform crossover between two parents"""
        child = [row[:] for row in parent1]
        block_start = random.randint(0, 2) * 3
        for i in range(block_start, block_start + 3):
            child[i] = parent2[i][:]
        return child
    
    
    def _swap_in_row(self, grid: List[List[int]]) -> List[List[int]]:
        """Swap two non-fixed numbers in a random row"""
        result = [row[:] for row in grid]
        row = random.randint(0, 8)
        
        # Find non-fixed positions in this row
        available = [j for j in range(9) if (row, j) not in zip(*self.fixed_positions)]
        
        if len(available) >= 2:
            pos1, pos2 = random.sample(available, 2)
            result[row][pos1], result[row][pos2] = result[row][pos2], result[row][pos1]
        
        return result
    
    def _swap_rows(self, grid: List[List[int]]) -> List[List[int]]:
        """Swap two rows within the same 3x3 block"""
        result = [row[:] for row in grid]
        
        # Select a random 3x3 block
        block = random.randint(0, 2)
        
        # Get rows in the selected block (e.g., if block=1, get rows 3,4,5)
        possible_rows = [3*block, 3*block + 1, 3*block + 2]
        
        # Pick 2 rows to swap
        r1, r2 = random.sample(possible_rows, 2)
        result[r1], result[r2] = result[r2][:], result[r1][:]

        return result
    
    def mutate(self, grid: List[List[int]]) -> List[List[int]]:
        """Apply mutation with multiple operators"""
        result = [row[:] for row in grid]
        
        # Multiple mutation operators with different probabilities
        mutation_ops = [
            (self._swap_in_row, 0.7),
            (self._swap_rows, 0.3),
            
        ]
        
        # Apply mutations based on probabilities
        for op, prob in mutation_ops:
            if random.random() < prob:
                result = op(result)
                
        return result
    

        
    def local_search(self, grid: List[List[int]], depth: int = None) -> List[List[int]]:
        """Enhanced local search with variable neighborhood"""
        if depth is None:
            depth = self.local_search_depth
            
        best_grid = [row[:] for row in grid]
        best_fitness = self.calculate_fitness(grid)
        no_improvement = 0
        
        for _ in range(depth):
            # Try different neighborhood structures
            candidates = [
                self._swap_in_row(best_grid),
                self._swap_rows(best_grid),
                
            ]
            
            improved = False
            for candidate in candidates:
                fitness = self.calculate_fitness(candidate)
                if fitness > best_fitness:
                    best_grid = candidate
                    best_fitness = fitness
                    improved = True
                    no_improvement = 0
                    break
            
            if not improved:
                no_improvement += 1
                if no_improvement > self.early_ls_stop:  # Early stopping
                    break
                    
        return best_grid
    

        

    def adaptive_local_search(self, grid: List[List[int]], generations_stagnant: int) -> List[List[int]]:
        """Adapt local search intensity based on stagnation"""
        base_depth = self.local_search_depth
        additional_depth = min(50, generations_stagnant * 2)
        return self.local_search(grid, base_depth + additional_depth)
    
    def _generate_new_population(self, population: List[List[List[int]]], generations_without_improvement: int) -> List[List[List[int]]]:
        """Generate a new population using elitism and various genetic operators
        
        Args:
            population (List[List[List[int]]]): Current population
            generations_without_improvement (int): Number of generations without fitness improvement
            
        Returns:
            List[List[List[int]]]: New population
        """
        new_population = []
        
        # Elitism with local search
        elite_count = max(5, self.population_size // 20)
        elite = sorted(population, key=self.calculate_fitness)[-elite_count:]
        optimized_elite = [self.adaptive_local_search(ind, generations_without_improvement) 
                         for ind in elite]
        new_population.extend(deepcopy(optimized_elite))
        
        # Generate rest of population
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                child = self.crossover(parent1, parent2)
            else:
                child = self.tournament_selection(population)
            
            if random.random() < self.mutation_rate:
                child = self.mutate(child)
            
            # Apply local search with probability
            if random.random() < self.local_search_frequency:
                child = self.local_search(child)
                
            new_population.append(child)
            
        return new_population
    
    
    
    def reset_population(self) -> List[List[List[int]]]:
        """Reset the entire population when search stagnates.
        
        Returns:
            List[List[List[int]]]: A new randomly generated population
        """
        return [self.generate_individual() for _ in range(self.population_size)]
    
   
    
    def print_error_details(self, error_locations: Dict):
        """Print detailed information about error locations"""
        print("\nDetailed Error Locations:")
        
        if error_locations['rows']:
            print("\nRow Errors:")
            for row, duplicates in sorted(error_locations['rows'].items()):
                print(f"  Row {row + 1}:")
                for value, positions in sorted(duplicates.items()):
                    print(f"    Value {value} appears at columns: {[p + 1 for p in positions]}")
        
        if error_locations['columns']:
            print("\nColumn Errors:")
            for col, duplicates in sorted(error_locations['columns'].items()):
                print(f"  Column {col + 1}:")
                for value, positions in sorted(duplicates.items()):
                    print(f"    Value {value} appears at rows: {[p + 1 for p in positions]}")
        
        if error_locations['boxes']:
            print("\nBox Errors:")
            for box, duplicates in sorted(error_locations['boxes'].items()):
                box_row, box_col = divmod(box, 3)
                print(f"  Box at position {box + 1} (rows {box_row*3 + 1}-{box_row*3 + 3}, cols {box_col*3 + 1}-{box_col*3 + 3}):")
                for value, positions in sorted(duplicates.items()):
                    position_strs = [f"(r{r+1},c{c+1})" for r, c in positions]
                    print(f"    Value {value} appears at positions: {', '.join(position_strs)}")

    def print_grid(self, grid: List[List[int]], title: str = "", error_locations: Dict = None):
        """Print the Sudoku grid with error highlighting"""
        if title:
            print(f"\n{title}")
            
        error_positions = set()
        if error_locations:
            # Add row error positions
            for row, duplicates in error_locations['rows'].items():
                for value, positions in duplicates.items():
                    for col in positions:
                        error_positions.add((row, col))
            
            # Add column error positions
            for col, duplicates in error_locations['columns'].items():
                for value, positions in duplicates.items():
                    for row in positions:
                        error_positions.add((row, col))
            
            # Add box error positions
            for box, duplicates in error_locations['boxes'].items():
                for value, positions in duplicates.items():
                    for pos in positions:
                        error_positions.add(pos)

        print("")
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("")
            print("║", end=" ")
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    print("│", end=" ")
                if (i, j) in error_positions:
                    print(f"{grid[i][j]}*", end=" ")
                else:
                    print(f"{grid[i][j]} ", end=" ")
            print("║")
        print("")


    def solve(self, max_generations: int = 5000) -> Tuple[List[List[int]], List[float]]:
        start_time = time.time()
        population = [self.generate_individual() for _ in range(self.population_size)] #initialize the initial population
        fitness_history = []
        generations_without_improvement = 0
        
        for generation in range(max_generations):
            # Time limit check
            current_time = time.time()
            if current_time - start_time > 100:
                print("\nTime limit reached - stopping search")
                break

            # Calculate fitness
            fitness_scores = [self.calculate_fitness(ind) for ind in population]
            current_best_fitness = max(fitness_scores)
            fitness_history.append(current_best_fitness)
            
            # Update best solution
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = deepcopy(population[fitness_scores.index(current_best_fitness)])
                generations_without_improvement = 0
                self._print_progress(generation, start_time)
            else:
                generations_without_improvement += 1
            
            if self.best_fitness == 1.0:
                break
                
            # Create new population using elitist strategy
            population = self._generate_new_population(population, generations_without_improvement)
            
            # Periodic reporting
            if generation % 100 == 0:
                self._print_progress(generation, start_time)
            
            # Reset if stagnant
            if generations_without_improvement > self.stagnation_threshold:
                population = self.reset_population()
                generations_without_improvement = 0
        
        self._print_final_results(generation, start_time, fitness_history)
        return self.best_solution, fitness_history

    def _print_progress(self, generation: int, start_time: float):
        total_errors, row_errs, col_errs, box_errs, error_locs = self.count_errors(self.best_solution)
        elapsed = time.time() - start_time
        print(f"\nGeneration {generation} | Time: {elapsed:.2f}s | Fitness: {self.best_fitness:.4f}")
        print(f"Errors - Row: {row_errs}, Col: {col_errs}, Box: {box_errs}, Total: {total_errors}")
        self.print_grid(self.best_solution, "Current Best", error_locs)

    def _print_final_results(self, generation: int, start_time: float, fitness_history: List[float]):
        elapsed = time.time() - start_time
        print(f"\n=== Final Results ===")
        print(f"Time: {elapsed:.2f}s | Generations: {generation + 1}")
        print(f"Final fitness: {self.best_fitness:.4f}")
        
        plt.figure(figsize=(10, 5))
        plt.plot(fitness_history)
        plt.title(f'Fitness History (Time: {elapsed:.2f}s)')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    sudoku = [
   [3,0,2,0,0,6,0,0,1],
[8,0,0,0,0,0,7,0,0],
[0,0,6,0,0,4,0,0,9],
[0,0,7,0,0,0,0,0,0],
[0,0,0,0,5,0,0,0,0],
[4,6,0,0,0,2,0,9,0],
[0,4,0,0,0,0,0,1,0],
[6,8,0,0,0,1,0,0,2],
[0,2,0,9,5,0,0,0,6]
    ]

    solver = SudokuSolver(sudoku)
    solution, history = solver.solve()   