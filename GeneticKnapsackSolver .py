#!/usr/bin/env python
# coding: utf-8

# In[6]:


import random
import numpy as np

# Fitness function
def fitness(chromosome, items, capacity):
    total_weight = sum(chromosome[i] * items[i][0] for i in range(len(chromosome)))
    total_benefit = sum(chromosome[i] * items[i][1] for i in range(len(chromosome)))
    if total_weight > capacity:
        return 0  # Penalize invalid solutions
    return total_benefit

# Tournament Selection
def select_parents(population, fitness_values, tournament_size=5):
    indices = random.sample(range(len(population)), tournament_size)
    selected = max(indices, key=lambda idx: fitness_values[idx])
    return population[selected]

# Multi-point Crossover
def crossover(parent1, parent2, crossover_prob=0.9, num_points=3):
    if random.random() > crossover_prob:
        return parent1, parent2  # No crossover
    
    crossover_points = sorted(random.sample(range(1, len(parent1)), num_points))
    child1 = parent1[:crossover_points[0]] + parent2[crossover_points[0]:crossover_points[1]] + parent1[crossover_points[1]:]
    child2 = parent2[:crossover_points[0]] + parent1[crossover_points[0]:crossover_points[1]] + parent2[crossover_points[1]:]
    
    return child1, child2

# Adaptive Mutation
def mutate(chromosome, mutation_rate, max_mutation_rate=0.1, min_mutation_rate=0.01):
    rate = max(min_mutation_rate, mutation_rate * random.uniform(0.5, 1.5))  # Adaptive mutation rate
    return [1 - gene if random.random() < rate else gene for gene in chromosome]

# Simulated Annealing Local Search
def improvement(chromosome, items, capacity, initial_temp=1000, cooling_rate=0.995, max_iterations=100):
    temp = initial_temp
    best_fitness = fitness(chromosome, items, capacity)
    best_chromosome = chromosome
    for _ in range(max_iterations):
        new_chromosome = chromosome[:]
        i = random.randint(0, len(chromosome) - 1)
        new_chromosome[i] = 1 - new_chromosome[i]  # Flip bit
        
        new_fitness = fitness(new_chromosome, items, capacity)
        if new_fitness > best_fitness or random.random() < np.exp((new_fitness - best_fitness) / temp):
            best_fitness = new_fitness
            best_chromosome = new_chromosome
        
        temp *= cooling_rate  # Cool down the temperature
        if temp < 1e-3:
            break
    return best_chromosome

# Genetic Algorithm
def genetic_knapsack(items, capacity, population_size=200, generations=1000, 
                     mutation_rate=0.02, elitism_rate=0.2, convergence_threshold=100):
    num_items = len(items)
    population = [random.choices([0, 1], k=num_items) for _ in range(population_size)]
    best_benefit = 0
    no_improvement_count = 0
    
    for gen in range(generations):
        fitness_values = [fitness(individual, items, capacity) for individual in population]
        new_population = []
        
        # Elitism: Preserve top individuals
        num_elites = max(1, int(population_size * elitism_rate))
        elite_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i], reverse=True)[:num_elites]
        for elite_idx in elite_indices:
            new_population.append(population[elite_idx])
        
        # Generate new population
        while len(new_population) < population_size:
            parent1 = select_parents(population, fitness_values)
            parent2 = select_parents(population, fitness_values)
            child1, child2 = crossover(parent1, parent2, crossover_prob=0.9, num_points=3)
            new_population.append(mutate(child1, mutation_rate))
            if len(new_population) < population_size:
                new_population.append(mutate(child2, mutation_rate))
        
        # Apply local search to the best individual using simulated annealing
        best_index = max(range(len(fitness_values)), key=lambda idx: fitness_values[idx])
        new_population[0] = improvement(population[best_index], items, capacity)
        
        # Update population
        population = new_population
        
        # Check for convergence
        max_fitness = max(fitness_values)
        if max_fitness > best_benefit:
            best_benefit = max_fitness
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if no_improvement_count >= convergence_threshold:
            break
    
    # Final evaluation of fitness
    final_fitness_values = [fitness(individual, items, capacity) for individual in population]
    best_index = max(range(len(final_fitness_values)), key=lambda idx: final_fitness_values[idx])
    best_benefit = final_fitness_values[best_index]
    return best_benefit

# Main function
def main():
    input_file = "C:/Users/Soft/Downloads/input_example.txt"  # Adjust to your file path
    with open(input_file, 'r') as file:
        data = file.readlines()
    
    data = [line.strip() for line in data if line.strip()]  # Remove empty lines
    num_cases = int(data[0])  # Number of cases
    line_index = 1  # Starting line index
    
    for case in range(1, num_cases + 1):
        num_items = int(data[line_index])  # Number of items
        line_index += 1
        knapsack_capacity = int(data[line_index])  # Knapsack capacity
        line_index += 1
        
        items = []
        for i in range(num_items):
            weight, benefit = map(int, data[line_index].split())
            items.append((weight, benefit))
            line_index += 1
        
        # Run genetic algorithm
        best_benefit = genetic_knapsack(items, knapsack_capacity, population_size=200, generations=1000, 
                                        mutation_rate=0.03, elitism_rate=0.2)
        print(f"Case: {case} {best_benefit}")

# Run the main function
if __name__ == "__main__":
    main()


# In[ ]:




