import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data
df = pd.read_csv('normalized_dataset.csv')
df_sitting = df[df['class'] == 4] # filter for class 4 (sitting)
df_others = df[df['class'] != 4] # filter for all other classes


# calculate mean vectors
mean_sitting = df_sitting.loc[:, 'x1':'z4'].mean().values
# print(mean_sitting)
mean_others = [df_others[df_others['class'] == i].loc[:, 'x1':'z4'].mean().values for i in range(4)]
# print(mean_others)


def fitness(individual):
    v = np.array(individual)
    c = 0  # or any constant value you want to choose
    cos_sim_sitting = np.dot(v, mean_sitting) / (np.linalg.norm(v) * np.linalg.norm(mean_sitting))
    # print(cos_sim_sitting)
    cos_sim_others = np.mean([np.dot(v, mean) / (np.linalg.norm(v) * np.linalg.norm(mean)) for mean in mean_others])
    # print(cos_sim_others)
    return (cos_sim_sitting + c * (1 - cos_sim_others)) / (1 + c),


# initialize population
def initialize_population(pop_size, ind_size):
    return [[random.uniform(0, 1) for _ in range(ind_size)] for _ in range(pop_size)]


# tournament selection
def tournament_selection(population, fitnesses, tourn_size):
    selected = []
    for _ in range(2):  # Select 2 parents
        individuals = random.choices(population, k=tourn_size)
        fitnesses_ind = [fitnesses[population.index(i)] for i in individuals]
        selected.append(individuals[fitnesses_ind.index(max(fitnesses_ind))])
    return selected


# uniform crossover
def crossover(parent1, parent2):
    child1, child2 = [], []
    for gene1, gene2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child1.append(gene1)
            child2.append(gene2)
        else:
            child1.append(gene2)
            child2.append(gene1)
    return [child1, child2]

# mutation
def mutation(individual, mu, sigma):
    for i in range(len(individual)):
        if random.random() < 0.1:  # mutation probability
            individual[i] += random.gauss(mu, sigma)
            individual[i] = max(min(individual[i], 1), 0)  # ensure within range (clipping)
    return individual

def genetic_algorithm(population_size, crossover_prob, mutation_prob, fitness_fn, max_generations=1000, 
                      no_improv_generations=100, improvement_threshold=0.001, elitism_rate=0.1):
    # Initialize the population
    population = initialize_population(population_size, 12)
    best_fitnesses = []
    stagnant_gen_count = 0
    for generation in range(max_generations):
        fitnesses = [fitness_fn(i)[0] for i in population]
        best_fitness = max(fitnesses)
        best_fitnesses.append(best_fitness)
        print(f'Generation {generation}, Best Fitness: {best_fitness}')
        
        # Check if improvement threshold or stagnation limit has been reached
        if len(best_fitnesses) > 1:
            if (best_fitnesses[-1] - best_fitnesses[-2]) / best_fitnesses[-2] < improvement_threshold:
                stagnant_gen_count += 1
            else:
                stagnant_gen_count = 0  # Reset stagnant generation count when improvement happens
                
            if stagnant_gen_count >= no_improv_generations:
                print('Terminating due to lack of improvement.')
                return best_fitnesses, population, fitnesses
        
        # Continue with the genetic algorithm
        next_population = []
        # Elitism: Preserve the fittest individuals.
        sorted_population = [x for _, x in sorted(zip(fitnesses, population), reverse=True)]
        elites = sorted_population[:int(len(population) * elitism_rate)]
        # Run the GA for the rest of the population.
        for _ in range((len(population) - len(elites)) // 2):
            if random.random() < crossover_prob: # Only crossover if the random number is less than crossover probability
                parents = tournament_selection(population, fitnesses, 3)
                children = crossover(*parents)
                next_population += [mutation(c, 0, 0.2) if random.random() < mutation_prob else c for c in children] # Only mutate if the random number is less than mutation probability
            else:
                next_population += random.sample(population, 2) # If not crossing over, then just carry forward individuals to next generation
        # Combine elites and next generation.
        population = elites + next_population
    
    return best_fitnesses, population, fitnesses


 
# determine parameter values
population_size = 20
crossover_prob = 0.6
mutation_prob = 0
generations = 1000


# run the genetic algorithm
best_fitnesses, population, fitnesses = genetic_algorithm(population_size=population_size, crossover_prob=crossover_prob, mutation_prob=mutation_prob, fitness_fn=fitness, max_generations=generations)


# calculate average best fitness
average_best_fitness = np.mean(best_fitnesses)
print('Average Best Fitness: ', average_best_fitness)


# best_fitnesses, population, fitnesses = genetic_algorithm(population_size=100, crossover_prob=0.7, mutation_prob=0.1, fitness_fn=fitness, max_generations=1000)
plt.figure(figsize=(10,5))
plt.plot(best_fitnesses)
plt.title('Best Fitness Over Generations')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.show()


