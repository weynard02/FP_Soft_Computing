import random
import math
import time

DATASET_PATH = ".\datasets\\berlin52.tsp"

class Point:
    def __init__(self, x: float, y: float):
        self.x: float = x
        self.y: float = y
        
class Individu:
    def __init__(self, chromosome, fitness):
        self.chromosome = chromosome
        self.fitness = fitness

def tsp_ga(data: list[Point]) -> Individu:
    # set parameter
    random_population_size = 200
    nn_population_size = 0
    mst_population_size = 0
    population_size = random_population_size + nn_population_size + mst_population_size
    cooling_param = 0
    chromosome_size = len(data)
    generations = 100
    mutation_rate = 0.05
    crossover_rate = 0.8
    
    start_time = time.time()

    # generate population
    chromosomes: list[list[int]] = _generate_populations(random_population_size, nn_population_size,\
        mst_population_size, chromosome_size)
    populations: list[Individu] = []
    
    # evaluate fitness
    for i in range(len(chromosomes)):
        fitness = calculate_fitness(data, chromosomes[i])
        populations.append(Individu(chromosomes[i], fitness))
    
    # generating new population
    best_solutions: list[Individu] = []
    for generation in range(generations):
        next_populations: list[Individu] = []
        
        while len(next_populations) < population_size:
            # select parent
            parent1: Individu = selection_tournament(populations, 2)
            parent2: Individu = selection_tournament(populations, 2)

            chromosome_child1: list[int]
            chromosome_child2: list[int]

            # crossover
            if random.random() < crossover_rate:
                chromosome_child1, chromosome_child2 = crossover_two_point(parent1.chromosome, parent2.chromosome)
            else:
                chromosome_child1 = parent1.chromosome
                chromosome_child2 = parent2.chromosome

            # mutation
            if random.random() < mutation_rate:
                mutation_swap(chromosome_child1)
            if random.random() < mutation_rate:
                mutation_swap(chromosome_child1)

            child1: Individu = Individu(chromosome_child1, calculate_fitness(data, chromosome_child1))
            child2: Individu = Individu(chromosome_child2, calculate_fitness(data, chromosome_child2))
            
            next_populations.extend([child1, child2])

        # Replace population
        populations = next_populations[:population_size]

        # Track the best solution
        best_solutions.append(populations[0])
        for i in range(1, len(populations)):
            if populations[i].fitness < best_solutions[generation].fitness:
                best_solutions[generation] = populations[i]

        print(f"Generation {generation + 1}: Best Fitness = {best_solutions[generation].fitness}")

    # print solution
    end_time = time.time()
    best_solution = min(best_solutions, key=lambda x: x.fitness)
    print(best_solution.chromosome)
    print(f"distance: {best_solution.fitness}")
    print(f"time: {end_time - start_time}")
    return Individu(1, 3)

# function to generate populations
def _generate_populations(random_size, nn_size, mst_size, chromosome_size) -> list[list[int]]:
    return _generate_random_population(random_size, chromosome_size) + _generate_nn_population(nn_size, chromosome_size)\
        + _generate_mst_population(mst_size, chromosome_size)

# function to generate random population
def _generate_random_population(size: int, chromosome_size: int) -> list[list[int]]:
    populations: list[list[int]] = []
    
    for _ in range(size):
        chromosome = [i for i in range(chromosome_size)]
        random.shuffle(chromosome)
        populations.append(chromosome)

    return populations

# TODO: function to generate random population
def _generate_nn_population(size: int, chromosome_size: int) -> list[list[int]]:
    return []

# TODO: function to generate random population
def _generate_mst_population(size: int, chromosome_size: int) -> list[list[int]]:
    return []

# funtion to select individu using tournament method
def selection_tournament(populations: list[Individu], tournament_size: int = 3) -> Individu:
    tournament = random.sample(populations, tournament_size)
    winner = min(tournament, key=lambda x: x.fitness)
    
    return winner

# function to do a crossover
def crossover_two_point(parent1: list[int], parent2: list[int]) -> tuple[list[int], list[int]]:
    # error if parent size is different
    if len(parent1) != len(parent2):
        raise Exception("Error: crossover in different parent size")

    # set crossover point, range [b, t]
    b = random.randint(1, len(parent1)-3)
    t = random.randint(b+1, len(parent1)-2)

    # initialize childs
    child1: list[int] = [i for i in range(len(parent1))]
    child2: list[int] = [i for i in range(len(parent2))]
    
    # helper to flag the visited node
    vis1: list[bool] = [False for _ in range(len(parent1))]
    vis2: list[bool] = [False for _ in range(len(parent1))]

    # set gene in crossover points
    for i in range(b, t+1):
        child1[i] = parent1[i]
        vis1[parent1[i]] = True
        
        child2[i] = parent2[i]
        vis2[parent2[i]] = True

    currIdx1 = 0
    currIdx2 = 0
    for i in range(len(parent1)):
        if not vis1[parent2[i]]:
            child1[currIdx1] = parent2[i]
            currIdx1 += 1
            if currIdx1 == b:
                currIdx1 = t+1
        
        if not vis2[parent1[i]]:
            child2[currIdx2] = parent1[i]
            currIdx2 += 1
            if currIdx2 == b:
                currIdx2 = t+1
    
    return child1, child2

# function to mutate a chromosome using swap method
# mutation directly done in passed list 
def mutation_swap(chromosome: list[int]):
    # set two genes will be swapped
    gene1: int = random.randint(0, len(chromosome)-1)
    gene2: int = random.randint(0, len(chromosome)-1)
    # make sure we get different genes
    while gene2 == gene1:
        gene2 = random.randint(0, len(chromosome)-1)

    temp = chromosome[gene1]
    chromosome[gene1] = chromosome[gene2]
    chromosome[gene2] = temp

# function to calculate fitness value for a chromosome
def calculate_fitness(node: list[Point], chromosome: list[int]) -> float:
    fitness = 0.0
    for i in range(1, len(chromosome)):
        prev = node[chromosome[i-1]]
        curr = node[chromosome[i]]
        fitness += distance(prev, curr)

    return fitness

# function to calculate distance between two points
def distance(p1: Point, p2: Point) -> float:
    return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)

def load_dataset() -> list[Point]:
    data: list[Point] = []
    file = open(DATASET_PATH, "r")
    # first 6 lines are metadata
    for _ in range(6):
        _ = file.readline()
    
    line = file.readline()
    while line != "EOF":
        _, x, y = line.split()
        data.append(Point(float(x), float(y)))
        line = file.readline()
        
    return data

if __name__ == '__main__':
    random.seed(0)
    data: list[Point] = load_dataset()
    tsp_ga(data)