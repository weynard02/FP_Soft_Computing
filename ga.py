import random
import math
import time
import matplotlib.pyplot as plt


class Point:
    def __init__(self, x: float, y: float):
        self.x: float = x
        self.y: float = y


class Individu:
    def __init__(self, chromosome: list[int], fitness: float):
        self.chromosome = chromosome
        self.fitness = fitness


data: list[Point] = []
data_distances: list[list[int]]


def tsp_ga(max_time, mutation_rate=0.4, crossover_rate=0.7):
    # set parameter
    population_size = 100
    random_population_size = 80
    nn_population_size = 20
    mst_population_size = 0
    population_size = random_population_size + \
        nn_population_size + mst_population_size
    cooling_param = 0
    chromosome_size = len(data)
    generations = 100
    tournament_size = 3

    start_time = time.time()

    # generate population
    chromosomes_random: list[list[int]] = _generate_random_population(random_population_size, chromosome_size)
    chromosomes_nn: list[list[int]] = _generate_nn_population(nn_population_size, chromosome_size)

    populations_random: list[Individu] = []
    populations_nn: list[Individu] = []
    populations: list[Individu] = []

    # evaluate fitness
    for i in range(len(chromosomes_random)):
        fitness = calculate_fitness(chromosomes_random[i])
        populations_random.append(Individu(chromosomes_random[i], fitness))
    for i in range(len(chromosomes_nn)):
        fitness = calculate_fitness(chromosomes_nn[i])
        populations_nn.append(Individu(chromosomes_nn[i], fitness))
    populations = populations_random + populations_nn

    # generating new population
    best_solutions: list[Individu] = []
    curr_time = 0.0
    generation = 0
    while curr_time < max_time:
        next_populations: list[Individu] = []

        while len(next_populations) < population_size-nn_population_size:
            # select parent
            parent1: Individu = selection_tournament(
                populations, tournament_size)
            parent2: Individu = selection_tournament(
                populations, tournament_size)

            chromosome_child1: list[int]
            chromosome_child2: list[int]

            # crossover
            if random.random() < crossover_rate:
                chromosome_child1, chromosome_child2 = crossover_two_point(
                    parent1.chromosome[:-1], parent2.chromosome[:-1])
            else:
                chromosome_child1 = parent1.chromosome[:-1]
                chromosome_child2 = parent2.chromosome[:-1]

            # mutation
            if random.random() < mutation_rate:
                mutation_swap(chromosome_child1)
            if random.random() < mutation_rate:
                mutation_swap(chromosome_child1)

            chromosome_child1.append(chromosome_child1[0])
            chromosome_child2.append(chromosome_child2[0])

            child1: Individu = Individu(
                chromosome_child1, calculate_fitness(chromosome_child1))
            child2: Individu = Individu(
                chromosome_child2, calculate_fitness(chromosome_child2))

            next_populations.extend([child1, child2])

        # Replace population
        populations = populations_nn + next_populations[:(population_size-nn_population_size)]

        # Track the best solution
        if generation > 0:
            best_solutions.append(best_solutions[generation-1])
        else:
            best_solutions.append(populations[0])
        for i in range(len(populations)):
            if populations[i].fitness < best_solutions[generation].fitness:
                best_solutions[generation] = populations[i]

        print(f"Generation {generation + 1}: Best Fitness = {best_solutions[generation].fitness}")

        generation += 1
        curr_time = time.time() - start_time

    # print solution
    end_time = time.time()
    best_solution = min(best_solutions, key=lambda x: x.fitness)
    print(best_solution.chromosome)
    print(f"distance: {best_solution.fitness}")
    print(f"time: {end_time - start_time}")
    return best_solution, best_solutions

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
        chromosome.append(chromosome[0])
        populations.append(chromosome)

    return populations

# TODO: function to generate random population


def _generate_nn_population(size: int, chromosome_size: int) -> list[list[int]]:
    first_nodes = random.sample(range(chromosome_size), size)
    populations: list[list[int]] = []

    for first_node in first_nodes:
        unvisited = set(range(chromosome_size))
        unvisited.remove(first_node)
        chromosome = [first_node]
        curr_idx = first_node
        while unvisited:
            nearest_city = min(
                unvisited, key=lambda i: distance(data[curr_idx], data[i]))
            chromosome.append(nearest_city)
            unvisited.remove(nearest_city)
            curr_idx = nearest_city
        chromosome.append(first_node)
        populations.append(chromosome)

    return populations

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


def calculate_fitness(chromosome: list[int]) -> float:
    fitness = 0.0
    for i in range(1, len(chromosome)):
        prev = data[chromosome[i-1]]
        curr = data[chromosome[i]]
        fitness += distance(prev, curr)

    return fitness

# function to calculate distance between two points


def distance(p1: Point, p2: Point) -> float:
    return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)


def plot_route(solution: Individu):
    x_coords = [data[i].x for i in solution.chromosome]
    y_coords = [data[i].y for i in solution.chromosome]

    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, y_coords, marker='o')
    plt.title('Optimal TSP Route')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()


def plot_dist(solution_history: list[Individu]):
    solution_fitness = [s.fitness for s in solution_history]
    plt.figure(figsize=[8, 6])
    plt.plot(solution_fitness, 'b', linewidth=3.0)
    plt.legend(['Solution Fitness'], fontsize=18)
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Distance', fontsize=16)
    plt.title('Solution History', fontsize=16)
    plt.show()


def load_dataset():
    file = open(DATASET_PATH, "r")
    # first 6 lines are metadata
    for _ in range(6):
        _ = file.readline()

    line = file.readline()
    while line != "EOF\n":
        _, x, y = line.split()
        data.append(Point(float(x), float(y)))
        line = file.readline()


if __name__ == '__main__':
    DATASET_PATH = "./datasets/berlin52.tsp"
    random.seed(0)
    load_dataset()
    
    # set parameter by change the tsp_ga argument
    # arg1, arg2, arg3 -> time, mutation rate, crossover rate
    solution, solution_history = tsp_ga(1, 0.4, 0.7)
    
    plot_route(solution)
    plot_dist(solution_history)