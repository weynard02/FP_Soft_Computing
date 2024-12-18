import abc
import time
import matplotlib.pyplot as plt
import random
from collections import namedtuple
from typing import Union, Tuple, List, Iterable, Sequence
from dataclasses import dataclass
from random import Random
from functools import total_ordering

INTERVAL_TIME = 1
NUM_OF_PARTICLE = 50
MAX_NO_IMPROVE = 3

########## Data Types ##########
FloatMatrix = Tuple[Tuple[float, ...], ...]

class Position(namedtuple("Position", "x y")):
    @staticmethod
    # Menyimpan posisi koordinat
    def make(xy: Tuple[float, float]):
        return Position._make(xy)


########## Utilities ##########

# Menghitung jarak Euclidean antara dua posisi a dan b dalam ruang 2D
def euclidean_distance(a: Position, b: Position) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

# Matriks jarak untuk semua pasangan posisi dalam daftar positions
def distances_matrix(positions: List[Position]) -> FloatMatrix:
    n = len(positions)
    return tuple(
        tuple(
            0 if i == j else euclidean_distance(positions[i], positions[j])
            for j in range(n)
        ) for i in range(n))

# Operasi "left shift" pada sebuah urutan seq sebanyak k kali.
def lshift(seq: list, k: int) -> list:
    n = len(seq)
    k = k - k // n * n if k > n else k
    right_seq = seq[k:]
    left_seq = seq[:k]
    shifted_seq = right_seq + left_seq
    return shifted_seq

# Membalik suburutan dalam urutan yang diberikan
def neighborhood_inversion(seq: List[int], i: int, j: int) -> List[int]:
    seq = seq.copy()
    if j > i:
        seq[i:j+1] = seq[i:j+1][::-1]
    elif i > j:
        n = len(seq)
        neighborhood = seq[i:] + seq[:j+1]
        inv_neighborhood = neighborhood[::-1]
        seq[i:] = inv_neighborhood[:n-i]
        seq[:j+1] = inv_neighborhood[n-i:]

    return seq

########## Datasets ##########
class TSPDataset(object):
    def __init__(self, positions: List[Position] = None):
        self.positions: List[Position] = list() if positions is None else positions

    def read(self, filepath: str):
        with open(filepath, 'r') as fs:
            is_node_coord_section = False
            for line in fs:
                if not is_node_coord_section:
                    is_node_coord_section = line.startswith('NODE_COORD_SECTION')
                    continue
                elif line.startswith('EOF'):
                    break
                lineparts = line.split(' ')
                x = float(lineparts[1])
                y = float(lineparts[2])
                position = Position(x, y)
                self.positions.append(position)

    def unique(self, eps: float = 1e-6, inplace: bool = False) -> List[Position]:
        sorted_positions = sorted(self.positions, key=(lambda pos: pos.x if pos.x <= pos.y else pos.y))
        unique = list()
        pos_a = sorted_positions[0]
        n = len(sorted_positions)
        for i in range(1, n - 1):
            pos_b = sorted_positions[i]
            if abs(pos_a.x - pos_b.x) + abs(pos_a.y - pos_b.y) > eps:
                unique.append(pos_a)
            pos_a = sorted_positions[i]

        pos_b = sorted_positions[-1]
        if abs(pos_a.x - pos_b.x) + abs(pos_a.y - pos_b.y) > eps:
            unique.append(pos_a)
            unique.append(pos_b)
        else:
            unique.append(pos_a)

        if inplace:
            self.positions = unique

        return unique

########## TSP ##########
class Problem(object):
    def __init__(
            self,
            x: Sequence[float] = None,
            y: Sequence[float] = None,
            xy: Iterable[Union[Position, Tuple[float, float]]] = None
    ):
        self.__positions: List[Position]
        self.__distances: FloatMatrix

        if x is not None or y is not None:
            assert len(x) == len(y), "x and y sequences must be of the same length."
            self.__positions = [Position(xcoord, ycoord) for xcoord, ycoord in zip(x, y)]
        elif xy is not None:
            self.__positions = [Position(xcoord, ycoord) for xcoord, ycoord in xy]
        else:
            raise ValueError('Either xy or x and y arguments must not be None.')

        # Pre-compute distances matrix between points.
        self.__distances = distances_matrix(self.__positions)

    @property
    def positions(self) -> List[Position]:
        return self.__positions

    @property
    def distances(self) -> FloatMatrix:
        return self.__distances

@dataclass
@total_ordering
class Solution(object):
    sequence: List[int]
    cost: float
    best_sequence: List[int]
    best_cost: float

    def __eq__(self, other):
        return self.cost == other.cost

    def __lt__(self, other):
        return self.cost < other.cost

def pso_minimize(
        interval: float,
        poolsize: int,
        distances: FloatMatrix,
        positions: List[Tuple[float, float]],
        p1: float = 0.9,
        p2: float = 0.05,
        p3: float = 0.05,
        max_no_improv: int = 3,
        rng: Random = None
) -> Solution:

    rng = Random(0) if rng is None else rng

    # Initialize pool of solutions.
    solutions = list()
    base_indices = list(range(len(distances)))
    for i in range(poolsize):
        solution_indices = base_indices.copy()

        # Membuat populasi awal partikel (solusi) dengan urutan kota yang diacak
        rng.shuffle(solution_indices)

        # Menghitung cost dari setiap solusi menggunakan fungsi evaluate_cost
        solution_cost = evaluate_cost(solution_indices, distances)
        solution = create_solution(solution_indices, solution_cost)

        # Setiap solusi disimpan dalam list solutions
        solutions.append(solution)

    # Menentukan partikel dengan cost terbaik di seluruh swarm sebagai solusi global awal
    global_solution_index = solutions.index(min(solutions))
    global_solution = copy_solution(solutions[global_solution_index])

    # Time
    start_time = time.time()
    counter = 1

    solution_history = []

    while time.time() - start_time < interval:
        for i, solution in enumerate(solutions):
            # Define velocity of ith particle.
            velocity = define_velocity([p1, p2, p3], rng)

            if velocity == 0:
				        # move independently on it's own.
                move_solution_independently(solution, distances, max_no_improv, rng)
            elif velocity == 1:
                # move toward personal best position.
                move_solution_to_personal_best(solution, distances)
            else:
                # move toward swarm best position.
                move_solution_to_swarm_best(solution, global_solution, distances)

            if solution.cost < solution.best_cost:
                # Update each particle's personal best solution.
                solution.best_sequence = solution.sequence
                solution.best_cost = solution.cost

        global_solution_index = solutions.index(min(solutions))
        copy_solution_to(solutions[global_solution_index], global_solution)

        # mengubah dinamika pengambilan keputusan partikel untuk eksplorasi
        p1 *= 0.95
        p2 *= 1.01
        p3 = 1 - (p1 + p2)

        if (p1 + p2 + p3) != 1.0:
            total = p1 + p2
            p1 /= total
            p2 /= total
            p3 = 1 - (p1 + p2)
        #print('iteration:', counter, 'g-best:', global_solution.cost)
        solution_history.append(global_solution.cost)
        counter += 1

    end_time = time.time()
    runtime = end_time - start_time
    print(f'Runtime: {runtime:.2f}s')
    print('iteration:', counter)
    plot(positions, global_solution, counter)
    plot_dist(solution_history)
    return global_solution, solution_history

# move independently on it's own.
def move_solution_independently(solution: Solution, distances: FloatMatrix, max_no_improv: int, rng: Random):
    sequence, delta_cost = neighborhood_inversion_search(solution.sequence, distances, max_no_improv, rng)
    solution.sequence = sequence
    solution.cost += delta_cost

# move toward personal best position.
def move_solution_to_personal_best(solution: Solution, distances: FloatMatrix):
    sequence, cost = path_relinking_search(
        solution.sequence, solution.best_sequence, solution.best_cost, distances)
    solution.sequence = sequence
    solution.cost = cost

# move toward swarm best position.
def move_solution_to_swarm_best(solution: Solution, swarm_solution: Solution, distances: FloatMatrix):
    sequence, cost = path_relinking_search(
        solution.sequence, swarm_solution.sequence, swarm_solution.cost, distances)
    solution.sequence = sequence
    solution.cost = cost

# Menentukan jenis gerakan partikel berdasarkan probabilitas.
def define_velocity(probas: List[float], rng: Random = None) -> int:
    assert sum(probas) == 1.0, 'Sum of all probabilities must be equal to 1.'
    indices = list(range(len(probas)))
    chosen_velocities_ids = rng.choices(indices, probas, k=1)
    return chosen_velocities_ids[0]

# Menghitung biaya total jalur berdasarkan urutan (sequence) dan matriks jarak (distances).
def evaluate_cost(seq: List[int], distances: FloatMatrix) -> float:
    cost = 0
    n = len(seq)
    for i in range(1, n):
        cost += distances[seq[i - 1]][seq[i]]
    return cost + distances[seq[-1]][seq[0]]

# Mencoba memandu urutan jalur (sequence) agar mendekati jalur target (target)
def path_relinking_search(
        origin: List[int],
        target: List[int],
        target_cost: float,
        distances: FloatMatrix
) -> Tuple[List[int], float]:

    best_seq = target
    best_cost = target_cost

    target_value = target[0]
    target_index = origin.index(target_value)
    seq = lshift(origin, target_index)

    n = len(target)
    for i in range(1, n - 1):
        target_value = target[i]
        right_seq = seq[i:]
        target_index = right_seq.index(target_value)  # target element index that is used as shifting distance.
        seq[i:] = lshift(right_seq, target_index)

        cost = evaluate_cost(seq, distances)
        if cost < best_cost:
            best_seq = seq.copy()
            best_cost = cost

    return best_seq, best_cost

# Membalikkan elemen-elemen dalam segmen kecil dari jalur untuk mencari tetangga dengan biaya lebih rendah.
def neighborhood_inversion_search(
        seq: List[int],
        distances: FloatMatrix,
        max_no_improv: int,
        rng: Random = None
) -> Tuple[List[int], float]:
    rng = Random() if rng is None else rng

    best_delta_cost = 0
    best_i = 0
    best_j = 0

    n = len(seq)  # sequence size.
    m = 2  # neighborhood size.
    no_improv_count = 0  # number of iterations with no improvement for current neighborhood size.

    while n - m > 1:
        # Generate neighborhood range [i, j].
        i = rng.randint(0, n - 1)
        j = i + m - 1
        j = j if j < n else j - n

        ia = seq[i - 1]
        ib = seq[i]
        ja = seq[j]
        jb = seq[j + 1 if j + 1 < n else 0]

        cost0 = distances[ia][ib] + distances[ja][jb]
        cost1 = distances[ia][ja] + distances[ib][jb]
        delta_cost = cost1 - cost0

        if delta_cost < best_delta_cost:
            best_delta_cost = delta_cost
            best_i = i
            best_j = j
            m += 1
            no_improv_count = 0
        else:
            no_improv_count += 1
            if no_improv_count >= max_no_improv:
                m += 1
                no_improv_count = 0

    result_seq = neighborhood_inversion(seq, best_i, best_j)
    return result_seq, best_delta_cost

# Membuat dan menyalin solusi.
def create_solution(sequence: List[int], cost: float) -> Solution:
    return Solution(sequence, cost, sequence, cost)

def copy_solution(solution: Solution) -> Solution:
    return Solution(solution.sequence, solution.cost, solution.best_sequence, solution.best_cost)

def copy_solution_to(src: Solution, dst: Solution):
    dst.sequence = src.sequence
    dst.cost = src.cost
    dst.best_sequence = src.best_sequence
    dst.best_cost = src.best_cost

########## PSO ##########
class PSO(abc.ABC):
    @abc.abstractmethod
    def minimize(self, problem: Problem) -> Solution:
        pass

@dataclass
class TSPPSO(PSO):
    interval: float
    poolsize: int
    p1: float = 0.9
    p2: float = 0.05
    p3: float = 0.05
    max_no_improv: int = 3
    rng: Random = None

    def minimize(self, problem: Problem) -> Solution:
        distance, solution_history = pso_minimize(
            self.interval, self.poolsize, problem.distances, problem.positions, self.p1,
            self.p2, self.p3, self.max_no_improv, self.rng)

        return distance, solution_history

def plot(positions, global_solution, iteration):
    plt.figure(figsize=(10, 6))

    # Plot semua kota
    x_coords, y_coords = zip(*positions)
    plt.scatter(x_coords, y_coords, label='Nodes (Cities)')

    # Plot solusi global terbaik
    best_sequence = global_solution.sequence
    best_x = [positions[i][0] for i in best_sequence]
    best_y = [positions[i][1] for i in best_sequence]
    plt.plot(best_x + [best_x[0]], best_y + [best_y[0]], marker='o', linewidth=2, label='Global Best Solution')

    # Judul dan legenda
    plt.title(f'Optimal TSP Route \nIteration {iteration}: Best Cost = {global_solution.cost:.2f}')
    # plt.legend()
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    # plt.grid()
    plt.show()

def plot_dist(solution_history: list):
    plt.figure(figsize=[8, 6])
    plt.plot(solution_history, 'b', linewidth=3.0)
    plt.legend(['Solution Fitness'], fontsize=18)
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Distance', fontsize=16)
    plt.title('Solution History', fontsize=16)
    plt.show()

# import TSPDataset
dataset = TSPDataset()
dataset.read('./datasets/berlin52.tsp')
dataset.unique(eps=1e-4, inplace=True)

problem = Problem(xy=dataset.positions)
optimizer = TSPPSO(interval=INTERVAL_TIME, poolsize=NUM_OF_PARTICLE, p1=0.95, p2=0.03, p3=0.02, max_no_improv=MAX_NO_IMPROVE)
solution, solution_history_1 = optimizer.minimize(problem)
print("Best Distance: {}".format(solution.cost))
print("-"*50)