import random
import math
import time
import matplotlib.pyplot as plt

class SolveTSPUsingACO:
    def __init__(self, mode='ACS', colony_size=10, steps=100000, nodes=None):
        self.mode = mode  # ACS or other ACO variants
        self.colony_size = colony_size
        self.steps = steps
        self.nodes = nodes if nodes else []
        self.num_nodes = len(self.nodes)
        self.distances = self._calculate_distances()
        self.pheromones = [[1 for _ in range(self.num_nodes)] for _ in range(self.num_nodes)]

    def _calculate_distances(self):
        distances = [[0 for _ in range(self.num_nodes)] for _ in range(self.num_nodes)]
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                distances[i][j] = math.sqrt((self.nodes[i][0] - self.nodes[j][0]) ** 2 +
                                            (self.nodes[i][1] - self.nodes[j][1]) ** 2)
        return distances

    def _choose_next_node(self, current_node, unvisited_nodes):
        probabilities = []
        for next_node in unvisited_nodes:
            pheromone = self.pheromones[current_node][next_node]
            visibility = 1 / self.distances[current_node][next_node]
            probabilities.append(pheromone * visibility)

        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        return random.choices(unvisited_nodes, probabilities)[0]

    def _update_pheromones(self, paths, path_lengths):
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                self.pheromones[i][j] *= 0.9  # evaporation rate

        for path, length in zip(paths, path_lengths):
            for i in range(len(path) - 1):
                self.pheromones[path[i]][path[i + 1]] += 1 / length

    def run(self):
        best_distance = float('inf')
        best_path = []
        start_time = time.time()

        while time.time() - start_time < 0.5:
            paths = []
            path_lengths = []

            for _ in range(self.colony_size):
                unvisited_nodes = list(range(self.num_nodes))
                path = [unvisited_nodes.pop(random.randint(0, len(unvisited_nodes) - 1))]

                while unvisited_nodes:
                    next_node = self._choose_next_node(path[-1], unvisited_nodes)
                    path.append(next_node)
                    unvisited_nodes.remove(next_node)

                path.append(path[0])  # Return to the start
                paths.append(path)
                path_lengths.append(self._calculate_path_length(path))

            shortest_length = min(path_lengths)
            # print(f"Shortest length step-{step}: {shortest_length}")
            if shortest_length < best_distance:
                best_distance = shortest_length
                best_path = paths[path_lengths.index(shortest_length)]

            self._update_pheromones(paths, path_lengths)

        end_time = time.time()
        self.best_path = best_path
        self.best_distance = best_distance
        return end_time - start_time, best_distance

    def _calculate_path_length(self, path):
        return sum(self.distances[path[i]][path[i + 1]] for i in range(len(path) - 1))

    def plot(self):
        x_coords = [self.nodes[i][0] for i in self.best_path]
        y_coords = [self.nodes[i][1] for i in self.best_path]

        plt.figure(figsize=(10, 6))
        plt.plot(x_coords, y_coords, marker='o')
        plt.title('Optimal TSP Route')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()

def load_tsp_file(file_path):
    
    with open(file_path, 'r') as file:
        lines = file.readlines()

    nodes = []
    node_section_started = False

    for line in lines:
        line = line.strip()
        if line == "NODE_COORD_SECTION":
            node_section_started = True
            continue
        if line == "EOF":
            break
        if node_section_started:
            _, x, y = line.split()
            nodes.append((float(x), float(y)))

    return nodes

if __name__ == '__main__':
    tsp_file_path = './datasets/berlin52.tsp'
    berlin52_nodes = load_tsp_file(tsp_file_path)
    random.seed(0)

    aco_solver = SolveTSPUsingACO(mode='ACS', colony_size=10, steps=100, nodes=berlin52_nodes)
    runtime, distance = aco_solver.run()
    print(f"Runtime: {runtime:.2f}s")
    print(f"Best Distance: {distance:.2f}")

    aco_solver.plot()
