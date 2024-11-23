import random
import math
import time
import matplotlib.pyplot as plt

class SolveTSPUsingACO:
    def __init__(self, mode='ACS', colony_size=10, steps=100, nodes=None):
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

        for step in range(self.steps):
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

if __name__ == '__main__':
    berlin52_nodes = [
        (565.0, 575.0), (25.0, 185.0), (345.0, 750.0), (945.0, 685.0),
        (845.0, 655.0), (880.0, 660.0), (25.0, 230.0), (525.0, 1000.0),
        (580.0, 1175.0), (650.0, 1130.0), (1605.0, 620.0), (1220.0, 580.0),
        (1465.0, 200.0), (1530.0, 5.0), (845.0, 680.0), (725.0, 370.0),
        (145.0, 665.0), (415.0, 635.0), (510.0, 875.0), (560.0, 365.0),
        (300.0, 465.0), (520.0, 585.0), (480.0, 415.0), (835.0, 625.0),
        (975.0, 580.0), (1215.0, 245.0), (1320.0, 315.0), (1250.0, 400.0),
        (660.0, 180.0), (410.0, 250.0), (420.0, 555.0), (575.0, 665.0),
        (1150.0, 1160.0), (700.0, 580.0), (685.0, 595.0), (685.0, 610.0),
        (770.0, 610.0), (795.0, 645.0), (720.0, 635.0), (760.0, 650.0),
        (475.0, 960.0), (95.0, 260.0), (875.0, 920.0), (700.0, 500.0),
        (555.0, 815.0), (830.0, 485.0), (1170.0, 65.0), (830.0, 610.0),
        (605.0, 625.0), (595.0, 360.0), (1340.0, 725.0), (1740.0, 245.0)
    ]
    random.seed(0)

    aco_solver = SolveTSPUsingACO(mode='ACS', colony_size=10, steps=100, nodes=berlin52_nodes)
    runtime, distance = aco_solver.run()
    print(f"Runtime: {runtime:.2f}s")
    print(f"Best Distance: {distance:.2f}")

    aco_solver.plot()
