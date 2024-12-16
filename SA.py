import tsplib95
import random
import math
import time
import copy
import matplotlib.pyplot as plt

# Change tsp file name to run on separate tsp datasets
data = tsplib95.load('./Datasets/berlin52.tsp')
cities = list(data.get_nodes())


def annealing(initial_state, time_limit=300, alpha=0.99):  # default time limit of 300 seconds (5 minutes)
    """Performs simulated annealing to find a solution with a time limit"""
    initial_temp = 5000
    current_temp = initial_temp
    distances = []

    solution = initial_state
    same_solution = 0
    same_cost_diff = 0

    start_time = time.time()  # Record start time
    last_save_time = start_time  # Initialize last save time
    i = 0
    while same_solution < 1500 and same_cost_diff < 150000:
        # Check if time limit has been reached
        current_time = time.time()

        if (current_time - start_time) >= time_limit:
            print(f"Time limit of {time_limit} seconds reached!")
            break

        neighbor = get_neighbors(solution)

        # Check if neighbor is better
        cost_diff = get_cost(neighbor + [neighbor[0]]) - get_cost(solution + [solution[0]])
        # if the new solution is better, accept it
        if cost_diff > 0:
            solution = neighbor
            same_solution = 0
            same_cost_diff = 0

        elif cost_diff == 0:
            solution = neighbor
            same_solution = 0
            same_cost_diff += 1
        # if the new solution is not better, accept it with a probability of e^(-cost/temp)
        else:
            if random.uniform(0, 1) <= math.exp(float(cost_diff) / float(current_temp)):
                solution = neighbor
                same_solution = 0
                same_cost_diff = 0
            else:
                same_solution += 1
                same_cost_diff += 1
        # decrement the temperature
        current_temp = current_temp * alpha
        distance = 1 / get_cost(solution)  # Calculate distance for the current solution

        # Save the distance for this iteration
        distances.append(distance)

        # Optional: print distance periodically
        if current_time - last_save_time >= 1:
            last_save_time = current_time
            print(f"Iteration {i}, Distance: {distance}")

        i += 1

    return solution + [solution[0]], 1 / get_cost(solution), distances

def plot_distances(distances):
    """Plots the distances tracked during the annealing process."""
    for i in range(len(distances)-1):
        if distances[i+1] > distances[i]:
            distances[i+1] = distances[i]
    plt.figure()
    plt.plot(distances, label='Distance per 0.2 Seconds')
    plt.xlabel('Recording Interval (every 0.2 sec)')
    plt.ylabel('Distance')
    plt.title('Distance Changes Over Time')
    plt.legend()
    plt.show()


def get_cost(state):
    """Calculates cost/fitness for the solution/route."""
    distance = 0

    for i in range(len(state)):
        from_city = state[i]
        to_city = state[i + 1] if i + 1 < len(state) else state[0]
        distance += data.get_weight(from_city, to_city)
    fitness = 1 / float(distance)
    return fitness


def get_neighbors(state):
    """Returns a neighbor of your solution."""
    neighbor = copy.deepcopy(state)

    func = random.choice([0, 1, 2, 3])
    if func == 0:
        inverse(neighbor)
    elif func == 1:
        insert(neighbor)
    elif func == 2:
        swap(neighbor)
    else:
        swap_routes(neighbor)

    return neighbor


def inverse(state):
    """Inverses the order of cities in a route between two randomly selected nodes."""
    node_one = random.choice(state)
    new_list = [city for city in state if city != node_one]  # route without the selected node one
    node_two = random.choice(new_list)
    index_one = state.index(node_one)
    index_two = state.index(node_two)
    if index_one < index_two:
        state[index_one:index_two] = state[index_one:index_two][::-1]
    else:
        state[index_two:index_one] = state[index_two:index_one][::-1]
    return state


def insert(state):
    """Inserts a randomly selected city at a new random position."""
    node_j = random.choice(state)
    state.remove(node_j)
    node_i = random.choice(state)
    index = state.index(node_i)
    state.insert(index, node_j)
    return state


def swap(state):
    """Swaps two randomly selected cities in the route."""
    pos_one, pos_two = random.sample(range(len(state)), 2)
    state[pos_one], state[pos_two] = state[pos_two], state[pos_one]
    return state


def swap_routes(state):
    """Selects a subroute and inserts it at another random position in the route."""
    if len(state) < 2:
        return state  # Cannot swap routes if there are fewer than 2 cities

    subroute_a, subroute_b = sorted(random.sample(range(len(state)), 2))
    subroute = state[subroute_a:subroute_b]
    del state[subroute_a:subroute_b]

    if len(state) == 0:
        state.extend(subroute)
        return state

    insert_pos = random.randint(0, len(state))
    for city in reversed(subroute):
        state.insert(insert_pos, city)
    return state


best_route_distance = []
best_route = []
convergence_time = []


def tsp_SA(time_limit=300, alpha=0.99):
    """Runs the Simulated Annealing algorithm for TSP."""
    start = time.time()
    route, route_distance, route_distances = annealing(cities, time_limit, alpha=alpha)
    print("Best route found:", route)
    print("Best route distance:", route_distance)
    print("Number of cities in route:", len(route))
    time_elapsed = time.time() - start
    best_route_distance.append(route_distance)
    best_route.append(route)
    convergence_time.append(time_elapsed)

    # Plot Routes
    xs = [data.node_coords[i][0]for i in route]
    ys = [data.node_coords[i][1] for i in route]

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, 'o-', label='Route')
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.title('Best TSP Route Found by Simulated Annealing')
    plt.legend()
    plt.show()

    plot_distances(route_distances)
    return route_distances

random.seed(0)
