import tsplib95
import random
import math
import time
import copy
import matplotlib.pyplot as plt

#Change tsp file name to run on separate tsp datasets
data = tsplib95.load('./datasets/berlin52.tsp')
cities = list(data.get_nodes())


def annealing(initial_state, time_limit=300,alpha=0.99):  # default time limit of 300 seconds (5 minutes)
    """Performs simulated annealing to find a solution with a time limit"""
    initial_temp = 5000
    current_temp = initial_temp
    distances = []
    
    # Start by initializing the current state with the initial state
    solution = initial_state
    same_solution = 0
    same_cost_diff = 0
    
    start_time = time.time()  # Record start time
    i = 0 
    while same_solution < 1500 and same_cost_diff < 150000:
        # Check if time limit has been reached
        current_time = time.time()

        if (current_time - start_time) >= time_limit:
            print(f"Time limit of {time_limit} seconds reached!")
            break
            
        neighbor = get_neighbors(solution)
        
        # Check if neighbor is best so far
        cost_diff = get_cost(neighbor+[neighbor[0]]) - get_cost(solution+[solution[0]])
        # if the new solution is better, accept it
        if cost_diff > 0:
            solution = neighbor
            same_solution = 0
            same_cost_diff = 0
            
        elif cost_diff == 0:
            solution = neighbor
            same_solution = 0
            same_cost_diff +=1
        # if the new solution is not better, accept it with a probability of e^(-cost/temp)
        else:
            if random.uniform(0, 1) <= math.exp(float(cost_diff) / float(current_temp)):
                solution = neighbor
                same_solution = 0
                same_cost_diff = 0
            else:
                same_solution +=1
                same_cost_diff+=1
        # decrement the temperature
        current_temp = current_temp*alpha
        kk = get_cost(solution)
        distances.append(1/kk)
        i+=1
        print("iteration:",i,"distance:",1/kk)

    return solution+[solution[0]], 1/get_cost(solution),distances

def plot_distances(distances):
    """Plots the distances tracked during the annealing process."""
    plt.figure()
    plt.plot(distances, label='Distance per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.title('Distance Changes Over Iterations')
    plt.legend()
    plt.show()
def get_cost(state):
    """Calculates cost/fitness for the solution/route."""
    distance = 0
    
    for i in range(len(state)):
        # print(state)
        from_city = state[i]
        to_city = None
        if i+1 < len(state):
            to_city = state[i+1]
        else:
            to_city = state[0]
        distance += data.get_weight(from_city, to_city)
    fitness = 1/float(distance)
    return fitness
    
def get_neighbors(state):
    """Returns neighbor of  your solution."""
    
    neighbor = copy.deepcopy(state)
        
    
    func = random.choice([0,1,2,3])
    if func == 0:
        inverse(neighbor)
        
    elif func == 1:
        insert(neighbor)
        
    elif func == 2 :
        swap(neighbor)
    else:
        swap_routes(neighbor)
        
    return neighbor 

def inverse(state):
    "Inverses the order of cities in a route between node one and node two"
   
    node_one = random.choice(state)
    new_list = list(filter(lambda city: city != node_one, state)) #route without the selected node one
    node_two = random.choice(new_list)
    state[min(node_one,node_two):max(node_one,node_two)] = state[min(node_one,node_two):max(node_one,node_two)][::-1]
    
    return state

def insert(state):
    "Insert city at node j before node i"
    node_j = random.choice(state)
    state.remove(node_j)
    node_i = random.choice(state)
    index = state.index(node_i)
    state.insert(index, node_j)
    
    return state

def swap(state):
    "Swap cities at positions i and j with each other"
    pos_one = random.choice(range(len(state)))
    pos_two = random.choice(range(len(state)))
    state[pos_one], state[pos_two] = state[pos_two], state[pos_one]
    
    return state

def swap_routes(state):
    "Select a subroute from a to b and insert it at another position in the route"
    subroute_a = random.choice(range(len(state)))
    subroute_b = random.choice(range(len(state)))
    subroute = state[min(subroute_a,subroute_b):max(subroute_a, subroute_b)]
    del state[min(subroute_a,subroute_b):max(subroute_a, subroute_b)]
    insert_pos = random.choice(range(len(state)))
    for i in subroute:
        state.insert(insert_pos, i)
    return state


best_route_distance = []
best_route = []
convergence_time = []
def tsp_SA(time_limit,alpha):
    start = time.time()
    route, route_distance,route_distances = annealing(cities, time_limit,alpha=alpha)  
    print(route, route_distance,len(route))
    time_elapsed = time.time() - start
    best_route_distance.append(route_distance)
    best_route.append(route)
    convergence_time.append(time_elapsed)
    
    #Plot Routes
    xs = [data.node_coords[i][0] for i in route]
    ys = [data.node_coords[i][1] for i in route]

    plt.clf()

    plt.plot(xs,ys,'o-')
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.show()   
    plot_distances(route_distances)
random.seed(0)
time_list = [0.5,1,2,200]
alpha = [0.5,0.75,0.95]
for i in time_list:
    for j in alpha:
        print("-"*50)
        print("TIME : ",i,"ALPHA : ",j)
        tsp_SA(i,j)
        print("-"*50)
        
time_list = [0.5,1,2,200]
alpha = [0.75]
for i in time_list:
    for j in alpha:
        print("-"*50)
        print("TIME : ",i,"ALPHA : ",j)
        tsp_SA(i,j)
        print("-"*50)

time_list = [0.5,1,2,200]     
alpha = [0.95]
for i in time_list:
    for j in alpha:
        print("-"*50)
        print("TIME : ",i,"ALPHA : ",j)
        tsp_SA(i,j)
        print("-"*50)

