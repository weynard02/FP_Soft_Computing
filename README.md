# FP_Soft_Computing

## How to run
At first, you must install dependencies specified in `requirements.txt` file by run command:
```
pip install -r requirements.txt
```

### Genetic Algorithm (ga.py)
#### The main function as follows:
```py
if __name__ == '__main__':
    DATASET_PATH = "./datasets/berlin52.tsp"
    random.seed(0)
    load_dataset()
    
    # set parameter by change the tsp_ga argument
    # arg1, arg2, arg3 -> time, mutation rate, crossover rate
    running_time = 1
    mutation_rate = 0.4
    crossover_rate = 0.7
    solution, solution_history = tsp_ga(running_time, 0.4, 0.7)
    
    plot_route(solution)
    plot_dist(solution_history)
```
1. Before run the program, make sure the dataset `berlin52.tsp` available with the same path as `DATASET_PATH` (you can change the path wherever you want)
2. Set the interval time for the program to run (for this example `running_time = 1`)
3. Set the parameters for mutation rate and crossover rate in the main function (for this example `mutation_rate = 0.4` and `crossover_rate = 0.7`)
4. Run the program and see the result

### Ant Colony Optimization (aco.py)

#### The main function as follows:

```py
if __name__ == '__main__':
    tsp_file_path = "./datasets/berlin52.tsp"
    berlin52_nodes = load_tsp_file(tsp_file_path)
    random.seed(0)


    INTERVAL_TIME = 2
    aco_solver = SolveTSPUsingACO(colony_size=3, steps=10000, nodes=berlin52_nodes)
    runtime, distance, path = aco_solver.run()
    print(f"Runtime: {runtime:.2f}s")
    print(f"Path: {path}")
    print(f"Best Distance: {distance:.2f}")

    aco_solver.plot()
    aco_solver.plot_dist()
```

1. Before run the program, make sure the dataset `berlin52.tsp` available with the same path as `tsp_file_path` (you can change the path wherever you want)
2. Set the interval time for the program to run (for this example `INTERVAL_TIME = 2`)
3. Set the parameters on colony size, steps (if needed), and nodes in `SolveTSPUsingACO` function
4. Run the program and see the result

### Simulated Annealing (SA.py)
```python
time_list = [0.5,1,2]
alpha = [0.95,0.96,0.97,0.98,0.99]
data = tsplib95.load('./Datasets/berlin52.tsp')
cities = list(data.get_nodes())
list_dist = []
for i in time_list:
    for j in alpha:
        print("-"*50)
        print("TIME : ",i,"ALPHA : ",j)
        list_dist.append(tsp_SA(i,j))
        print("-"*50)

```
1. Before run the program, make sure the dataset `berlin52.tsp` available with the same path as `tsp_file_path` (you can change the path wherever you want)
2. Set the interval time for the program to run (for this example `INTERVAL_TIME = 0.5`)
3. Set the parameters of cooling rate(alpha)
4. Run `python SA.py` and see the result
### Particle Swarm Optimization (pso.py)
#### The main function as follows:
```py
dataset = TSPDataset()
dataset.read('./datasets/berlin52.tsp')
dataset.unique(eps=1e-4, inplace=True)

problem = Problem(xy=dataset.positions)
optimizer = TSPPSO(interval=INTERVAL_TIME, poolsize=NUM_OF_PARTICLE, p1=0.95, p2=0.03, p3=0.02, max_no_improv=MAX_NO_IMPROVE)
solution, solution_history_1 = optimizer.minimize(problem)
print("Best Distance: {}".format(solution.cost))
print("-"*50)
```
1. Before run the program, make sure the dataset `berlin52.tsp` available with the same path as `tsp_file_path` (you can change the path wherever you want)
2. Set the interval time for the program to run (for this example `INTERVAL_TIME = 0.5`)
3. Set the parameters number of particle in `NUM_OF_PARTICLE` and maximum number improve in `MAX_NO_IMPROVE`
4. Run the program and see the result

Pre-run code can be seen in this google colab:
https://colab.research.google.com/drive/1RGVRa5RtDB65tkmWF1ZWoxbohGvVAQHW?usp=sharing
