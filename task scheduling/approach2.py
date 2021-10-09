import random
import copy
import numpy as np

processors = 2
nodes = 4  # number of tasks
graph = []
graph_par = []
root_node = 0
computation_cost = []
crossover_prob = 0.5
mutation_prob = 0.03
DAT = []
FT = []

# function to take user input 
def take_graph_input():
    global processors, nodes, computation_cost, graph
    print("Enter the number of processors :", end="")
    processors = int(input().strip())
    print("Enter the number of nodes :", end="")
    nodes = int(input().strip())
    print("Enter computation cost of each node :", end="")
    computation_cost = list(map(int, input().strip().split(" ")))[:nodes]
    print("Enter the number of edges")
    cnt = int(input().strip())
    print("Enter edges and cost in the following format - StartNode EndNode Cost")
    graph = [[] for i in range(nodes)]
    for i in range(cnt):
        l = list(map(int, input().strip().split(" ")))[:3]
        graph[l[0]].append((l[1], l[2]))
    create_child_parent_graph()

# convert parent child mapping to child parent mapping 
def create_child_parent_graph():
    global graph_par
    graph_par = [[] for i in range(nodes)]
    for i in range(nodes):
        for pair in graph[i]:
            graph_par[pair[0]].append((i, pair[1]))

def selection_operation(init_pop_):
    fitness = [(ind, get_fitness_value(ind)) for ind in init_pop_]
    fitness = sorted(fitness, key = lambda x:x[1])
    length = len(init_pop_)
    weights = []
    weights = weights + [1 for i in range(length//4)]
    length = length - length//4
    weights = weights + [2 for i in range(length//4)]
    length = length - length//4
    weights = weights + [4 for i in range(length//4)]
    weights = weights + [8 for i in range(length - length//4)]
    ind1 = random.choices(fitness, weights)
    ind2 = random.choices(fitness, weights)
	
    return ind1[0][0], ind2[0][0]

def crossover_operation(ind1, ind2):
    c1, c2 = ind1, ind2
    length = len(ind1)
    if length > len(ind2):
        length = len(ind2)
    if random.random() <= crossover_prob:
        point1, point2 = random.randint(0, length), random.randint(0, length)
        if point1 > point2:
            temp = point1
            point1 = point2
            point2 = temp
        for i in range(point1+1, point2):
            temp = c1[i]
            c1[i] = c2[i]
            c2[i] = temp
    return c1, c2

def mutation_operation(ind):
    i = random.randint(0,len(ind)-1)
    if random.random() <= mutation_prob:
        ind[i] = (ind[i][0], random.randint(0, processors - 1))
    return ind

def invalid(ind):
    tasks = [i for i in range(nodes)]
    for i in range(len(ind)):
        if ind[i][0] in tasks:
            tasks.remove(ind[i][0])
        for par in graph_par[ind[i][0]]:
            if par[0] not in [ind[j][0] for j in range(i)]:
                return True
    if len(tasks):
        return True
    return False

def get_fitness_value(ind):
    if invalid(ind):
        return 0
    FPRED, S_list, idle_time = schedule_length(ind)
    return 1 / S_list[0][1]

def GA(init_pop):
    itr = 1000
    while itr:
        first, second = True, True
        ind1, ind2 = selection_operation(copy.deepcopy(init_pop))
        ind1, ind2 = crossover_operation(ind1, ind2)
        ind1 = mutation_operation(ind1)
        ind2 = mutation_operation(ind2)
        if invalid(ind1) and invalid(ind2):
            continue
        ind1_fitness = get_fitness_value(ind1)
        ind2_fitness = get_fitness_value(ind2)
        fitness_pop = []
        for val in init_pop:
            fitness_pop.append(get_fitness_value(val))
        pos = np.argmin(fitness_pop)
        if fitness_pop[pos] < ind1_fitness:
            init_pop[pos] = ind1
            first = False
        elif fitness_pop[pos] < ind2_fitness:
            init_pop[pos] = ind2
            second = False
        fitness_pop.remove(fitness_pop[pos])
        pos = np.argmin(fitness_pop)
        if fitness_pop[pos] < ind1_fitness and first and not invalid(ind1):
            init_pop[pos] = ind1
        elif fitness_pop[pos] < ind2_fitness and second and not invalid(ind2):
            init_pop[pos] = ind2
        itr -= 1
    fitness_pop = []
    for val in init_pop:
        fitness_pop.append(get_fitness_value(val))
    return init_pop[np.argmax(fitness_pop)]

def schedule_length(task_list):
    global DAT, FT
    DAT = [0 for i in range(nodes)]
    FT = [0 for i in range(nodes)]
    RT = [0 for i in range(processors)]
    ST = [0 for i in range(nodes)]
    FPRED = [None for i in range(nodes)]

    for vector in task_list:
        dat = 0
        for k in graph_par[vector[0]]:
            for task in task_list:
                if task[0] == k[0]:	# parent task
                    if task[1] == vector[1]:	# on same processor
                        if dat < FT[k[0]]:
                            dat = FT[k[0]]
                            FPRED[vector[0]] = k[0]
                    else:
                        if dat < FT[k[0]] + k[1]:
                            dat = FT[k[0]] + k[1]
                            FPRED[vector[0]] = k[0]
        DAT[vector[0]] = dat
        ST[vector[0]] = max(RT[vector[1]], dat)
        FT[vector[0]] = ST[vector[0]] + computation_cost[vector[0]]
        RT[vector[1]] = FT[vector[0]]
    S_list = [(i,FT[i]) for i in range(nodes)]
    S_list = sorted(S_list, reverse = True, key = lambda x:x[1])
	
    temp = [[] for i in range(processors)]
    for vec in task_list:
        temp[vec[1]].append((ST[vec[0]], FT[vec[0]]))
    idle_time = [[] for i in range(processors)]
    for i in range(processors):
        start, end = 0, 1000
        for pair in temp[i]:
            if pair[0] > start:
                idle_time[i].append((start, pair[0] - 1))
            start = pair[1] + 1
        idle_time[i].append((start, end))	
    return FPRED, S_list, idle_time


def duplicate_process(ind):
    FPRED, S_list, idle_time = schedule_length(ind)
    
    for task in S_list:
        proc = 0
        proc_fpred = 0
        for each in ind:
            if each[0] == task[0]:
                proc = each[1]
            if each[0] == FPRED[task[0]]:
                proc_fpred = each[1]
        if proc != proc_fpred:
            for slot in idle_time[proc]:
                if FPRED[task[0]] is not None:
                    if slot[1] - slot[0] >= computation_cost[FPRED[task[0]]] and slot[0] + computation_cost[FPRED[task[0]]] < FT[FPRED[task[0]]]:
                        ind.append((FPRED[task[0]],proc))
                        
            

def create_individual():
    ind = [(i, random.randint(0, processors - 1)) for i in range(nodes)]
    #random.shuffle(ind)
    duplicate_process(ind)
    return ind

def generate_initial_population():
    pop = []
    no_pop = 100
    while no_pop:
        ind = create_individual()
        if not invalid(ind):
            pop.append(create_individual())
            no_pop -= 1
    return pop

if __name__ == "__main__":
    take_graph_input()
    population = generate_initial_population()
    solution = GA(population)
    print(solution)
    FPRED, S_list, idle_time = schedule_length(solution)
    print('schedule length = ',S_list[0][1])
    
	
	