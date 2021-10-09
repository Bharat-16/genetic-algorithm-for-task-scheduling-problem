import random
import copy
import numpy as np

processors = 3
nodes = 3  # number of tasks
graph = []  # (node,edge cost) , mapping from parent to child
graph_par = []  # mapping from child to parent
DAT = []  # DAT for each node
FPRED = []  # favorite pred for each node
root_node = 0
computation_cost = []
no_iter = 100
no_pop = 10  # number of individuals in the population
crossover_prob = 0.5
mutation_prob = 0.03
ST = []  # start time of each node
FT = []  # finish time of each node
a = 10  # some constant, used to calculate fitness value
Total_cost = -1  # total cost of the critical path
critical_path = []


# function to take user input 
def take_graph_input():
    global nodes, computation_cost, graph
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


# function to reuse idle time 
def test_slots(individual):
    LT = individual[1]
    # creating idle time list for each processor 
    temp = [[] for i in range(processors)]
    for i in range(nodes):
        temp[individual[0][i]].append((ST[i], FT[i]))
    idle_time = [[] for i in range(processors)]
    for i in range(processors):
        start, end = 0, 1000
        for pair in temp[i]:
            if pair[0] > start:
                idle_time[i].append((start, pair[0] - 1))
            start = pair[1] + 1
        idle_time[i].append((start, end))
    # finding idle time slot and inserting 
    S_idle_time = 0
    E_idle_time = 0
    for t_i in LT:
        Min_St = 10000
        for processor in range(processors):
            if individual[0][t_i] == processor:
                this_St = ST[t_i]
                Min_St = min(Min_St, this_St)
                for pair in idle_time[processor]:
                    if E_idle_time - S_idle_time >= computation_cost[t_i]:
                        pass


# part of function to find critical path in the graph 
def bfs(node, current_list, cost):
    global Total_cost, critical_path
    flag = False
    for pair in graph[node]:
        flag = True
        bfs(pair[0], copy.deepcopy(current_list) + [pair[0]], cost + pair[1])
    if cost > Total_cost and not flag:
        Total_cost = cost
        critical_path = current_list


def get_critical_path():
    bfs(root_node, [root_node], computation_cost[root_node])
    return critical_path


# function to get the schedule length of the given schedule 
def schedule_length(task_list):
    global ST, FT, DAT, FPRED
    RT = [0 for i in range(processors)]
    ST = [0 for i in range(nodes)]
    FT = [0 for i in range(nodes)]
    DAT = [0 for i in range(nodes)]
    FPRED = [0 for i in range(nodes)]

    for t_i in task_list[1]:
        for j in range(processors):
            if task_list[0][t_i] == j:
                dat = 0
                for k in graph_par[t_i]:
                    if task_list[k[0]][0] == j:  # on same processor
                        if dat < FT[k[0]]:
                            dat = FT[k[0]]
                            FPRED[t_i] = k[0]
                    else:
                        if dat < FT[k[0]] + k[1]:
                            dat = FT[k[0]] + k[1]
                            FPRED[t_i] = k[0]
                DAT[t_i] = dat
                ST[t_i] = max(RT[j], dat)
                FT[t_i] = ST[t_i] + computation_cost[t_i]
                RT[j] = FT[t_i]
    return max(FT)


# function to get the topological order of the graph
def get_topologicalOrder():
    global root_node
    inD = [0 for i in range(nodes)]
    visited = [False for i in range(nodes)]
    order = []
    for l in graph:
        for pair in l:
            inD[pair[0]] += 1
    # find the root node
    root_node = np.argmin(inD)
    order.append(root_node)
    visited[root_node] = True
    cnt = 1
    l = [root_node]
    while cnt != nodes:
        l2 = []
        for node in l:
            for pair in graph[node]:
                inD[pair[0]] -= 1
        for i in range(nodes):
            if inD[i] == 0 and not visited[i]:
                l2.append(i)
                visited[i] = True
                cnt += 1
        order += l2
        l = copy.deepcopy(l2)
    return order


# function to create individuals for the population
def create_individual():
    ind = [[0 for i in range(nodes)], []]
    ind[1] += get_topologicalOrder()
    for i in range(nodes):
        ind[0][i] = random.randint(0, processors - 1)
    return ind


# function to generate initial population 
def generate_initial_population():
    pop = []
    for i in range(no_pop):
        pop.append(create_individual())
    return pop


# function to apply selection operation 
def selection_operation(init_pop_):
    pos1 = random.randint(0, len(init_pop_) - 1)
    pos2 = random.randint(0, len(init_pop_) - 1)
    return init_pop_[pos1], init_pop_[pos2]


# function to apply crossover operation 
def crossover_operation(ind1, ind2):
    c1, c2 = ind1, ind2
    if random.random() >= crossover_prob:
        point = random.randint(0, nodes - 1)
        c1 = ind1[0: point + 1] + ind2[point + 1:]
        c2 = ind2[0: point + 1] + ind1[point + 1:]
    return c1, c2


# function to apply mutation operation 
def mutation_operation(ind):
    for i in range(nodes):
        if random.random() >= mutation_prob:
            ind[0][i] = random.randint(0, processors - 1)
    return ind


# function to calculate the fitness value of the given individual
def get_fitness_value(ind):
    s_length = schedule_length(ind)
    return a / s_length


# function to implement load balancing 
def load_balancing(l):
    # schedule length function will calculate the start time and finish time according to the individual
    best_indi = l[0]
    load_balance = schedule_length(l[0])
    Avg = 0
    for i in range(nodes):
        Avg += FT[i] - ST[i]
    Avg /= processors
    load_balance /= Avg
    for i in range(1, len(l)):
        lb = schedule_length(l[i])
        Avg = 0
        for i in range(nodes):
            Avg += FT[i] - ST[i]
        Avg /= processors
        lb /= Avg
        if lb < load_balance:
            load_balance = lb
            best_indi = l[i]

    return best_indi


# function to prioritise nodes of the critical path
def reschedule_CPNs(individual):
    schedule_length(individual)  # call to update the value of FPRED
    CPNs = get_critical_path()
    for node in CPNs:
        VIP = FPRED[node]
        processor_fpred = individual[0][VIP]
        individual[0][node] = processor_fpred


# genetic algorithm main function 
def GA(init_pop):
    global no_iter
    while no_iter:
        no_iter -= 1
        first, second = True, True
        # select two individuals
        ind1, ind2 = selection_operation(copy.deepcopy(init_pop))
        # apply crossover
        ind1, ind2 = crossover_operation(ind1, ind2)
        # apply mutation
        ind1 = mutation_operation(ind1)
        ind2 = mutation_operation(ind2)
        # assign priority to critical path nodes
        reschedule_CPNs(ind1)
        reschedule_CPNs(ind2)
        # replace the 2 individual in the population which has the least fitness value from the new individuals
        ind1_fitness = get_fitness_value(ind1)
        ind2_fitness = get_fitness_value(ind2)
        fitness_pop = []
        # try replacing the first smallest fitness individual
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
        # try removing the second smallest individual
        pos = np.argmin(fitness_pop)
        if fitness_pop[pos] < ind1_fitness and first:
            init_pop[pos] = ind1
        elif fitness_pop[pos] < ind2_fitness and second:
            init_pop[pos] = ind2
    # after the iteration for GA are complete extract the individual with the largest fitness value 
    fitness_pop = []
    for val in init_pop:
        fitness_pop.append(get_fitness_value(val))
    best_fitness = fitness_pop[np.argmax(fitness_pop)]
    l = []
    for i in range(len(fitness_pop)):
        if fitness_pop[i] == best_fitness:
            l.append(init_pop[i])
    # if there is only one individual with the highest fitness value return it otherwise go for load balancing 
    if len(l) == 1:
        return l[0]
    else:
        return load_balancing(l)


if __name__ == "__main__":
    take_graph_input()
    population = generate_initial_population()
    solution = GA(population)
    print(solution)