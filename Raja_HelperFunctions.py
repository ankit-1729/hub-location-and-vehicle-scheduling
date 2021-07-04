import copy,random,math,numpy,sys

population_size=100                              
max_num_of_generations=50
discount_factor=0.5
unit_transportation_cost=1.2
mutation_probability=0.3
p=6
D=3
Ef=1.096
Ee=0.772
weight_limit=40
QRP=0.4
shelf_life=2600
vehicle_speed=1

n=25
tournament_size=4

Large_Int=pow(10,12)


def get_the_flow_of_products_list(W):

    flow_of_products_list=[]

    for i in range(n):
        total_outward_flow_for_the_ith_node=0

        for j in range(n):
            total_outward_flow_for_the_ith_node+=W[i][j]
        
        total_inward_flow_for_the_ith_node=0

        for j in range(n):
            total_inward_flow_for_the_ith_node+=W[j][i]

        total_flow_for_the_ith_node=total_outward_flow_for_the_ith_node+total_inward_flow_for_the_ith_node
        flow_of_products_list.append((total_flow_for_the_ith_node,i))
    
    return flow_of_products_list


def randomly_generate_node_allocation_array(potential_nodes_as_hubs):

    chosen_hubs=[]
    is_node_chosen_as_hub=[False for _ in range(n)]

    for _ in range(p):
        chosen_node=random.choice(potential_nodes_as_hubs)
        is_node_chosen_as_hub[chosen_node]=True
        chosen_hubs.append(chosen_node)
        potential_nodes_as_hubs.remove(chosen_node)

    node_allocation_array=[]

    for i in range(n):

        if is_node_chosen_as_hub[i]==True:
            node_allocation_array.append(i)
        else:
            node_allocation_array.append(random.choice(chosen_hubs))

    return node_allocation_array


def init_population(W,C):

    flow_of_products_list=get_the_flow_of_products_list(W)
    flow_of_products_list.sort(reverse=True)

    population_size_1=math.floor(0.75*population_size)
    population_size_2=population_size-population_size_1

    P0=[]
    potential_nodes_as_hubs=[]

    for i in range(math.floor((2/3)*n)):
        potential_nodes_as_hubs.append(flow_of_products_list[i][1])

    for i in range(population_size_1):
        node_allocation_array=randomly_generate_node_allocation_array(copy.deepcopy(potential_nodes_as_hubs))
        vehicle_assign_array=[]

        for _ in range(n):
            vehicle_assign_array.append(random.randint(0,D-1))

        vehicle_sequence_array=numpy.random.permutation([i for i in range(n)]).tolist()
        P0.append([node_allocation_array,vehicle_assign_array,vehicle_sequence_array])

    potential_nodes_as_hubs=[i for i in range(n)]

    for i in range(population_size_2):
        node_allocation_array=randomly_generate_node_allocation_array(copy.deepcopy(potential_nodes_as_hubs))
        vehicle_assign_array=[]

        for _ in range(n):
            vehicle_assign_array.append(random.randint(0,D-1))

        vehicle_sequence_array=numpy.random.permutation([i for i in range(n)]).tolist()
        P0.append([node_allocation_array,vehicle_assign_array,vehicle_sequence_array])

    return P0
