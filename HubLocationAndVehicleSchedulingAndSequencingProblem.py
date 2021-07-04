import copy,random,math,numpy,sys,timeit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Dheeraj_HelperFunctions import compute_the_objective_function_values_of_the_solution,compute_total_transportation_cost,compute_total_responsiveness_of_the_network,compute_total_carbon_emission,do_fast_non_dominated_sort,if_solution_i_dominates_solution_j,do_crowding_distance_assignment,update_the_crowding_distance_value_for_the_objective_function
from Raja_HelperFunctions import init_population,get_the_flow_of_products_list,randomly_generate_node_allocation_array

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


def print_the_solution(solution):

    node_allocation_array=solution[0]
    vehicle_assign_array=solution[1]
    vehicle_sequence_array=solution[2]
    n=len(node_allocation_array)

    print('The solution components are....')
    print()

    print('Node Allocation Array = ',node_allocation_array)
    print('Vehicle Assign Array = ',vehicle_assign_array)
    print('Vehicle Sequence Array = ',vehicle_sequence_array)
    print()

    print('Node allocation....')
    print()

    hubs=[]

    for i in range(n):

        if node_allocation_array[i]==i:
            hubs.append(i)

    print('Hubs = ',end='')
    print(' '.join(str(i) for i in hubs))

    hubs_to_non_hubs_map=dict()

    for i in range(p):
        print(str(hubs[i])+' -> ',end='')
        non_hubs_allocated_to_hub_hubs_i=[]

        for j in range(n):

            if j!=hubs[i] and node_allocation_array[j]==hubs[i]:
                non_hubs_allocated_to_hub_hubs_i.append(j)
                print(j,end=' ')

        print()

        hubs_to_non_hubs_map[hubs[i]]=non_hubs_allocated_to_hub_hubs_i

    
    print()
    print('Vehicle Assignment and Sequencing....')
    print()

    vehicle_sequence_map=dict()

    for i in range(n):
        vehicle_sequence_map[vehicle_sequence_array[i]]=i

    for i in range(p):
        print('hub index = '+str(hubs[i]))
        non_hubs_allocated_to_hub_hubs_i=hubs_to_non_hubs_map[hubs[i]]
        vehicle_to_non_hubs_map=[[] for _ in range(D)]

        for j in range(len(non_hubs_allocated_to_hub_hubs_i)):
            vehicle_to_non_hubs_map[vehicle_assign_array[non_hubs_allocated_to_hub_hubs_i[j]]].append(non_hubs_allocated_to_hub_hubs_i[j])

        for d in range(D):
            print('vehicle index = '+str(d)+' ->  ',end='')
            cust_nodes_served_by_vehicle_d_at_hub_i_and_cust_nodes_index_tuple=[]

            for k in range(len(vehicle_to_non_hubs_map[d])):
                cust_nodes_served_by_vehicle_d_at_hub_i_and_cust_nodes_index_tuple.append((vehicle_sequence_map[vehicle_to_non_hubs_map[d][k]],vehicle_to_non_hubs_map[d][k]))

            cust_nodes_served_by_vehicle_d_at_hub_i_and_cust_nodes_index_tuple.sort()
            
            for k in range(len(cust_nodes_served_by_vehicle_d_at_hub_i_and_cust_nodes_index_tuple)):
                print(cust_nodes_served_by_vehicle_d_at_hub_i_and_cust_nodes_index_tuple[k][1],end=' ')

            print()

        print()


def plot_the_pareto_fronts(pareto_fronts,objective_function_values_of_the_solution_R):

    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    L=len(pareto_fronts)

    for i in range(L):

        total_transportation_cost=[objective_function_values_of_the_solution_R[j][0] for j in pareto_fronts[i]]
        total_responsiveness=[objective_function_values_of_the_solution_R[j][1] for j in pareto_fronts[i]]
        total_carbon_emission=[objective_function_values_of_the_solution_R[j][2] for j in pareto_fronts[i]]

        if i==0:
            ax.scatter(total_transportation_cost,total_responsiveness,total_carbon_emission,c='r',marker='x')
        elif i==1:
            ax.scatter(total_transportation_cost,total_responsiveness,total_carbon_emission,c='g',marker='o')
        else:
            ax.scatter(total_transportation_cost,total_responsiveness,total_carbon_emission,c='b',marker='^')

    ax.set_xlabel('Transportation Cost')
    ax.set_ylabel('Responsiveness')
    ax.set_zlabel('Carbon Emission')

    plt.show()


def print_the_objective_values_of_the_solution(objective_function_values):

    print('Printing the values of the objective functions....')
    print()

    print('Total transportation cost = ',objective_function_values[0])
    print('Total responsiveness of the network = ',objective_function_values[1])
    print('Total carbon emission = ',objective_function_values[2])
    print()


def read_input_from_datafile(file_path,W,C):

    with open(file_path,'r') as f:
        for entry in f:
            entry=entry.split()
            i=int(entry[0])-1
            j=int(entry[1])-1
            W[i][j]=int(entry[2][:len(entry[2])-1])
            C[i][j]=float(entry[3])


def compute_the_objective_function_values_of_all_the_solutions(P,W,C):

    objective_function_values_of_the_solution=[]

    for i in range(population_size):
        objective_function_values_of_the_solution.append(compute_the_objective_function_values_of_the_solution(P[i],W,C))
        
    return objective_function_values_of_the_solution


def compute_the_min_max_values_of_all_the_objective_functions(objective_function_values_of_the_solution,num_of_solutions):

    min_max_total_transportation_cost=[Large_Int,0]
    min_max_total_responsiveness_of_the_network=[n,0]
    min_max_total_carbon_emission=[Large_Int,0]

    for i in range(num_of_solutions):
        total_transportation_cost=objective_function_values_of_the_solution[i][0]
        total_responsiveness_of_the_network=objective_function_values_of_the_solution[i][1]
        total_carbon_emission=objective_function_values_of_the_solution[i][2]

        min_max_total_transportation_cost[0]=min(min_max_total_transportation_cost[0],total_transportation_cost)
        min_max_total_transportation_cost[1]=max(min_max_total_transportation_cost[1],total_transportation_cost)
        min_max_total_responsiveness_of_the_network[0]=min(min_max_total_responsiveness_of_the_network[0],total_responsiveness_of_the_network)
        min_max_total_responsiveness_of_the_network[1]=max(min_max_total_responsiveness_of_the_network[1],total_responsiveness_of_the_network)
        min_max_total_carbon_emission[0]=min(min_max_total_carbon_emission[0],total_carbon_emission)
        min_max_total_carbon_emission[1]=max(min_max_total_carbon_emission[1],total_carbon_emission)

    return [min_max_total_transportation_cost,min_max_total_responsiveness_of_the_network,min_max_total_carbon_emission]


def get_the_updated_min_max_values_of_the_merged_population(min_max_values_of_the_objective_function_P,min_max_values_of_the_objective_function_Q):

	min_max_values_of_the_objective_function_R=copy.deepcopy(min_max_values_of_the_objective_function_P)

	min_max_values_of_the_objective_function_R[0][0]=min(min_max_values_of_the_objective_function_R[0][0],min_max_values_of_the_objective_function_Q[0][0])
	min_max_values_of_the_objective_function_R[0][1]=max(min_max_values_of_the_objective_function_R[0][1],min_max_values_of_the_objective_function_Q[0][1])
	min_max_values_of_the_objective_function_R[1][0]=min(min_max_values_of_the_objective_function_R[1][0],min_max_values_of_the_objective_function_Q[1][0])
	min_max_values_of_the_objective_function_R[1][1]=max(min_max_values_of_the_objective_function_R[1][1],min_max_values_of_the_objective_function_Q[1][1])
	min_max_values_of_the_objective_function_R[2][0]=min(min_max_values_of_the_objective_function_R[2][0],min_max_values_of_the_objective_function_Q[2][0])
	min_max_values_of_the_objective_function_R[2][1]=max(min_max_values_of_the_objective_function_R[2][1],min_max_values_of_the_objective_function_Q[2][1])
	return min_max_values_of_the_objective_function_R


def compute_the_normalized_fitness_values_of_all_the_solutions(objective_function_values_of_the_solution,min_max_values_of_the_objective_function,num_of_solutions):

    normalized_fitness_value_of_the_solution=[None for _ in range(num_of_solutions)]
    min_max_total_transportation_cost=min_max_values_of_the_objective_function[0]
    min_max_total_responsiveness_of_the_network=min_max_values_of_the_objective_function[1]
    min_max_total_carbon_emission=min_max_values_of_the_objective_function[2]

    for i in range(num_of_solutions):
        total_transportation_cost=objective_function_values_of_the_solution[i][0]
        total_responsiveness_of_the_network=objective_function_values_of_the_solution[i][1]
        total_carbon_emission=objective_function_values_of_the_solution[i][2]

        if min_max_total_transportation_cost[1]==min_max_total_transportation_cost[0]:
            normalized_total_transportation_cost=1
        else:
            normalized_total_transportation_cost=1-((total_transportation_cost-min_max_total_transportation_cost[0])/(min_max_total_transportation_cost[1]-min_max_total_transportation_cost[0]))

        if min_max_total_responsiveness_of_the_network[1]==min_max_total_responsiveness_of_the_network[0]:
            normalized_total_responsiveness_of_the_network=1
        else:
            normalized_total_responsiveness_of_the_network=(total_responsiveness_of_the_network-min_max_total_responsiveness_of_the_network[0])/(min_max_total_responsiveness_of_the_network[1]-min_max_total_responsiveness_of_the_network[0])

        if min_max_total_carbon_emission[1]==min_max_total_carbon_emission[0]:
            normalized_total_carbon_emission=1
        else:
            normalized_total_carbon_emission=1-(total_carbon_emission-min_max_total_carbon_emission[0])/(min_max_total_carbon_emission[1]-min_max_total_carbon_emission[0])

        normalized_fitness_value_of_the_solution[i]=(normalized_total_transportation_cost+normalized_total_responsiveness_of_the_network+normalized_total_carbon_emission)/3

    return normalized_fitness_value_of_the_solution


def get_the_solution_index_of_the_best_solution(normalized_fitness_value_of_the_solution):

    max_normalized_fitness_value=normalized_fitness_value_of_the_solution[0]
    best_solution_index=0

    for i in range(1,population_size):

    	if normalized_fitness_value_of_the_solution[i]>max_normalized_fitness_value:
    		max_normalized_fitness_value=normalized_fitness_value_of_the_solution[i]
    		best_solution_index=i

    return best_solution_index


def get_the_overall_fittest_solution(best_solution_of_the_generation,objective_function_values_of_the_best_solution_of_the_generation):

    min_max_values_of_the_objective_function=compute_the_min_max_values_of_all_the_objective_functions(objective_function_values_of_the_best_solution_of_the_generation,max_num_of_generations)
    normalized_fitness_values_of_the_solution=compute_the_normalized_fitness_values_of_all_the_solutions(objective_function_values_of_the_best_solution_of_the_generation,min_max_values_of_the_objective_function,max_num_of_generations)
    
    max_normalized_fitness_value=normalized_fitness_values_of_the_solution[0]
    overall_fittest_solution_index=0

    for i in range(1,max_num_of_generations):

        if normalized_fitness_values_of_the_solution[i]>max_normalized_fitness_value:
            max_normalized_fitness_value=normalized_fitness_values_of_the_solution[i]
            overall_fittest_solution_index=i

    overall_fittest_solution=best_solution_of_the_generation[overall_fittest_solution_index]
    objective_function_values_of_the_overall_fittest_solution=objective_function_values_of_the_best_solution_of_the_generation[overall_fittest_solution_index]
    return overall_fittest_solution,objective_function_values_of_the_overall_fittest_solution


def tournament_selection(P,normalized_fitness_value_of_the_solution,W,C):

    participant_parents_index=random.sample([i for i in range(population_size)],tournament_size)
    parent_1_index=participant_parents_index[0]
    parent_2_index=participant_parents_index[1]
    parent_3_index=participant_parents_index[2]
    parent_4_index=participant_parents_index[3]

    if normalized_fitness_value_of_the_solution[parent_1_index]>=normalized_fitness_value_of_the_solution[parent_2_index]:
    	winner_1_index=parent_1_index
    else:
    	winner_1_index=parent_2_index

    if normalized_fitness_value_of_the_solution[parent_3_index]>=normalized_fitness_value_of_the_solution[parent_4_index]:
    	winner_2_index=parent_3_index
    else:
    	winner_2_index=parent_4_index

    selected_parent=None

    if normalized_fitness_value_of_the_solution[winner_1_index]>=normalized_fitness_value_of_the_solution[winner_2_index]:
    	selected_parent=P[winner_1_index]
    else:
    	selected_parent=P[winner_2_index]

    return selected_parent


def repair_the_vehicle_sequence_array(vehicle_sequence_array):

    repeated_hubs_index=set()
    is_node_present=[False for _ in range(n)]

    for i in range(n):
        node=vehicle_sequence_array[i]  
        
        if is_node_present[node]==False:
            is_node_present[node]=True
        else:
            repeated_hubs_index.add(i)

    nodes_absent_in_vehicle_seq_array=[]

    for i in range(n):

        if is_node_present[i]==False:
            nodes_absent_in_vehicle_seq_array.append(i)

    i=0

    for index in repeated_hubs_index:
        vehicle_sequence_array[index]=nodes_absent_in_vehicle_seq_array[i]
        i+=1


def generate_offsprings_from_parents(parent_1,parent_2):

    node_allocation_array_1=parent_1[0]
    node_allocation_array_2=parent_2[0]

    hubs_of_parent_1=set()

    for i in range(n):
        
        if node_allocation_array_1[i]==i:
            hubs_of_parent_1.add(i)
    
    hubs_of_parent_2=set()

    for i in range(n):

        if node_allocation_array_2[i]==i:
            hubs_of_parent_2.add(i)

    hubs_common_in_both=hubs_of_parent_1.intersection(hubs_of_parent_2)

    offspring_1=[[] for _ in range(3)]
    offspring_2=[[] for _ in range(3)]
    offspring_1[0]=[None for _ in range(n)]
    offspring_2[0]=[None for _ in range(n)]

    hubs_of_offspring_1=[]
    hubs_of_offspring_2=[]

    for common_hub in hubs_common_in_both:
        offspring_1[0][common_hub]=common_hub
        offspring_2[0][common_hub]=common_hub
        hubs_of_offspring_1.append(common_hub)
        hubs_of_offspring_2.append(common_hub)

    hubs_not_common_to_both=hubs_of_parent_1.union(hubs_of_parent_2).difference(hubs_common_in_both)

    random_nodes_chosen_as_remaining_hubs_in_offspring_1=set(random.sample(hubs_not_common_to_both,p-len(hubs_common_in_both)))
    random_nodes_chosen_as_remaining_hubs_in_offspring_2=hubs_not_common_to_both.difference(random_nodes_chosen_as_remaining_hubs_in_offspring_1)
    
    for chosen_hub in random_nodes_chosen_as_remaining_hubs_in_offspring_1:
        offspring_1[0][chosen_hub]=chosen_hub
        hubs_of_offspring_1.append(chosen_hub)
    
    for chosen_hub in random_nodes_chosen_as_remaining_hubs_in_offspring_2:
        offspring_2[0][chosen_hub]=chosen_hub
        hubs_of_offspring_2.append(chosen_hub)
    
    for i in range(n):

        if offspring_1[0][i]==i:
            continue
        else:
            hub_allocated_to_node_i_in_parent_1=node_allocation_array_1[i]
            hub_allocated_to_node_i_in_parent_2=node_allocation_array_2[i]

            if offspring_1[0][hub_allocated_to_node_i_in_parent_1]==hub_allocated_to_node_i_in_parent_1 and offspring_1[0][hub_allocated_to_node_i_in_parent_2]==hub_allocated_to_node_i_in_parent_2:
                if random.randint(0,1)==0:
                    offspring_1[0][i]=hub_allocated_to_node_i_in_parent_1
                else:
                    offspring_1[0][i]=hub_allocated_to_node_i_in_parent_2

            elif offspring_1[0][hub_allocated_to_node_i_in_parent_1]==hub_allocated_to_node_i_in_parent_1 and offspring_1[0][hub_allocated_to_node_i_in_parent_2]!=hub_allocated_to_node_i_in_parent_2:
                offspring_1[0][i]=hub_allocated_to_node_i_in_parent_1

            elif offspring_1[0][hub_allocated_to_node_i_in_parent_1]!=hub_allocated_to_node_i_in_parent_1 and offspring_1[0][hub_allocated_to_node_i_in_parent_2]==hub_allocated_to_node_i_in_parent_2:
                offspring_1[0][i]=hub_allocated_to_node_i_in_parent_2

            else:
                randomly_chosen_hub_for_node_i=hubs_of_offspring_1[random.randint(0,p-1)]
                offspring_1[0][i]=randomly_chosen_hub_for_node_i
        
    for i in range(n):

        if offspring_2[0][i]==i:
            continue
        else:
            hub_allocated_to_node_i_in_parent_1=node_allocation_array_1[i]
            hub_allocated_to_node_i_in_parent_2=node_allocation_array_2[i]

            if offspring_2[0][hub_allocated_to_node_i_in_parent_1]==hub_allocated_to_node_i_in_parent_1 and offspring_2[0][hub_allocated_to_node_i_in_parent_2]==hub_allocated_to_node_i_in_parent_2:
                if random.randint(0,1)==0:
                    offspring_2[0][i]=hub_allocated_to_node_i_in_parent_1
                else:
                    offspring_2[0][i]=hub_allocated_to_node_i_in_parent_2

            elif offspring_2[0][hub_allocated_to_node_i_in_parent_1]==hub_allocated_to_node_i_in_parent_1 and offspring_2[0][hub_allocated_to_node_i_in_parent_2]!=hub_allocated_to_node_i_in_parent_2:
                offspring_2[0][i]=hub_allocated_to_node_i_in_parent_1

            elif offspring_2[0][hub_allocated_to_node_i_in_parent_1]!=hub_allocated_to_node_i_in_parent_1 and offspring_2[0][hub_allocated_to_node_i_in_parent_2]==hub_allocated_to_node_i_in_parent_2:
                offspring_2[0][i]=hub_allocated_to_node_i_in_parent_2

            else:
                randomly_chosen_hub_for_node_i=hubs_of_offspring_2[random.randint(0,p-1)]
                offspring_2[0][i]=randomly_chosen_hub_for_node_i

    vehicle_assign_array_1=parent_1[1]
    vehicle_assign_array_2=parent_2[1]
    vehicle_sequence_array_1=parent_1[2]
    vehicle_sequence_array_2=parent_2[2]

    crossover_point=random.randint(0,n)

    offspring_1[1]=copy.deepcopy(vehicle_assign_array_1[0:crossover_point])+copy.deepcopy(vehicle_assign_array_2[crossover_point:n])
    offspring_2[1]=copy.deepcopy(vehicle_assign_array_2[0:crossover_point])+copy.deepcopy(vehicle_assign_array_1[crossover_point:n])
    offspring_1[2]=copy.deepcopy(vehicle_sequence_array_1[0:crossover_point])+copy.deepcopy(vehicle_sequence_array_2[crossover_point:n])
    offspring_2[2]=copy.deepcopy(vehicle_sequence_array_2[0:crossover_point])+copy.deepcopy(vehicle_sequence_array_1[crossover_point:n])

    repair_the_vehicle_sequence_array(offspring_1[2])
    repair_the_vehicle_sequence_array(offspring_2[2])

    return offspring_1,offspring_2


def apply_shift_operator(node_allocation_array):

    if p<=1 or n<=2:
        return
    else:
        non_hub_nodes=set()
        hub_nodes=set()

        for i in range(n):

            if node_allocation_array[i]==i:
                hub_nodes.add(i)
            else:
                non_hub_nodes.add(i)

        randomly_selected_non_hub_node=random.sample(non_hub_nodes,1)[0]
        hub_allocated_to_the_randomly_selected_non_hub_node=node_allocation_array[randomly_selected_non_hub_node]
        hub_nodes.discard(hub_allocated_to_the_randomly_selected_non_hub_node)
        randomly_selected_hub_node_for_reallocation=random.sample(hub_nodes,1)[0]
        node_allocation_array[randomly_selected_non_hub_node]=randomly_selected_hub_node_for_reallocation
        return


def apply_exchange_operator(node_allocation_array):

    non_hub_nodes=set()

    for i in range(n):

        if node_allocation_array[i]!=i:
            non_hub_nodes.add(i)
        
    randomly_selected_first_non_hub_node=random.sample(non_hub_nodes,1)[0]
    hub_allocated_to_randomly_selected_first_non_hub_node=node_allocation_array[randomly_selected_first_non_hub_node]
    non_hub_nodes.discard(randomly_selected_first_non_hub_node)
    randomly_selected_second_non_hub_node=random.sample(non_hub_nodes,1)[0]
    
    while node_allocation_array[randomly_selected_second_non_hub_node]==hub_allocated_to_randomly_selected_first_non_hub_node:
        randomly_selected_second_non_hub_node=random.sample(non_hub_nodes,1)[0]

    hub_allocated_to_randomly_selected_second_non_hub_node=node_allocation_array[randomly_selected_second_non_hub_node]
    node_allocation_array[randomly_selected_first_non_hub_node]=hub_allocated_to_randomly_selected_second_non_hub_node
    node_allocation_array[randomly_selected_second_non_hub_node]=hub_allocated_to_randomly_selected_first_non_hub_node
    return


def apply_mutation_operator(node_allocation_array):

    apply_shift_operator(node_allocation_array)
    apply_exchange_operator(node_allocation_array)
    return


def generate_offspring_population(P,normalized_fitness_value_of_the_solution,W,C):

    Q=[]

    for _ in range(0,population_size,2):
        parent_1=tournament_selection(P,normalized_fitness_value_of_the_solution,W,C)
        parent_2=tournament_selection(P,normalized_fitness_value_of_the_solution,W,C)

        offspring_1,offspring_2=generate_offsprings_from_parents(parent_1,parent_2)

        if random.random()<mutation_probability:
            apply_mutation_operator(offspring_1[0])

        if random.random()<mutation_probability:
            apply_mutation_operator(offspring_2[0])

        Q.append(offspring_1)
        Q.append(offspring_2)

    return Q


def add_the_front_to_the_new_population(P_new,R,pareto_front):

	for i in pareto_front:
		P_new.append(R[i])


def add_the_remaining_solutions_of_the_curr_front_to_the_new_population(P_new,R,crowding_distance_and_solution_index_tuple,new_population_size):

    num_of_solutions_to_add=population_size-new_population_size

    for i in range(num_of_solutions_to_add):
    	solution_index=crowding_distance_and_solution_index_tuple[i][1]
    	P_new.append(R[solution_index])


def apply_NSGA_II_algorithm(W,C):

    P=init_population(W,C)
    objective_function_values_of_the_solution_P=compute_the_objective_function_values_of_all_the_solutions(P,W,C)
    min_max_values_of_the_objective_function_P=compute_the_min_max_values_of_all_the_objective_functions(objective_function_values_of_the_solution_P,population_size)
    normalized_fitness_value_of_the_solution_P=compute_the_normalized_fitness_values_of_all_the_solutions(objective_function_values_of_the_solution_P,min_max_values_of_the_objective_function_P,population_size)

    best_solution_index=get_the_solution_index_of_the_best_solution(normalized_fitness_value_of_the_solution_P)
    
    print()
    print('Printing the best solution in the generation 0....')
    print()
    print_the_solution(P[best_solution_index])
    print()
    print_the_objective_values_of_the_solution(objective_function_values_of_the_solution_P[best_solution_index])

    best_solution_of_the_generation=[]
    best_solution_of_the_generation.append(P[best_solution_index])

    objective_function_values_of_the_best_solution_of_the_generation=[]
    objective_function_values_of_the_best_solution_of_the_generation.append(objective_function_values_of_the_solution_P[best_solution_index])

    flag=False

    for k in range(max_num_of_generations):

        Q=generate_offspring_population(P,normalized_fitness_value_of_the_solution_P,W,C)

        objective_function_values_of_the_solution_Q=compute_the_objective_function_values_of_all_the_solutions(Q,W,C)
        min_max_values_of_the_objective_function_Q=compute_the_min_max_values_of_all_the_objective_functions(objective_function_values_of_the_solution_Q,population_size)
        
        objective_function_values_of_the_solution_R=objective_function_values_of_the_solution_P+objective_function_values_of_the_solution_Q
        min_max_values_of_the_objective_function_R=get_the_updated_min_max_values_of_the_merged_population(min_max_values_of_the_objective_function_P,min_max_values_of_the_objective_function_Q)
        
        R=P+Q
        pareto_fronts=do_fast_non_dominated_sort(objective_function_values_of_the_solution_R,W,C)

        if flag==False:
            plot_the_pareto_fronts(pareto_fronts,objective_function_values_of_the_solution_R)
            flag=True
        
        P_new=[]
        new_population_size=0
        i=0

        while new_population_size+len(pareto_fronts[i])<=population_size:
            add_the_front_to_the_new_population(P_new,R,pareto_fronts[i])
            new_population_size+=len(pareto_fronts[i])
            i+=1
        
        if new_population_size<population_size:
            crowding_distance_and_solution_tuple=do_crowding_distance_assignment(R,pareto_fronts[i],objective_function_values_of_the_solution_R,min_max_values_of_the_objective_function_R)
            crowding_distance_and_solution_tuple.sort(reverse=True)
            add_the_remaining_solutions_of_the_curr_front_to_the_new_population(P_new,R,crowding_distance_and_solution_tuple,new_population_size)
            
        P=copy.deepcopy(P_new)
        
        objective_function_values_of_the_solution_P=compute_the_objective_function_values_of_all_the_solutions(P,W,C)
        min_max_values_of_the_objective_function_P=compute_the_min_max_values_of_all_the_objective_functions(objective_function_values_of_the_solution_P,population_size)
        normalized_fitness_value_of_the_solution_P=compute_the_normalized_fitness_values_of_all_the_solutions(objective_function_values_of_the_solution_P,min_max_values_of_the_objective_function_P,population_size)

        best_solution_index=get_the_solution_index_of_the_best_solution(normalized_fitness_value_of_the_solution_P)
        
        #print()
        #print('Printing the best solution in the generation '+str(k+1)+'....')
        #print()
        #print_the_solution(P[best_solution_index])
        #print()
        #print_the_objective_values_of_the_solution(objective_function_values_of_the_solution_P[best_solution_index])

        best_solution_of_the_generation.append(P[best_solution_index])
        objective_function_values_of_the_best_solution_of_the_generation.append(objective_function_values_of_the_solution_P[best_solution_index])

    overall_fittest_solution,objective_values_of_the_overall_fittest_solution=get_the_overall_fittest_solution(best_solution_of_the_generation,objective_function_values_of_the_best_solution_of_the_generation)

    print()
    print('Printing the overall fittest solution of all the generations....')
    print()
    print_the_solution(overall_fittest_solution)
    print()
    print_the_objective_values_of_the_solution(objective_values_of_the_overall_fittest_solution)

    print('End of the Algorithm')


def main():

    file_path='E:\\Academics\\4-2\\Project\\Project Related Material\\Datasets for Hub-location problems\\CAB dataset.txt'
    W=[[None for _ in range(n)] for _ in range(n)]
    C=[[None for _ in range(n)] for _ in range(n)]
    read_input_from_datafile(file_path,W,C)

    apply_NSGA_II_algorithm(W,C)
    

if __name__=='__main__':
    start_time=timeit.default_timer()
    main()
    end_time=timeit.default_timer()
    print('Execution Time = ',end_time-start_time)