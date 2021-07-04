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


def compute_total_transportation_cost(node_allocation_array,W,C):

    total_transportation_cost=0

    for i in range(n):

        hub_allocated_to_node_i=node_allocation_array[i]
        total_flow_of_products_originating_from_node_i=0

        for j in range(n):
            total_flow_of_products_originating_from_node_i+=W[i][j]

        origin_to_hub_cost_due_to_node_i=(unit_transportation_cost*C[i][hub_allocated_to_node_i])*total_flow_of_products_originating_from_node_i
        inter_hub_cost_due_to_node_i=0

        for j in range(n):

            if hub_allocated_to_node_i==node_allocation_array[j]:
                continue
            else:
                k=hub_allocated_to_node_i
                l=node_allocation_array[j]
                inter_hub_cost_due_to_node_i+=unit_transportation_cost*C[k][l]*W[i][j]

        total_flow_of_products_destined_for_node_i=0

        for j in range(n):
            total_flow_of_products_destined_for_node_i+=W[j][i]

        hub_to_destination_cost_for_node_i=(unit_transportation_cost*C[i][hub_allocated_to_node_i])*total_flow_of_products_destined_for_node_i
        total_cost_due_to_node_i=origin_to_hub_cost_due_to_node_i+inter_hub_cost_due_to_node_i+hub_to_destination_cost_for_node_i
        total_transportation_cost+=total_cost_due_to_node_i

    return total_transportation_cost


def compute_total_carbon_emission(node_allocation_array,W,C):

    total_carbon_emission=0
    non_hubs=[]

    for i in range(n):

        if node_allocation_array[i]!=i:
            non_hubs.append(i)

    for j in non_hubs:
        total_flow_of_products_destined_for_non_hub_node_j=0

        for i in range(n):
            total_flow_of_products_destined_for_non_hub_node_j+=W[i][j]

        k=node_allocation_array[j]
        carbon_emission_due_to_full_loaded_vehicle=C[k][j]*(Ef-Ee)*(total_flow_of_products_destined_for_non_hub_node_j/weight_limit)
        carbon_emission_due_to_empty_loaded_vehicle=C[k][j]*Ee*(math.ceil(total_flow_of_products_destined_for_non_hub_node_j/weight_limit))
        carbon_emission_due_to_non_hub_node_j=carbon_emission_due_to_full_loaded_vehicle+carbon_emission_due_to_empty_loaded_vehicle
        total_carbon_emission+=carbon_emission_due_to_non_hub_node_j

    return total_carbon_emission


def compute_total_responsiveness_of_the_network(solution,W,C):

    node_allocation_array=solution[0]
    vehicle_assign_array=solution[1]
    vehicle_sequence_array=solution[2]                      

    cust_order_arrival_time=[0 for _ in range(n)]
    hub_to_index_map=dict()
    index=0

    for i in range(n):

        if node_allocation_array[i]==i:
            hub_to_index_map[i]=index
            index+=1

    hub_and_vehicle_to_cust_nodes_sequenced_map=[[[] for _ in range(D)] for _ in range(p)]

    for i in range(n):
        vehicle_sequence_number=vehicle_sequence_array[i]

        if node_allocation_array[i]==i:
            continue
        else:

            hub_allocated_to_node_i=node_allocation_array[i]
            hub_index_of_the_allocated_hub=hub_to_index_map[hub_allocated_to_node_i]
            vehicle_index_allocated_to_node_i=vehicle_assign_array[i]
            hub_and_vehicle_to_cust_nodes_sequenced_map[hub_index_of_the_allocated_hub][vehicle_index_allocated_to_node_i].append((vehicle_sequence_number,i))

    for k in range(p):
        for d in range(D):
            hub_and_vehicle_to_cust_nodes_sequenced_map[k][d].sort()	
			
    for k in range(p):
        for d in range(D):
            vehicle_seq=[hub_and_vehicle_to_cust_nodes_sequenced_map[k][d][i][1] for i in range(len(hub_and_vehicle_to_cust_nodes_sequenced_map[k][d]))]

            for j in range(len(vehicle_seq)):

                if j==0:
                    i=vehicle_seq[j]
                    hub_allocated_to_node_i=node_allocation_array[i]
                    travel_time_from_hub_to_node_i=C[hub_allocated_to_node_i][i]/vehicle_speed
                    cust_order_arrival_time[i]=travel_time_from_hub_to_node_i

                else:
                    i1=vehicle_seq[j-1]
                    i2=vehicle_seq[j]
                    assert(node_allocation_array[i1]==node_allocation_array[i2])
                    hub_allocated_to_both_nodes_i1_and_i2=node_allocation_array[i1]
                    travel_time_from_node_i1_to_hub=C[i1][hub_allocated_to_both_nodes_i1_and_i2]/vehicle_speed
                    travel_time_from_hub_to_node_i2=C[hub_allocated_to_both_nodes_i1_and_i2][i2]/vehicle_speed
                    cust_order_arrival_time[i2]=cust_order_arrival_time[i1]+travel_time_from_node_i1_to_hub+travel_time_from_hub_to_node_i2

    total_responsiveness_of_the_network=0

    for i in range(n):
        purchase_probability_of_cust_node_i=(1-(cust_order_arrival_time[i]/shelf_life))/(1-QRP)
        total_responsiveness_of_the_network+=max(0,min(1,purchase_probability_of_cust_node_i))

    return total_responsiveness_of_the_network


def compute_the_objective_function_values_of_the_solution(solution,W,C):

    total_transportation_cost=compute_total_transportation_cost(solution[0],W,C)
    total_responsiveness_of_the_network=compute_total_responsiveness_of_the_network(solution,W,C)
    total_carbon_emission=compute_total_carbon_emission(solution[0],W,C)
    return [total_transportation_cost,total_responsiveness_of_the_network,total_carbon_emission]


def if_solution_i_dominates_solution_j(i,j,objective_function_values_of_the_solution):

    objective_function_values_of_solution_i=objective_function_values_of_the_solution[i]
    objective_function_values_of_solution_j=objective_function_values_of_the_solution[j]

    total_transportation_cost_of_solution_i=objective_function_values_of_solution_i[0]
    total_responsiveness_of_solution_i=objective_function_values_of_solution_i[1]
    total_carbon_emission_of_solution_i=objective_function_values_of_solution_i[2]

    total_transportation_cost_of_solution_j=objective_function_values_of_solution_j[0]
    total_responsiveness_of_solution_j=objective_function_values_of_solution_j[1]
    total_carbon_emission_of_solution_j=objective_function_values_of_solution_j[2]

    if total_transportation_cost_of_solution_i<total_transportation_cost_of_solution_j and total_responsiveness_of_solution_i>total_responsiveness_of_solution_j and total_carbon_emission_of_solution_i<total_carbon_emission_of_solution_j:
    	return True
    elif total_transportation_cost_of_solution_i==total_transportation_cost_of_solution_j and total_responsiveness_of_solution_i>total_responsiveness_of_solution_j and total_carbon_emission_of_solution_i<total_carbon_emission_of_solution_j:
    	return True
    elif total_transportation_cost_of_solution_i<total_transportation_cost_of_solution_j and total_responsiveness_of_solution_i==total_responsiveness_of_solution_j and total_carbon_emission_of_solution_i<total_carbon_emission_of_solution_j:
    	return True
    elif total_transportation_cost_of_solution_i<total_transportation_cost_of_solution_j and total_responsiveness_of_solution_i>total_responsiveness_of_solution_j and total_carbon_emission_of_solution_i==total_carbon_emission_of_solution_j:
    	return True
    elif total_transportation_cost_of_solution_i==total_transportation_cost_of_solution_j and total_responsiveness_of_solution_i==total_responsiveness_of_solution_j and total_carbon_emission_of_solution_i<total_carbon_emission_of_solution_j:
    	return True
    elif total_transportation_cost_of_solution_i<total_transportation_cost_of_solution_j and total_responsiveness_of_solution_i==total_responsiveness_of_solution_j and total_carbon_emission_of_solution_i==total_carbon_emission_of_solution_j:
    	return True
    elif total_transportation_cost_of_solution_i==total_transportation_cost_of_solution_j and total_responsiveness_of_solution_i>total_responsiveness_of_solution_j and total_carbon_emission_of_solution_i==total_carbon_emission_of_solution_j:
    	return True
    else:
    	return False


def do_fast_non_dominated_sort(objective_function_values_of_the_solution,W,C):

    pareto_fronts=[]
    indices_of_solutions_dominated_by_the_solution=[[] for _ in range(2*population_size)]
    num_of_solutions_dominating_the_solution=[0 for _ in range(2*population_size)]
    curr_front=[]

    for i in range(2*population_size):
    	for j in range(2*population_size):

    		if if_solution_i_dominates_solution_j(i,j,objective_function_values_of_the_solution)==True:
    			indices_of_solutions_dominated_by_the_solution[i].append(j)
    		elif if_solution_i_dominates_solution_j(j,i,objective_function_values_of_the_solution)==True:
    			num_of_solutions_dominating_the_solution[i]+=1

    	if num_of_solutions_dominating_the_solution[i]==0:
    		curr_front.append(i)

    next_front=[]

    while len(curr_front)>0:
    	pareto_fronts.append(curr_front)

    	for i in curr_front:
    		for j in indices_of_solutions_dominated_by_the_solution[i]:
    			num_of_solutions_dominating_the_solution[j]-=1

    			if num_of_solutions_dominating_the_solution[j]==0:
    				next_front.append(j)

    	curr_front=copy.deepcopy(next_front)
    	next_front.clear()
		
    return pareto_fronts


def update_the_crowding_distance_value_for_the_objective_function(pareto_front,L,objective_function_index,crowding_distance_and_solution_index_tuple,solution_index_to_index_map,objective_function_values_of_the_solution,min_max_values_of_the_objective_function):

    objective_function_and_solution_index_tuple=[[objective_function_values_of_the_solution[solution_index][objective_function_index],solution_index] for solution_index in pareto_front]
    objective_function_and_solution_index_tuple.sort()
    first_solution_index=objective_function_and_solution_index_tuple[0][1]
    last_solution_index=objective_function_and_solution_index_tuple[-1][1]
    crowding_distance_and_solution_index_tuple[solution_index_to_index_map[first_solution_index]][0]=sys.maxsize
    crowding_distance_and_solution_index_tuple[solution_index_to_index_map[last_solution_index]][0]=sys.maxsize

    for i in range(1,L-1):
    	solution_index=objective_function_and_solution_index_tuple[i][1]
    	prev_solution_index=objective_function_and_solution_index_tuple[i-1][1]
    	next_solution_index=objective_function_and_solution_index_tuple[i+1][1]
    	t=(objective_function_values_of_the_solution[next_solution_index][objective_function_index]-objective_function_values_of_the_solution[prev_solution_index][objective_function_index])/(min_max_values_of_the_objective_function[objective_function_index][1]-min_max_values_of_the_objective_function[objective_function_index][0])
    	crowding_distance_and_solution_index_tuple[solution_index_to_index_map[solution_index]][0]+=t


def do_crowding_distance_assignment(R,pareto_front,objective_function_values_of_the_solution,min_max_values_of_the_objective_function):

	L=len(pareto_front)
	crowding_distance_and_solution_index_tuple=[]
	solution_index_to_index_map=dict()

	for i in range(len(pareto_front)):
		solution_index=pareto_front[i]
		crowding_distance_and_solution_index_tuple.append([0,solution_index])
		solution_index_to_index_map[solution_index]=i

	for i in range(3):
		update_the_crowding_distance_value_for_the_objective_function(pareto_front,L,i,crowding_distance_and_solution_index_tuple,solution_index_to_index_map,objective_function_values_of_the_solution,min_max_values_of_the_objective_function)

	return crowding_distance_and_solution_index_tuple
