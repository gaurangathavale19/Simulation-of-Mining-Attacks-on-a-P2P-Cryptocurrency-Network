import argparse
import random
import pandas as pd
import numpy as np
import hashlib
from node import Node
import heapq
from event import Event
from block import Block
import time
from datetime import datetime
import os
from transaction import Transaction
import networkx as nx
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import copy
# nodes = []
latencies = []
global_queue=[]

adversary = None

# populate initial node/peer balances
def populate_peer_balance(transaction_list):
    peer_balance = {}
    for txn in transaction_list:
        peer_balance[txn.receiver_id] = txn.coins
    return peer_balance

# initialize the blockchain with a starting genesis block
def initialize_blockchain(genesis_block):
    blockchain_tree = {}
    blockchain_tree[genesis_block.block_id] = (genesis_block, 1)
    return blockchain_tree # this will also be the candidate block

# Creation of the network topology into an adjacency list
def create_network_topology(total_nodes, adversary_index, zeta):
        print('Adversary Index: ', adversary_index)
        mat = {}

        min1 = 4
        max1 = min(total_nodes-1, 8)

        for i in range(total_nodes):
            mat[i] = []
        start_time = time.time()
        for i in range(total_nodes):
            if(i == adversary_index):
                if(zeta!=100):
                    peers = (total_nodes * zeta)//100
                else:
                    peers = total_nodes-1
                upper_bound = peers
                print(peers)
            else:
                peers = random.randint(min1, max1)
                upper_bound = 8
            # print('Peers:', peers)
            if(len(mat[i]) >= peers):
                continue
            set1 = set()
            if(len(mat[i]) > 0):
                test = mat[i]
                for e in test:
                    set1.add(e)
            while(len(set1) < peers):
                if(time.time() - start_time > 1):
                    return False
                ans = False
                if(ans == False):
                    peer = random.randint(0, total_nodes-1)
                    while(len(mat[peer]) >= upper_bound or i==peer):
                        peer = random.randint(0, total_nodes-1)
                if(peer not in set1):
                    set1.add(peer)
            # print(set1)
            # print(i)
            # print('Peer:',peer)
            mat[i] = list(set1)
            for ele in set1:
                if(i not in mat[ele]):
                    list1 = mat[ele]
                    list1.append(i)
                    mat[ele] = list1

        # Convert the adjacency list into an adjacency matrix
        adj_matrix = [[0 for _ in range(total_nodes)] for _ in range(total_nodes)]

        for k,v in mat.items():
            if(len(v) < 4 or len(v) > 8):
                # print(len(v))
                pass
            for index in v:
                adj_matrix[k][index] = 1
        return adj_matrix

def is_graph_connected(adj_matrix):
    n = len(adj_matrix)
    visited = [False] * n

    def DFS(node):
        visited[node] = True
        for i in range(n):
            if adj_matrix[node][i] == 1 and visited[i] == False:
                DFS(i)

    for i in range(n):
        if visited[i] == False:
            DFS(i)
            break
        
    return all(visited)

if __name__ == "__main__":
    then = time.time()

    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_peers', required=True, help='Enter number of nodes')
    parser.add_argument('--slow_nodes', required=True, help='Enter the percentage of slow nodes')
    parser.add_argument('--lowCPU_nodes', required=True, help='Enter the percentage of low CPU nodes')
    parser.add_argument('--txn_mean_time', required=True, help='Enter interarrival mean time between transactions')
    parser.add_argument('--blk_mean_time', required=True, help='Enter interarrival mean time between blocks')
    parser.add_argument('--termination_time', required=True, help='Enter the termination time of the simulation')
    parser.add_argument('--zeta', required=True, help='Enter fraction of honest nodes, adversary is connected to')
    parser.add_argument('--adversary_hashing_power', required=True, help='Enter the hashing power of adversary')
    parser.add_argument('--attack_type', required=True, help='selfish or stubborn')

    args = parser.parse_args()

    # Create folders for loggers and results
    folder = datetime.now().strftime("%Y-%m-%d %H:%M:%S ") + args.attack_type
    os.mkdir(str(folder))
    os.mkdir(str(folder) + '/results')
    os.mkdir(str(folder) + '/loggers')
    os.mkdir(str(folder) + '/loggers/block')
    os.mkdir(str(folder) + '/loggers/transaction')

    ##### Start 1 #####
    simulator_global_time = 0
    txn_mean_time = int(args.txn_mean_time)
    block_inter_arrival_mean_time = int(args.blk_mean_time)
    termination_time = int(args.termination_time)
    total_nodes = int(args.n_peers)
    z0 = int(args.slow_nodes)
    z1 = int(args.lowCPU_nodes)
    attack_type = args.attack_type
    number_of_slow_nodes = int(total_nodes*z0/100)
    number_of_low_CPU_nodes = int(total_nodes*z1/100)
    number_of_high_nodes=total_nodes-number_of_low_CPU_nodes
    latencies = [[0 for i in range(total_nodes)] for j in range(total_nodes)]

    print('Number of nodes:', total_nodes)
    print('Number of slow nodes:', number_of_slow_nodes)
    print('Number of low CPU nodes:', number_of_low_CPU_nodes)

    speeds = []
    computation_powers = []
    nodes = []
    
    # Allocate nodes with speed and computation power randomly
    speeds = [0]*number_of_slow_nodes + [1]*(total_nodes - number_of_slow_nodes)
    computation_powers = [0]*number_of_low_CPU_nodes + [1]*(total_nodes - number_of_low_CPU_nodes)
    random.shuffle(speeds)
    random.shuffle(computation_powers)

    adversary_index = speeds.index(1)
    adversary_computation_power = computation_powers[adversary_index]

    if(adversary_computation_power == 0): number_of_low_CPU_nodes -= 1 
    else: number_of_high_nodes -= 1

    adversary_hashing_power = float(args.adversary_hashing_power)
    honest_hashing_power = 1-adversary_hashing_power
    low_hk=honest_hashing_power/(number_of_low_CPU_nodes+10*(number_of_high_nodes))
    high_hk=10*low_hk

    print("high", low_hk*number_of_low_CPU_nodes + high_hk*number_of_high_nodes + adversary_hashing_power)
    # print(high_hk)

    # Populate the run_configurations.txt file with the runtime parameters
    file_name = str(folder) + '/run_configurations.txt'
    file = open(file_name, 'w')
    line = "No. of nodes: {}\nSlow percentage nodes: {}%\nNo. of slow nodes: {}\nLow CPU percentage nodes: {}%\nNo. of low CPU nodes: {}\nMean transaction interarrival time: {}\nMean block interarrival time: {}\nTermination time: {}".format(total_nodes, z0, number_of_slow_nodes, z1, number_of_low_CPU_nodes, txn_mean_time, block_inter_arrival_mean_time, termination_time)
    file.write(line)
    line = "\nHigh CPU nodes hashing power: {}\nLow CPU nodes hashing power: {}\nAdversary hashing power: {}\n".format(high_hk, low_hk, adversary_hashing_power)
    file.write(line)
    file.close()
    

    # Creation of the network topology into an adjacency list
    adj_matrix = create_network_topology(total_nodes, adversary_index, int(args.zeta))
    while(adj_matrix == False):
        print("Graph was disconnected! Trying again....")
        adj_matrix = create_network_topology(total_nodes, adversary_index, int(args.zeta))

    for i in adj_matrix:
        print(i)
        if(sum(i) > 8):
            print(sum(i))
    # Compute the hashing power depending on the CPU power of the
    hashing_power_list = []

    for i in range(total_nodes):
        if computation_powers[i]:
            hashing_power_list.append(high_hk)
        else:
            hashing_power_list.append(low_hk)
    hashing_power_list[adversary_index] = adversary_hashing_power

    # Initialize all the peers/nodes with some random amout of BTC in the range 20 to 40 BTC
    initial_txns=[]
    for id in range(total_nodes):
        coins=random.randint(20,40)
        # nodes.append(Node(id, speeds[id], computation_powers[id], coins))
        next_mining_time = simulator_global_time + np.random.exponential(block_inter_arrival_mean_time/hashing_power_list[id])
        nodes.append(Node(node_id=id, speed=speeds[id], computation_power=computation_powers[id], coins=coins, hashing_power=hashing_power_list[id], block_inter_arrival_mean_time=block_inter_arrival_mean_time, transaction_inter_arrival_mean_time=txn_mean_time, simulator_global_time=simulator_global_time, next_mining_time=next_mining_time))
        # sender_id, receiver_id, coins, transaction_type, timestamp
        # print(hashing_power_list)
        txn=Transaction("coinbase",id,coins,"init",0)
        initial_txns.append(txn)
    
    adversary = nodes[adversary_index]
    print(adversary.hashing_power)
    
    # Initialize blockchain tree of all the nodes with the genesis block
    for id in range(total_nodes):
        nodes[id].genesis_block = Block(creator_id=nodes[id].node_id, creation_time=simulator_global_time, peer_balance=populate_peer_balance(initial_txns), transaction_list=initial_txns, previous_block_hash=0)
        nodes[id].blockchain_tree = nodes[i].candidate_blocks = initialize_blockchain(nodes[id].genesis_block)
        nodes[id].longest_chain_last_block = {'block': nodes[id].genesis_block, 'length': 1}
        nodes[id].block_arrival_timing = { nodes[id].genesis_block.block_id : simulator_global_time}
        # print(nodes[id].longest_chain_last_block)


    # Initializing latencies between each pair of directly connected nodes
    latency_matrix=[[0 for i in range(total_nodes)] for i in range(total_nodes)]
    for i in range(0,total_nodes):
        neighbour_list=[]
        for j in range(0,total_nodes):
            if(adj_matrix[i][j]):
                peer_inf={}
                if(latency_matrix[j][i]==0):
                    peer_inf['propagation_delay']=(np.random.uniform(low=10, high=500))/1000  #todo
                    latency_matrix[i][j]=peer_inf['propagation_delay']
                else:
                    peer_inf['propagation_delay']=latency_matrix[j][i]
                peer_inf['node']=nodes[j]
                peer_inf['node_id']=j
                if nodes[i].speed==1 and nodes[j].speed==1:
                    peer_inf['bottleneck_bandwidth']=100
                else:
                    peer_inf['bottleneck_bandwidth']=5

                neighbour_list.append(peer_inf)
                    
        nodes[i].neighbours=neighbour_list
        # print(nodes[i].neighbours)
    #end 5a


    # Initally generate transaction with each peer/node
    for id in range(total_nodes):
        new_event=nodes[id].generate_transaction(n=total_nodes, current_time=simulator_global_time, txn_mean_time=txn_mean_time)
        heapq.heappush(global_queue,new_event)
    
    # Initially create a block event from each peer/node
    for id in range(total_nodes):
        #todo interblock_arrival_time?
        # new_event=Event(nodes[id],"BLK",None,nodes[i],"all",simulator_global_time+d)
        new_event = Event(curr_node=nodes[id].node_id, type="BLK", event_data=None, sender_id=nodes[id].node_id, receiver_id="all", event_start_time=nodes[id].next_mining_time)
        heapq.heappush(global_queue, new_event)
    
    # Loggers for the initial events in the event queue
    for i in global_queue:
        print(i.type, i.event_start_time)
    ##end 5c

    # Creating network topology as a visual aid - can be found in the <current-timestamp> folder with name 'network_topology.png'
    G = nx.Graph()
    node_colors = []
    for node in nodes:
        node_colors.append('red' if adversary_index == node.node_id else 'green' if node.computation_power else 'blue')
        for adj_vertex in node.neighbours:
            edge_color = 'green' if adj_vertex['bottleneck_bandwidth']==100 else 'red'
            G.add_edge(node.node_id, adj_vertex['node_id'], color=edge_color, weight=10)

    edge_colors = nx.get_edge_attributes(G,'color').values()
    nx.draw(G, edge_color=edge_colors, node_color=node_colors, with_labels=True, font_color='white')
    plt.savefig('./{}/network_topology.png'.format(str(folder)), dpi=300, bbox_inches='tight')

    # Logging the events in events.csv
    file_name = str(folder) + '/events.csv'
    events_log_file = open(file=file_name, mode='w')
    line = "Event type,Event start time,sender_node,receiver_node,current_node\n"
    events_log_file.write(line)

    # Simulator will run until it reaches the termination time
    while(simulator_global_time<termination_time):
        # print(simulator_global_time)
        # print(termination_time)
    
        # Pop an event and process it based on the sender and received ids
        curr_event = heapq.heappop(global_queue)

        # Update the simulator time with the time that event took to arrive
        simulator_global_time = curr_event.event_start_time
        # print(curr_event.type)

        # If the event type is BLK i.e. Block
        if curr_event.type == "BLK":
            # pass
            curr_node_id = curr_event.curr_node
            event_content = curr_event.event_data
            sender_id = curr_event.sender_id
            # print(curr_node_id)

            # Logging into events.csv
            line = "{},{},{},{},{}\n".format(curr_event.type,curr_event.event_start_time,sender_id,curr_event.receiver_id,curr_node_id)
            events_log_file.write(line)

            #print("BLK:", simulator_global_time, curr_node_id, sender_id)
            # simulator_global_time = curr_event.event_start_time

            if curr_node_id == sender_id:
                events_generated = nodes[curr_node_id].generate_block(simulator_global_time, curr_event, adversary_index, attack_type)
                # simulator_global_time += next_mining_time
                # print(events_generated)
                # print('Done with generate block')
            else:
                events_generated = nodes[curr_node_id].receive_block(simulator_global_time, event_content, adversary_index, attack_type)
                #print('Done with receive block')
                print(events_generated)
                
        # If the the even type is TXN i.e. Transaction
        else:
            curr_node = curr_event.curr_node
            curr_node_id = curr_node.node_id
            event_content = curr_event.event_data
            sender_id = curr_event.sender_id

            # Logging into events.csv
            line = "{},{},{},{},{}\n".format(curr_event.type,curr_event.event_start_time,sender_id,curr_event.receiver_id,curr_node_id)
            events_log_file.write(line)

            # print(sender_id)
            # print(curr_event.event_data.coins)
            # simulator_global_time = curr_event.event_start_time
            #print("TXN:", simulator_global_time, " ", curr_node_id, " ", event_content.transaction_message, " ", sender_id)
            events_generated = curr_node.get_transactions(simulator_global_time, event_content)

            if curr_node_id == sender_id:
                new_event = curr_node.generate_transaction(total_nodes, simulator_global_time, txn_mean_time)
                events_generated.append(new_event)
        
        for event in events_generated:
            heapq.heappush(global_queue,event)
    print(len(adversary.private_blockchain_tree))
    while(len(adversary.private_blockchain_tree) > 0):
        block_to_be_broadcasted = adversary.private_blockchain_tree.popleft()
        for i in range(total_nodes):
            nodes[i].blockchain_tree[block_to_be_broadcasted[0].block_id] = block_to_be_broadcasted
            nodes[i].longest_chain_last_block = {'block': block_to_be_broadcasted[0], 'length': block_to_be_broadcasted[1]}


    print('Reached termination time')
    print('Simulation time in seconds:',time.time() - then)

    # Loggers 
    for node in nodes:
        # #print(count,len(node.non_verfied_transaction), len(node.all_transaction), len(node.block_tree), node.longest_chain_last_block[1], len(node.all_block_ids.keys()),sep='\t\t')
        ##print(node.genesis_block.id)
        try:
            # Create the blockchain tree for each node
            node.visualize(folder, adversary_index)
        except Exception as e:
            pass

    node_info_table = PrettyTable(["Node ID", "Speed", "Computation Power", "Total Generated blocks", "Blocks in the longest chain", "Total blocks/blocks in longest chain"])
    number_of_blocks_across_all_nodes = 0
    number_of_block_in_main_chain = 0
    number_of_block_mined_by_the_adversary_in_main_chain = 0
    total_number_of_blocks_mined_by_adversary = 0
    for node in nodes:
        
        # Calculate the ratio of the number of blocks generated by each node in the Longest Chain of the tree to the total number of blocks it generates
        generated_blocks = len(node.generated_blocks)
        count_of_generated_blocks_in_longest_blockchain = node.get_count_of_generated_blocks_in_longest_blockchain()
        if(generated_blocks != 0):
            # print("Node", node.node_id, ":", count_of_generated_blocks_in_longest_blockchain/generated_blocks, count_of_generated_blocks_in_longest_blockchain, node.speed, node.computation_power)
            fraction_of_generated_blocks_in_longest_blockchain = count_of_generated_blocks_in_longest_blockchain/generated_blocks
        else:
            # print('Node', node.node_id, ': No blocks generated',  node.speed, node.computation_power)
            fraction_of_generated_blocks_in_longest_blockchain = 'No blocks generated'

        if(node.node_id == adversary_index):
            number_of_block_mined_by_the_adversary_in_main_chain = count_of_generated_blocks_in_longest_blockchain
            total_number_of_blocks_mined_by_adversary = generated_blocks
        else:
            number_of_blocks_across_all_nodes += generated_blocks
            number_of_block_in_main_chain += count_of_generated_blocks_in_longest_blockchain
        node_info_table.add_row([node.node_id, node.speed, node.computation_power, generated_blocks, count_of_generated_blocks_in_longest_blockchain, fraction_of_generated_blocks_in_longest_blockchain])

        file_name = str(folder) + '/loggers/{}/log_' + str(node.node_id) + '_{}.tsv'
        file = open(file_name.format('block','block'), 'w')
        line = "Ratio of the number of blocks generated by each node in the Longest Chain of the tree to the total number of blocks it generates\t{}\n".format(fraction_of_generated_blocks_in_longest_blockchain) 
        file.write(line)

        if(node.speed == 1): line = "Fast Node\n" 
        else: line = "Slow Node\n"

        if(node.computation_power == 1): line += "High CPU Node\n" 
        else: line += "Low CPU Node\n"
        
        file.write(line)

        line = "Block ID\tBlock arrival time\tNo. of transactions\tPeer Balance\n"
        file.write(line)

        for block_id, block in node.blockchain_tree.items():
            try:
                line = "{}\t{}\t{}\t{}\n".format(block_id, node.block_arrival_timing[block_id], len(block[0].transaction_list), block[0].peer_balance)
                # print(line)
                file.write(line)
            except:
                continue
        file.close()

        file = open(file_name.format('transaction', 'transaction'), 'w')
        line = "Transaction ID\tTransaction Type\tTimestamp\tSender\tReceiver\tAmount (in BTC)\n"
        file.write(line)

        for txn in node.genesis_block.transaction_list:
            line = "{}\t{}\t{}\t{}\t{}\t{}\n".format(txn.transaction_id, txn.transaction_type, txn.timestamp, txn.sender_id, txn.receiver_id, txn.coins)
            file.write(line)

        for txn in node.verified_transactions:
            line = "{}\t{}\t{}\t{}\t{}\t{}\n".format(txn.transaction_id, txn.transaction_type, txn.timestamp, txn.sender_id, txn.receiver_id, txn.coins)
            file.write(line)
        file.close()
    
    print(number_of_block_mined_by_the_adversary_in_main_chain)
    print(total_number_of_blocks_mined_by_adversary)
    print(number_of_block_in_main_chain)
    print(number_of_blocks_across_all_nodes)
    
    if(total_number_of_blocks_mined_by_adversary != 0):
        print("MPU Adversary:", number_of_block_mined_by_the_adversary_in_main_chain/total_number_of_blocks_mined_by_adversary)
        mpu_adversary = number_of_block_mined_by_the_adversary_in_main_chain/total_number_of_blocks_mined_by_adversary
    else:
        print("MPU Adversary:", 0)
        mpu_adversary = 0

    print("MPU Overall:", number_of_block_in_main_chain/number_of_blocks_across_all_nodes)
    mpu_overall = number_of_block_in_main_chain/number_of_blocks_across_all_nodes
    
    results = pd.DataFrame({
        "n_peers" : [args.n_peers],
        "slow_nodes" : [args.slow_nodes],
        "lowCPU_nodes" : [args.lowCPU_nodes],
        "txn_mean_time" : [args.txn_mean_time],
        "blk_mean_time" : [args.blk_mean_time],
        "termination_time" : [args.termination_time],
        "zeta" : [args.zeta],
        "adversary_hashing_power" : [args.adversary_hashing_power],
        "attack_type" : [args.attack_type],
        "MPU_Adversary" : [mpu_adversary],
        "MPU_Overall": [mpu_overall]
    })

    # if file does not exist write header 
    if not os.path.isfile('./simulation.csv'):
        results.to_csv('./simulation.csv', header='column_names', index=False)
    else: # else it exists so append without writing the header
        results.to_csv('./simulation.csv', mode='a', header=False, index=False)

    print(node_info_table)