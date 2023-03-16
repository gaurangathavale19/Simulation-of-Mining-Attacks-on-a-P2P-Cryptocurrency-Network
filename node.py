import random
import numpy as np
from block import Block
from event import Event
from transaction import Transaction
import hashlib
from queue import Queue
import copy
from graphviz import Graph
from collections import deque
attacker_blocks = []
class Node:
    # speed: slow = 0, fast = 1
    # computation_power: low = 0, high = 1
    def __init__(self, node_id, speed, computation_power, coins, hashing_power, block_inter_arrival_mean_time, transaction_inter_arrival_mean_time, simulator_global_time, next_mining_time):
        '''
            -node_id: ID of the current node
            -speed: Speed of the node i.e. slow/fast - 0/1
            -computation_power: Computation power of the node i.e. low/high - 0/1
            -coins: current balance of the node
            -hashing_power: hashing power of the node
            -block_inter_arrival_mean_time: Mean interarrival time between blocks 
            -transaction_inter_arrival_mean_time: Mean interarrival time between transactions
            -simulator_global_time: Global time of the simulator
            -next_mining_time: Time at which the next block can be mined
        '''
        
        self.node_id = node_id
        self.speed = speed
        self.computation_power = computation_power
        self.block_inter_arrival_mean_time = block_inter_arrival_mean_time
        self.transaction_inter_arrival_mean_time = transaction_inter_arrival_mean_time
        self.hashing_power = hashing_power

        #added for maintaing transactions
        self.unverfied_transactions = {}
        self.verified_transactions = []

        # added for maintaining blocks and blockchain
        self.next_mining_time = next_mining_time
        # self.blockchain_tree, self.candidate_blocks = self.initialize_blockchain()
        self.blockchain_tree = {}
        self.private_blockchain_tree = deque()
        self.candidate_blocks = {}
        # self.longest_chain_last_block = {'block': self.genesis_block, 'length': 1}
        self.longest_chain_last_block = {}
        self.private_longest_chain_last_block = {}
        self.blocks = set()
        self.unverified_blocks = {}
        
        self.coins = coins
        self.neighbours=[]
        self.visited_transactions={}
        self.unverified_txn={}
        self.block_curr_mine_time=None
        # self.hashing_power=None
        self.transactions=None
        self.block_arrival_timing={}
        self.generated_blocks = set()
        self.my_blocks = []

    # Generate the transaction
    def generate_transaction(self, n, current_time,txn_mean_time):
        '''
         -n: total number of nodes in the network
         -current_time: current time of the simulation
         -txn_mean_time: Mean interarrival time between transactions
        '''

        # We have the sender node
        sender_id=self.node_id

        # Find the receiver such that sender and receiver are not the same
        receiver_id = random.randint(0,n-1)
        while(sender_id == receiver_id):
            receiver_id = random.randint(0,n-1)

        # Allocate some amount based on the balance in sender's account
        coins = random.randint(1,self.coins)

        # Create the generated event time based on current time + random exponential distribution with mean as txn_mean_time (Ttx)
        exp_time = np.random.exponential(txn_mean_time)
        generated_event_time = exp_time + current_time

        # Create a transaction object and add it into an event, which will be further added to the event queue (global_queue)
        txn=Transaction(sender_id,receiver_id,coins,"payment",generated_event_time)
        event=Event(self,"TXN",txn,sender_id,receiver_id,generated_event_time)
        return event

    # Calculate latency between each pair of directly connected nodes
    def calc_latency(self,neighbour,message_len):
        '''
            -neighbour: Adjacent nodes to the current node
            -message_len: Length of the message
        '''

        # Calculate the bottleneck backwidth (c)
        if(self.speed == 1 and neighbour['node'].speed==1):
            c = 100 * 10**6
        else:
            c = 5 * 10**6

        # Calculate d from exponential distribution with mean 96kbits/c
        mean = 96 * 10**3 / c
        d = np.random.exponential(mean)

        # Calculate propagation delay (p)
        p = neighbour['propagation_delay']

        # Calculate final latency with the given formula
        latency=p+message_len/c+d

        return round(latency,2)
    

    # Receive transactions 
    def get_transactions(self,current_time,txn):
        '''
            -current_time: current time of the simulation
            -txn: Transaction object
        '''
        reciever_id=txn.receiver_id
        sender_id=txn.sender_id
        message_len=8192 
        txnid=txn.transaction_id
        self.visited_transactions[txnid]=1
        new_events_generated=[]
        self.unverified_txn[txnid]=txn
        for neighbour in self.neighbours:
            lat=self.calc_latency(neighbour,message_len)
            new_event_time=lat+current_time
            if txnid not in neighbour['node'].visited_transactions:
                new_event=Event(neighbour['node'],"TXN",txn,sender_id,reciever_id,new_event_time)#todo
                new_events_generated.append(new_event)
            # new_event=Event(self,"TXN",txn,sender_id,reciever_id,new_event_time)#todo
            # new_events_generated.append(new_event)

        return new_events_generated

    # Generate blocks
    def generate_block(self, simulator_global_time, event, adversary_index):
        '''
            -simulator_global_time: Global time of the simulator
            -event: object of the event
        '''
        print(self.hashing_power)
        if self.next_mining_time != event.event_start_time: # need to analyze this once
        # if self.next_mining_time != event.event_start_time: # need to analyze this once
            self.next_mining_time = simulator_global_time + np.random.exponential(self.block_inter_arrival_mean_time/self.hashing_power) # need to analyze this once
            return [Event(curr_node=self.node_id, type="BLK", event_data=None, sender_id=self.node_id, receiver_id="all", event_start_time=self.next_mining_time)]
            # return []
        events = []

        # Calculate the interarrival time between block with exponential random distribution using block interarrival mean time and hashing power
        exp_time = np.random.exponential(self.block_inter_arrival_mean_time/self.hashing_power)
        self.next_mining_time = simulator_global_time + exp_time # need to analyze this once
        valid_txns = []
        
        parent_block = copy.deepcopy(self.longest_chain_last_block)
        # If private longest chain last block is NONE, means that attacker has to mine on the longest honest/public chain last block
        private_parent_block = copy.deepcopy(self.private_longest_chain_last_block)
        if not private_parent_block:
            private_parent_block = copy.deepcopy(self.longest_chain_last_block)
        # print(parent_block)
        if(adversary_index == self.node_id):
            parent_peer_balance = copy.deepcopy(private_parent_block['block'].peer_balance)
        else:
            parent_peer_balance = copy.deepcopy(parent_block['block'].peer_balance)
        
        to_be_deleted = []
        # Check for valid transactions and add it to valid_transactions
        for txn_id, txn  in self.unverified_txn.items():
            if(txn.transaction_type=='payment'):
                if (parent_peer_balance[txn.sender_id] >= txn.coins):
                    parent_peer_balance[txn.sender_id] -= txn.coins
                    parent_peer_balance[txn.receiver_id] += txn.coins
                    valid_txns.append(txn)
                    to_be_deleted.append(txn_id)
            else:
                parent_peer_balance[txn.to_id] += txn.coins
                valid_txns.append(txn)
                to_be_deleted.append(txn_id)
            
            # Since one block can have atmost 1000 transactions i.e. 999 regular transactions and 1 coinbase(mandatory) transaction
            if len(valid_txns) == 999: # 1000th Transaction would be added as mining reward
                break
        
        self.verified_transactions += valid_txns
        
        # Delete valid transactions from the unverified transactions list
        for i in to_be_deleted:
            del self.unverified_txn[i]
        
        # Add the mandatory coinbas transaction
        valid_txns.append(Transaction(sender_id="coinbase", receiver_id=self.node_id, coins=50, transaction_type="mines", timestamp=simulator_global_time)) # mining reward is 50
        
        # Update the peer balance by applying the coinbase transaction to the peer balance of the miner
        parent_peer_balance[self.node_id] += 50

        # Create the blocks with above details
        if(adversary_index == self.node_id):
            block = Block(creator_id=self.node_id , creation_time=event.event_start_time, peer_balance=parent_peer_balance, transaction_list=valid_txns, previous_block_hash=private_parent_block['block'].block_id) # need to understand creation_time=event.event_start_time
            attacker_blocks.append(block.block_id)
        else:
            # print(private_parent_block['block'].block_id)
            block = Block(creator_id=self.node_id , creation_time=event.event_start_time, peer_balance=parent_peer_balance, transaction_list=valid_txns, previous_block_hash=parent_block['block'].block_id) # need to understand creation_time=event.event_start_time
        self.my_blocks.append(block.block_id)
        self.generated_blocks.add(block.block_id)
        block.peers_visited.append(self.node_id)

        # Create a block event 
        events.append(Event(curr_node=self.node_id, type="BLK", event_data=None, sender_id=self.node_id, receiver_id="all", event_start_time=self.next_mining_time))
        
        # if(self.node_id != adversary_index):
        if(self.node_id == adversary_index):
            # Update the blockchain tree by adding the above block to the blockchain tree
            # self.private_blockchain_tree[block.block_id] = (block, private_parent_block['length']+1)
            self.private_blockchain_tree.append((block, private_parent_block['length']+1))

            # Update the longest chain and maintain the block arrival timings
            self.private_longest_chain_last_block = {'block': block, 'length': private_parent_block['length']+1}
        else:
            # Update the blockchain tree by adding the above block to the blockchain tree
            self.blockchain_tree[block.block_id] = (block, parent_block['length']+1)

            # Update the longest chain and maintain the block arrival timings
            self.longest_chain_last_block = {'block': block, 'length': parent_block['length']+1}
        self.block_arrival_timing[block.block_id] = simulator_global_time
        self.blocks.add(block.block_id)

        # if(self.node_id == adversary_index):
        #     attacker_lead = self.private_longest_chain_last_block['length'] - self.longest_chain_last_block['length']
            # print("Gnereate Attacker lead is:", attacker_lead)
        
        # Broadcast the blocks to the miner's peers
        print('Block created by node id:', self.node_id, block.block_id)
        if(adversary_index != self.node_id):
            return self.broadcast_block(simulator_global_time, block, events)
        else:
            return [Event(curr_node=adversary_index, type="BLK", event_data=None, sender_id=adversary_index, receiver_id="all", event_start_time=self.next_mining_time)]

    def receive_block(self, simulator_global_time, block, adversary_index):
        '''
            -simulator_global_time: Global time of the simulator
            -block: object of the block
        '''
        
        # Check if the block is seen earlier - to avoid loop
        if block.block_id in self.blocks:
            return []
        
        self.blocks.add(block.block_id)
        block.peers_visited.append(self.node_id)

        previous_block_hash = block.previous_block_hash

        # Check if the incoming block's previous_hash is present in the blockchain tree
        if previous_block_hash not in self.blockchain_tree.keys():
            # Add to list of unverified blocks
            self.unverified_blocks[block.block_id] = block
        else:
            # Check the validity of blocks, if verified, then:
            if self.verify_block(block):
                # Since the block is verified, add the block to the blockchain tree and update the longest chain length
                self.blockchain_tree[block.block_id] = (block, self.blockchain_tree[block.previous_block_hash][1] + 1)
                self.block_arrival_timing[block.block_id] = simulator_global_time

                # Update the longest chain if new block added changes the longest chain length
                if(self.longest_chain_last_block['length'] < self.blockchain_tree[block.block_id][1]):
                    self.longest_chain_last_block['block'] = block
                    self.longest_chain_last_block['length'] = self.blockchain_tree[block.block_id][1]
            
                # Check attacker_lead iff a new block is added to the honest blockchain tree
                if(adversary_index == self.node_id):
                    print('Private Longest chain', self.private_longest_chain_last_block)
                    print('Public longest chain', self.longest_chain_last_block)
                    if(not self.private_longest_chain_last_block):
                        self.private_blockchain_tree = deque()
                        self.private_longest_chain_last_block = {}
                        return []
                
                    attacker_lead = self.private_longest_chain_last_block['length'] - self.longest_chain_last_block['length']
                    print("Attacker lead is:", attacker_lead)
                    if(attacker_lead < 0): # Lead was 0, and became -1, hence attacker will start mining on the longest honest chain
                        self.private_blockchain_tree = deque()
                        self.private_longest_chain_last_block = {}
                        return []

                    # For attacker_lead = 0 : Lead became 0 after new honest block was added, hence release the only attacker block
                    # For attacker_lead = 1 : Release both the attacker blocks
                    elif(attacker_lead == 0): 
                        blocks_to_be_broadcasted_event_list = []
                        self.longest_chain_last_block = copy.deepcopy(self.private_longest_chain_last_block)
                        for private_block in self.private_blockchain_tree:
                            self.blockchain_tree[private_block[0].block_id] = private_block
                            blocks_to_be_broadcasted_event_list += self.broadcast_block(simulator_global_time, private_block[0], event_list=[])
                        self.private_blockchain_tree = deque()
                        return blocks_to_be_broadcasted_event_list

                        ######################################################################################
                        ########################## Check with CHAUDHARI AND GABANI  ##########################
                        ######################################################################################
                    elif(attacker_lead == 1): # Since, the attacker has to release both the blocks hence, now we reach state 0. That's the reason we flush put private_blockchain_tree and the private_longest_chain_last_block
                        blocks_to_be_broadcasted_event_list = []
                        self.longest_chain_last_block = copy.deepcopy(self.private_longest_chain_last_block)
                        for private_block in self.private_blockchain_tree:
                            self.blockchain_tree[private_block[0].block_id] = private_block
                            blocks_to_be_broadcasted_event_list += self.broadcast_block(simulator_global_time, private_block[0], event_list=[])
                        self.private_blockchain_tree = deque()

                        # self.longest_chain_last_block = self.private_longest_chain_last_block
                        self.private_longest_chain_last_block = {} 

                        return blocks_to_be_broadcasted_event_list

                    # elif(attacker_lead == 1): 
                    #     blocks_to_be_broadcasted_event_list = []
                    #     self.longest_chain_last_block = self.private_longest_chain_last_block
                    #     for private_block in self.private_blockchain_tree:
                    #         self.blockchain_tree[private_block[0].block_id] = private_block
                    #         blocks_to_be_broadcasted_event_list.extend(self.broadcast_block(simulator_global_time, private_block, event_list=blocks_to_be_broadcasted_event_list))
                    #     self.private_blockchain_tree = []
                    #     return blocks_to_be_broadcasted_event_list


                    # Attacker lead was > 2, hence release one block (i.e. first private blockchain tree) as and when new honest blocks are added
                    else: 
                        blocks_to_be_broadcasted_event_list = []
                        private_block_to_be_broadcasted = self.private_blockchain_tree.popleft()
                        self.blockchain_tree[private_block_to_be_broadcasted[0].block_id] = private_block_to_be_broadcasted
                        return self.broadcast_block(simulator_global_time, private_block_to_be_broadcasted[0], event_list=blocks_to_be_broadcasted_event_list)


        unverified_block_flag = True

        while(unverified_block_flag and len(self.unverified_blocks)!=0):
            for unverified_block_id, unverified_block in self.unverified_blocks.items():
                if(unverified_block.previous_block_hash in self.blockchain_tree.keys()):
                    if self.verify_block(unverified_block):
                        self.blockchain_tree[unverified_block.block_id] = (unverified_block, self.blockchain_tree[unverified_block.previous_block_hash][1] + 1)
                        self.block_arrival_timing[unverified_block.block_id] = simulator_global_time
                        if(self.longest_chain_last_block['length'] < self.blockchain_tree[unverified_block.block_id][1]):
                            self.longest_chain_last_block['block'] = unverified_block
                            self.longest_chain_last_block['length'] = self.blockchain_tree[unverified_block.block_id][1]
                            if(adversary_index == self.node_id):
                                print('Private Longest chain', self.private_longest_chain_last_block)
                                print('Public longest chain', self.longest_chain_last_block)
                                if(not self.private_longest_chain_last_block):
                                    self.private_blockchain_tree = deque()
                                    self.private_longest_chain_last_block = {}
                                    return []
                            
                                attacker_lead = self.private_longest_chain_last_block['length'] - self.longest_chain_last_block['length']
                                print("Attacker lead is:", attacker_lead)
                                if(attacker_lead < 0): # Lead was 0, and became -1, hence attacker will start mining on the longest honest chain
                                    self.private_blockchain_tree = deque()
                                    self.private_longest_chain_last_block = {}
                                    return []

                                # For attacker_lead = 0 : Lead became 0 after new honest block was added, hence release the only attacker block
                                # For attacker_lead = 1 : Release both the attacker blocks
                                elif(attacker_lead == 0): 
                                    blocks_to_be_broadcasted_event_list = []
                                    self.longest_chain_last_block = copy.deepcopy(self.private_longest_chain_last_block)
                                    for private_block in self.private_blockchain_tree:
                                        self.blockchain_tree[private_block[0].block_id] = private_block
                                        blocks_to_be_broadcasted_event_list += self.broadcast_block(simulator_global_time, private_block[0], event_list=blocks_to_be_broadcasted_event_list)
                                    self.private_blockchain_tree = deque()
                                    return blocks_to_be_broadcasted_event_list

                                    ######################################################################################
                                    ########################## Check with CHAUDHARI AND GABANI  ##########################
                                    ######################################################################################
                                elif(attacker_lead == 1): # Since, the attacker has to release both the blocks hence, now we reach state 0. That's the reason we flush put private_blockchain_tree and the private_longest_chain_last_block
                                    blocks_to_be_broadcasted_event_list = []
                                    self.longest_chain_last_block = copy.deepcopy(self.private_longest_chain_last_block)
                                    for private_block in self.private_blockchain_tree:
                                        self.blockchain_tree[private_block[0].block_id] = private_block
                                        blocks_to_be_broadcasted_event_list += self.broadcast_block(simulator_global_time, private_block[0], event_list=[])
                                    self.private_blockchain_tree = deque()

                                    # self.longest_chain_last_block = self.private_longest_chain_last_block
                                    self.private_longest_chain_last_block = {} 

                                    return blocks_to_be_broadcasted_event_list

                                # elif(attacker_lead == 1): 
                                #     blocks_to_be_broadcasted_event_list = []
                                #     self.longest_chain_last_block = self.private_longest_chain_last_block
                                #     for private_block in self.private_blockchain_tree:
                                #         self.blockchain_tree[private_block[0].block_id] = private_block
                                #         blocks_to_be_broadcasted_event_list.extend(self.broadcast_block(simulator_global_time, private_block, event_list=blocks_to_be_broadcasted_event_list))
                                #     self.private_blockchain_tree = []
                                #     return blocks_to_be_broadcasted_event_list


                                # Attacker lead was > 2, hence release one block (i.e. first private blockchain tree) as and when new honest blocks are added
                                else: 
                                    blocks_to_be_broadcasted_event_list = []
                                    private_block_to_be_broadcasted = self.private_blockchain_tree.popleft()
                                    self.blockchain_tree[private_block_to_be_broadcasted[0].block_id] = private_block_to_be_broadcasted
                                    return self.broadcast_block(simulator_global_time, private_block_to_be_broadcasted[0], event_list=blocks_to_be_broadcasted_event_list)

                        del self.unverified_blocks[unverified_block_id]
                        unverified_block_flag = True
                        break

                unverified_block_flag = False
        
        # Broadcast the blocks - to the node's peers
        return self.broadcast_block(simulator_global_time, block, event_list=[])

    def broadcast_block(self, simulator_global_time, block, event_list):
        '''
            -simulator_global_time: Global time of the simulator
            -block: object of the block
            -event_list: List of all events
        '''
        for peer in self.neighbours:
            # Check if the node has already seen this block
            if peer['node'].node_id not in block.peers_visited:
                delay = peer['propagation_delay']
                bottleneck_bandwidth = peer['bottleneck_bandwidth']
                delay += (8*1000*len(block.transaction_list)/(bottleneck_bandwidth * 10**6)) # in seconds
                delay += np.random.exponential((96*1000)/(bottleneck_bandwidth * 10**6)) # d_ij in seconds
                event_list.append(Event(curr_node=peer['node'].node_id, type="BLK", event_data=block, sender_id=block.creator_id, receiver_id="all", event_start_time=simulator_global_time+delay))
        return event_list

    def verify_block(self, block):
        '''
            -block: object of the block - This block is to be verified
        '''
        block_transactions = block.transaction_list
        previous_block = self.blockchain_tree[block.previous_block_hash][0]
        previous_peer_balance = copy.deepcopy(previous_block.peer_balance)
        previous_peer_balance1 = copy.deepcopy(previous_peer_balance)

        # For each transaction in the block, check if the sender's balance >= transaction amount (Using the parent/prev block's peer_balance)
        for transaction in block_transactions:
            if(transaction.transaction_type == 'payment'):
                previous_peer_balance[transaction.sender_id] -= transaction.coins
                previous_peer_balance[transaction.receiver_id] += transaction.coins
            else:
                previous_peer_balance[transaction.receiver_id] += transaction.coins

        # For validity, check for each node if the peer_balance after applying the transactions in the blocks in non-zero
        for node_id, balance in previous_peer_balance.items():
            if(balance < 0):
                return False
        
        # Check whether the peer_balance after applying transactions on previous_peer_balance is same as the one provided in the incoming block
        if block.peer_balance != previous_peer_balance:      
            return False
        
        # Remove the verified transactions from the unverified treansactions list
        for transaction in block_transactions:
            self.verified_transactions.append(transaction)
            if transaction.transaction_id in self.unverified_txn.keys():
                del self.unverified_txn[transaction.transaction_id]
        return True
    
    def get_count_of_generated_blocks_in_longest_blockchain(self):
        count = 0
        last_block_in_blockchain_tree = copy.deepcopy(self.longest_chain_last_block['block'])
        while(last_block_in_blockchain_tree.previous_block_hash != 0):
            # Check if the cuurent miner created the current block in the longest blockchain tree, and increase the count
            if(last_block_in_blockchain_tree.block_id in self.generated_blocks):
                count += 1
            # Change the pointer to the it's parent until we reach the genesis block
            last_block_in_blockchain_tree = self.blockchain_tree[last_block_in_blockchain_tree.previous_block_hash][0]
        return count

    def visualize(self, folder, adversary_index):
        # print("visualize")
        block_map={}
        id_to_count = {}
        node_counter=0
        hash_queue=Queue()
        hash_queue.put(0)
        g=Graph('parent',filename=str(self.node_id),node_attr={'shape':'box'})
        node_counter = 0
        id_to_count = {}

        for block_id,(block,_) in self.blockchain_tree.items():
            previous_block_hash=block.previous_block_hash
            block_map.setdefault(previous_block_hash,{})[block_id]=block

        g.attr(rankdir='LR',splines='line')
        while not hash_queue.empty():
            size=hash_queue.qsize()
            
            for i in range(size):
                parent_hash = hash_queue.get()
                parent_hash_dict=block_map[parent_hash]
                for id,block in parent_hash_dict.items():
                    node_counter_str=str(node_counter)
                    if(node_counter==0):
                        g.node("G")
                        id_to_count[id]="G"

                    else:
                        if(id in attacker_blocks):
                            hi = "attacker " + id[:4]
                            color = '#FF0000'
                            g.node(hi, style='dashed', color=color)
                        else:
                            hi=id[:4]
                            g.node(hi)
                        id_to_count[id]=hi
                        # g.node(node_counter_str)
                        # id_to_count[id]=node_counter_str

                    if id in block_map:
                        hash_queue.put(id)
                    if parent_hash!=0:
                        hash_prev_block=block.previous_block_hash
                        g.edge(id_to_count[hash_prev_block],hi)
                        # g.edge(id_to_count[hash_prev_block],node_counter_str)
                    else:
                        g.node('G')
                    node_counter=node_counter+1
        g.render(folder + '/results/'+str(self.node_id),view=False) 
