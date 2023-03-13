import hashlib
class Block:
    def __init__(self, creator_id, creation_time, peer_balance, transaction_list, previous_block_hash):
        '''
            -creator_id: ID of the miner
            -creation_time: Creation time of the block
            -peer_balance: Peer balance of all the nodes after processing of the block
            -transaction_list: List of all transactions in the block (including coinbase transaction)
            -previous_block_hash: Block hash of the previous block i.e. parent block hash
        '''
        self.creator_id = creator_id
        self.creation_time = creation_time
        self.transaction_list = transaction_list
        self.previous_block_hash = previous_block_hash
        self.peer_balance = peer_balance
        self.block_id = self.set_id()
        self.peers_visited = []

    # Create the block id by hashing the conccatenation of transaction message and the previous block hash
    def set_id(self):
        result = " ".join([txn.transaction_message for txn in self.transaction_list]) + " " + str(self.previous_block_hash)
        return hashlib.sha256(result.encode('utf-8')).hexdigest()
