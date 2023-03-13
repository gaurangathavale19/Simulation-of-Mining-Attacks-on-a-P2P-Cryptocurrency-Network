import hashlib
class Transaction:
    def __init__(self, sender_id, receiver_id, coins, transaction_type, timestamp):
        '''
            -sender_id: ID of the sender node
            -receiver_id: ID of the receiver node
            -coins: Amount (in BTC) to be transferred
            -transaction_type: Type of transaction i.e. payment/mines/init
            -timestamp: creatio time of the transaction
        '''
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.coins = coins
        self.timestamp = timestamp
        self.transaction_type = transaction_type
        if(self.transaction_type == "payment"):
            self.transaction_message = "{} pays {} {} coins".format(sender_id, receiver_id, coins)
        elif(self.transaction_type == "mines"):
            self.transaction_message = "{} mines {} coins".format(receiver_id, coins)
        else:

            self.transaction_message = "{} init {} coins".format(receiver_id, coins)
        self.transaction_id = hashlib.sha256((str(timestamp)+self.transaction_message).encode('utf-8')).hexdigest()

