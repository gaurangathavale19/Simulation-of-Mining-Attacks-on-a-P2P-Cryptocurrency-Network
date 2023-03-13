class Event:

    def __init__(self, curr_node, type, event_data, sender_id, receiver_id, event_start_time):
        '''
            -curr_node: Current location of the node, where the event is present
            -type: Type of event i.e. BLK or TXN
            -event_data: object of the event
            -sender_id: ID of the sender node
            -receiver_id: ID of the receiver node
            -event_start_time: Event creation time
        '''
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.curr_node = curr_node
        self.event_start_time = event_start_time
        self.event_data = event_data
        self.type = type

    # For min-heap comparisons
    def __lt__(self, event):
        return self.event_start_time < event.event_start_time

