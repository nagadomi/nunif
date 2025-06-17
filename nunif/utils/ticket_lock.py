import threading


class TicketLock():
    def __init__(self):
        self.condtion = threading.Condition()  # RLock
        self.tickets = []
        self.ticket_id = 0

    def new_ticket(self):
        with self.condtion:
            ticket_id = self.ticket_id
            self.tickets.append(ticket_id)
            self.ticket_id += 1
        return ticket_id

    def wait(self, ticket_id):
        with self.condtion:
            assert len(self.tickets) > 0
            while self.tickets[0] != ticket_id:
                self.condtion.wait()

    def release(self, ticket_id):
        with self.condtion:
            assert len(self.tickets) > 0
            assert self.tickets[0] == ticket_id
            self.tickets.pop(0)
            self.condtion.notify_all()

    def __enter__(self):
        return self.condtion.__enter__()

    def __exit__(self, *args):
        return self.condtion.__exit__(*args)
