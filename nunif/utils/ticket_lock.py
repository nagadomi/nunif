import threading


class TicketLock():
    def __init__(self):
        self.condition = threading.Condition()
        self.tickets = []
        self.ticket_id = 0

    def new_ticket(self):
        with self.condition:
            ticket_id = self.ticket_id
            self.tickets.append(ticket_id)
            self.ticket_id += 1
        return ticket_id

    def __call__(self, ticket_id):
        return _TicketLockContext(self, ticket_id)

    def _wait(self, ticket_id):
        self.condition.acquire()
        assert ticket_id in self.tickets
        while self.tickets[0] != ticket_id:
            self.condition.wait()

    def _release(self, ticket_id):
        if self.tickets:
            relesed_ticket = self.tickets.pop(0)
        else:
            relesed_ticket = None
        self.condition.notify_all()
        self.condition.release()
        assert relesed_ticket is not None and relesed_ticket == ticket_id

    def __enter__(self):
        return self.condition.__enter__()

    def __exit__(self, *args):
        return self.condition.__exit__(*args)


class _TicketLockContext():
    def __init__(self, ticket_lock, ticket_id):
        self.ticket_lock = ticket_lock
        self.ticket_id = ticket_id

    def __enter__(self):
        self.ticket_lock._wait(self.ticket_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ticket_lock._release(self.ticket_id)
        return False
