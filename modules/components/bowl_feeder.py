import simpy
from modules.process.random_delay import delay

class BowlFeeder(simpy.Resource):
    """
    This class represents the bowl feeders giving parts to the manual
    assembly steps.
    Simulated RetryDelay faults only apply to BowlFeeders.
    """
    def __init__(self, name, duration, logger, env, debug=True):
        super(BowlFeeder, self).__init__(env, capacity=1)
        self.debug = debug
        self.duration = duration
        self.logger = logger
        self.env = env
        self.name = name
        self.states = {'status': 0, 'run_acc': 0, 'fault': 0}
        self.hidden_states = {'accumulated_wear': 0}
        self.faults = []

    def add_fault(self, fault):
        self.faults.append(fault)

    def process(self):
        with self.request() as req:
            self.states['status'] = 1 #running
            yield req

            if self.debug:
                print(self.name + ": give")
            self.states['status'] = 1 #running
            yield self.env.timeout(delay(self.duration, 1))

            for fault in self.faults:
                self.states['fault'] = 1 #fault
                self.states['status'] = 0 #waiting
                yield fault.spawn()

            if self.debug:
                print(self.name + ": given")
            self.states['fault'] = 0 #no fault
            self.states['status'] = 1 #running
            self.logger.addMessage(self.name + " GIVEN");
            self.states['run_acc'] += 1
        return

    def spawn(self):
        return self.env.process(self.process())

    def get_events(self):
        return [self.name + " GIVEN"]
