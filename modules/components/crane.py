import simpy
from modules.process.random_delay import delay

class Crane(simpy.Resource):
    """ This class represents the cranes. They transport one item at a time. """
    def __init__(self, name, duration, logger, env, debug=True):
        super(Crane, self).__init__(env, capacity=1)
        self.debug = debug
        self.duration = duration
        self.queue = 0
        self.logger = logger
        self.env = env
        self.name = name
        self.states = {'state': 0, 'run_acc': 0, 'last_repair': 0, 'anomaly': 0}
        self.hidden_states = {'accumulated_wear': 0}
        

    def process(self):
        if self.debug:
            print(self.name + ": input")
        self.queue = self.queue + 1
        with self.request() as req:
            self.states['state'] = 1 #running
            yield req

            if self.debug:
                print(self.name + ": go_forward")
            self.logger.addMessage(self.name + " FORWARD");
            self.queue = self.queue - 1
            self.states['state'] = 1 #running
            yield self.env.timeout(delay(self.duration, 1))

            if self.debug:
                print(self.name + ": wait")
            if self.debug:
                print(self.name + ": item_taken")
                print(self.name + ": go_back")
            self.logger.addMessage(self.name + " BACKWARD");
            self.states['state'] = 1 #running
            yield self.env.timeout(delay(self.duration, 1))

            if self.debug:
                print(self.name + ": stop")
            self.logger.addMessage(self.name + " STOP");
            self.states['state'] = 0 #waiting
            self.states['run_acc'] += 1
        return

    def spawn(self):
        return self.env.process(self.process())

    def get_events(self):
        return [self.name + " FORWARD", self.name + " BACKWARD",
                self.name + " STOP"]

    def repair(self):
        pass

    def do_maintenance(self):
        pass
