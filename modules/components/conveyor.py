import simpy
from modules.process.random_delay import delay

class Conveyor(simpy.Resource):
    """
    This class represents the conveyors. The difference to cranes is that conveyors do not wait
    before processing the next item. There is no queue for the conveyor.
    Simulated WearAndTear faults only apply to Conveyors.
    """
    def __init__(self, name, duration, logger, env, debug=True):
        super(Conveyor, self).__init__(env)
        self.debug = debug
        self.duration = duration
        self.logger = logger
        self.env = env
        self.name = name
        self.states = {'state': 0, 'run_acc': 0, 'last_repair': 0, 'anomaly': 0}
        self.hidden_states = {'accumulated_wear': 0}
        self.faults = []

    def process(self):
        with self.request() as req:
            self.states['state'] = 1 #running
            yield req

            if self.debug:
                print(self.name + ": input")
            self.logger.addMessage(self.name + " CONVEYOR_GATE");
            self.states['state'] = 1 #running
            yield self.env.timeout(delay(self.duration, 1))

            for fault in self.faults:
                self.states['state'] = 2 #fault
                yield fault.spawn()

            self.states['state'] = 1 #running
            if self.debug:
                print(self.name + ": to_next_step")
            self.states['run_acc'] += 1
        return


    def get_events(self):
        return [self.name + " CONVEYOR_GATE"]

    def add_wear(self):
        # accumulate wear
        # self.faults.append(fault)
        #print("FAULT, module: %s" % module.name)
        pass

    def add_fault(self, fault):
        self.faults.append(fault)

    def repair(self):

        pass

    def do_maintainence(self):
        pass

    def spawn(self):
        return self.env.process(self.process())