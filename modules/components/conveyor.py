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
            while self.states['state'] != 2:
                self.states['state'] = 1 #running
                yield req

                if self.debug:
                    print(self.name + ": input")
                self.logger.addMessage(self.name + " CONVEYOR_GATE");
                #self.states['state'] = 1 #running
                yield self.env.timeout(delay(self.duration, 1))

                # for fault in self.faults:
                #     self.states['state'] = 2 #fault
                #     yield fault.spawn()

                if self.hidden_states['accumulated_wear'] > 300:
                    self.states['state'] = 2 #fault
                    return self.fault()

                #self.states['state'] = 1 #running
                if self.debug:
                    print(self.name + ": to_next_step")
                self.states['run_acc'] += 1
                self.add_wear()
        return


    def add_wear(self):
        # accumulate wear
        self.hidden_states['accumulated_wear'] += 1

    def fault(self):
        delay_factor = 600 #(exp(self.t/5.0) - 1 ) / 30
        extra_delay = delay_factor * self.duration
        if self.debug:
            print("FAULT: %s, extra delay: %s" % (self.name, extra_delay))
        return self.env.timeout(extra_delay)

    # def add_fault(self, fault):
    #     self.faults.append(fault)

    def repair(self):
        self.states['run_acc'] = 0 #maybe do not reset?
        self.hidden_states['accumulated_wear'] = 0
        self.states['last_repair'] = self.env.now//60
        repair_delay = 1500
        self.states['state'] = 0 #waiting
        if self.debug:
            print("REPAIR: %s, delay: %s" % (self.name, repair_delay))
        return self.env.timeout(repair_delay)

    def do_maintainence(self):
        self.states['run_acc'] = 0    #maybe do not reset?
        self.hidden_states['accumulated_wear'] = 0     #maybe do not reset completely?
        self.states['last_repair'] = self.env.now//60     #maybe do not reset?
        #self.states['state'] = 1 #running


    def get_events(self):
        return [self.name + " CONVEYOR_GATE"]

    def spawn(self):
        return self.env.process(self.process())