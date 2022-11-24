import simpy
from modules.process.random_delay import delay

class ManualStep(simpy.Resource):
    """ This class represents the manual steps. """
    def __init__(self, name, duration, logger, env, debug=True):
        super(ManualStep, self).__init__(env, capacity=1)
        self.debug = debug
        self.duration = duration
        self.queue = 0
        self.logger = logger
        self.env = env
        self.name = name
        self.states = {'status': 0}
        

    def process(self):
        if self.debug:
            print(self.name + ": input")
        self.queue = self.queue + 1
        if (self.queue >= 5):
            self.logger.addMessage(self.name + " QUEUE_ALARM");
        with self.request() as req:
            self.states['status'] = '1' #running
            yield req

            if self.debug:
                print(self.name + ": process")
            self.queue = self.queue - 1
            self.states['status'] = '1' #running
            yield self.env.timeout(delay(self.duration, 5))

            if self.debug:
                print(self.name + ": ok")
            self.logger.addMessage(self.name + " OK");
            if self.debug:
                print(self.name + ": wait")
            self.states['status'] = '1' #running
        return

    def spawn(self):
        return self.env.process(self.process())

    def get_events(self):
        return [self.name + " QUEUE_ALARM", self.name + " OK"]
