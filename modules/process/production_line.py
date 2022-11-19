import functools
import simpy

from modules.components.bowl_feeder import BowlFeeder
from modules.components.clock import Clock
from modules.components.conveyor import Conveyor
from modules.components.crane import Crane
from modules.components.manual_step import ManualStep
from simulator.logger import Logger 


# Models one production line
class ProductionLine:
    def __init__(self, debug=True):
        self.env = simpy.Environment()
        self.process = []
        logger = Logger(self.env)
        clock = Clock(self.env, logger)
        clock.spawn()
        
        # The modules of the assembly line are created here in order, but wired together in the process definition.
        # All these are simpy resources with a limited capacity.
        self.input = simpy.resources.store.Store(self.env)
        self.crane1 = Crane("CRANE1", 30000, logger, self.env, debug)
        self.manual_inspection = ManualStep("MANUAL_INSPECTION", 37000, logger, self.env, debug)
        self.conveyor1 = Conveyor("CONVEYOR1", 30000, logger, self.env, debug)
        self.bowl1 = BowlFeeder("BOWL1", 5000, logger, self.env, debug)
        self.manual_add_components1 = ManualStep("MANUAL_ADD_COMPONENTS1", 21000, logger, self.env, debug)
        self.conveyor2 = Conveyor("CONVEYOR2", 30000, logger, self.env, debug)
        self.bowl2 = BowlFeeder("BOWL2", 10000, logger, self.env, debug)
        self.manual_add_components2 = ManualStep("MANUAL_ADD_COMPONENTS2", 34000, logger, self.env, debug)
        self.conveyor3 = Conveyor("CONVEYOR3", 30000, logger, self.env, debug)
        self.input_subassembly_a = simpy.resources.store.Store(self.env)
        self.crane_input_subassembly_a = Crane("CRANE_INPUT_SUBASSEMBLY_A", 10000, logger, self.env, debug)
        self.manual_combine_subassembly_a = ManualStep("MANUAL_COMBINE_SUBASSEMBLY_A", 34000, logger, self.env, debug)
        self.conveyor4 = Conveyor("CONVEYOR4", 30000, logger, self.env, debug)
        self.input_subassembly_b = simpy.resources.store.Store(self.env)
        self.conveyor_input_subassembly_b = Conveyor("CONVEYOR_INPUT_SUBASSEMBLY_B", 10000, logger, self.env, debug)
        self.manual_combine_subassembly_b = ManualStep("MANUAL_COMBINE_SUBASSEMBLY_B", 35000, logger, self.env, debug)
        self.conveyor5 = Conveyor("CONVEYOR5", 30000, logger, self.env, debug)
        self.bowl3 = BowlFeeder("BOWL3", 5000, logger, self.env, debug)
        self.conveyor6 = Conveyor("CONVEYOR6", 10000, logger, self.env, debug)
        self.manual_add_cover_and_bolts = ManualStep("MANUAL_ADD_COVER_AND_BOLTS", 76000, logger, self.env, debug)
        self.conveyor7 = Conveyor("CONVEYOR7", 30000, logger, self.env, debug)
        self.manual_tighten_bolts1 = ManualStep("MANUAL_TIGHTEN_BOLTS1", 28000, logger, self.env, debug)
        self.conveyor8 = Conveyor("CONVEYOR8", 30000, logger, self.env, debug)
        self.input_subassembly_c = simpy.resources.store.Store(self.env)
        self.conveyor_input_subassembly_c = Conveyor("CONVEYOR_INPUT_SUBASSEMBLY_C", 10000, logger, self.env, debug)
        self.manual_combine_subassembly_c = ManualStep("MANUAL_COMBINE_SUBASSEMBLY_C", 60000, logger, self.env, debug)
        self.conveyor9 = Conveyor("CONVEYOR9", 21000, logger, self.env, debug)
        self.manual_tighten_bolts2 = ManualStep("MANUAL_TIGHTEN_BOLTS2", 16000, logger, self.env, debug)
        self.conveyor10 = Conveyor("CONVEYOR10", 21000, logger, self.env, debug)
        self.bowl4 = BowlFeeder("BOWL4", 5000, logger, self.env, debug)
        self.manual_add_components3 = ManualStep("MANUAL_ADD_COMPONENTS3", 11000, logger, self.env, debug)
        self.conveyor11 = Conveyor("CONVEYOR11", 21000, logger, self.env, debug)
        self.manual_tighten_bolts3 = ManualStep("MANUAL_TIGHTEN_BOLTS3", 32000, logger, self.env, debug)
        self.output = simpy.resources.store.Store(self.env)

        self.conveyors = [self.conveyor1, self.conveyor2, self.conveyor3,
            self.conveyor4, self.conveyor5, self.conveyor6, self.conveyor7,
            self.conveyor8, self.conveyor9, self.conveyor10, self.conveyor11,
            self.conveyor_input_subassembly_b,
            self.conveyor_input_subassembly_c]
        self.bowl_feeders = [self.bowl1, self.bowl2, self.bowl3, self.bowl4]
        self.cranes = [self.crane1, self.crane_input_subassembly_a]
        self.manual_steps = [self.manual_inspection,
            self.manual_add_components1, self.manual_add_components2,
            self.manual_combine_subassembly_a,
            self.manual_combine_subassembly_b,
            self.manual_add_cover_and_bolts,
            self.manual_tighten_bolts1,
            self.manual_combine_subassembly_c,
            self.manual_tighten_bolts2,
            self.manual_add_components3,
            self.manual_tighten_bolts3
            ]
        self.all_modules = self.conveyors + self.bowl_feeders + self.cranes + self.manual_steps
        
        # Line State Space dictionary
        self.line_states = {'prod_volume_instant': '3', 'Prod_volume_accum': '', 'last_maintannance': ''}
        self.observation_size = len(self.get_observations())
        # Machine States (index of machine)
            #   – Machine status (work in progress, failure, wait, maintenance) 
            #   – Remaining time for current job
            #   – Time for remaining operations in the queue
            #   – elapsed time since last maintenance
            #   – Indication of anomaly (sound, vibration)
        # Global (System) States
            #   – instant production volume
            #   – accumulated production volume
            #   – elapsed time since last maintenance
        
        # Action space (discreet)
            #   – Schedule Planned Maintenance (for several components)
            #   – Schedule urgent maintenance for specific machine
        self.action_size = 5
        self.actions = []


    def get_events(self):
        return functools.reduce(lambda a, b: a + b, [module.get_events() for module in self.all_modules], [])


    def _process(self):
        yield self._run_resource(self.crane1)
        yield self._run_resource(self.conveyor1)
        #yield self.run_resource(self.manual_inspection.spawn)

        # yield self.pl.conveyor1.spawn()
        # yield self.pl.bowl1.spawn()
        # yield self.pl.manual_add_components1.spawn()
        # yield self.pl.conveyor2.spawn()
        # yield self.pl.bowl2.spawn()
        # yield self.pl.manual_add_components2.spawn()
        # yield self.pl.conveyor3.spawn()
        # yield self.pl.crane_input_subassembly_a.spawn()
        # yield self.pl.manual_combine_subassembly_a.spawn()
        # yield self.pl.conveyor3.spawn()
        # yield self.pl.conveyor_input_subassembly_b.spawn()
        # yield self.pl.manual_combine_subassembly_b.spawn()
        # yield self.pl.conveyor4.spawn()
        # yield self.pl.bowl3.spawn()
        # yield self.pl.conveyor5.spawn()
        # yield self.pl.manual_add_cover_and_bolts.spawn()
        # yield self.pl.conveyor6.spawn()
        # yield self.pl.manual_tighten_bolts1.spawn()
        # yield self.pl.conveyor7.spawn()
        # yield self.pl.conveyor_input_subassembly_c.spawn()
        # yield self.pl.manual_combine_subassembly_c.spawn()
        # yield self.pl.conveyor8.spawn()
        # yield self.pl.manual_tighten_bolts2.spawn()
        # yield self.pl.conveyor9.spawn()
        # yield self.pl.bowl4.spawn()
        # yield self.pl.manual_add_components3.spawn()
        # yield self.pl.conveyor10.spawn()
        # yield self.pl.manual_tighten_bolts3.spawn()
        # yield self.pl.conveyor11.spawn()
        print("---PROCESS COMPLETED---")

    # Pass resource (machine) to be run and update State
    def _run_resource(self, resource):
        print("RUN RESOURCE")
        resource.get_state() # self.resource_state[] = resource.get_state()
        # production line update state
        #self.line_state = 
        return resource.spawn()
        # add update State

    def get_states(self):
        resource_states = {}
        for i in range (0, len(self.all_modules)):
            resource_states.update({self.all_modules[i].name: self.all_modules[i].states})
        return resource_states


    def get_observations(self):
        # Put state dictionary items into observations list
        observation = []
        observation.extend([v for k,v in self.line_states.items()])
        for module in self.all_modules:
            observation.extend([v for k,v in module.states.items()])
        return observation



    # RL Interfacing Methods
    # render:
      #   Display state 
    def render(self):
        print("render not implemented")
      
    #  reset:
      #    Return first state observations
    def reset(self):
        self.process = self.env.process(self._process())
        self.states.get_observations()
      
    #  step:
      #  * A step method takes and passes an action to the environment and returns:
        #  1. the state new observations (update state, returns observation...)
        #  2. reward
        #  3. whether state is terminal (A way to recognise and return a terminal state (end of episode))
        #  4. additional information
    def step(self):
        #self.env.run(until=self.process)
        until = 10
        # while self.env.peek() < until:
        #     self.env.step()
        
        self.env.step()
        print(self.states.line_state['prod_volume_instant'])
        self.env.step()
        print(self.states.line_state['prod_volume_instant'])
        self.env.step()
        print(self.states.line_state['prod_volume_instant'])
        self.env.step()
        print(self.states.line_state['prod_volume_instant'])
        self.env.step()
        print(self.states.line_state['prod_volume_instant'])
        self.env.step()
        print(self.states.line_state['prod_volume_instant'])

