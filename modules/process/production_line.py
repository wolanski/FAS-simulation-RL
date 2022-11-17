import functools

import simpy

from modules.components.bowl_feeder import BowlFeeder
from modules.components.conveyor import Conveyor
from modules.components.crane import Crane
from modules.components.manual_step import ManualStep
from modules.process.random_delay import delay


# Models one production line
class ProductionLine:
    def __init__(self, env, logger, debug=True):
        self.env = env
        # The modules of the assembly line are created here in order, but wired together in the process definition.
        # All these are simpy resources with a limited capacity.
        self.input = simpy.resources.store.Store(env)
        self.crane1 = Crane("CRANE1", 30000, logger, env, debug)
        self.manual_inspection = ManualStep("MANUAL_INSPECTION", 37000, logger, env, debug)
        self.conveyor1 = Conveyor("CONVEYOR1", 30000, logger, env, debug)
        self.bowl1 = BowlFeeder("BOWL1", 5000, logger, env, debug)
        self.manual_add_components1 = ManualStep("MANUAL_ADD_COMPONENTS1", 21000, logger, env, debug)
        self.conveyor2 = Conveyor("CONVEYOR2", 30000, logger, env, debug)
        self.bowl2 = BowlFeeder("BOWL2", 10000, logger, env, debug)
        self.manual_add_components2 = ManualStep("MANUAL_ADD_COMPONENTS2", 34000, logger, env, debug)
        self.conveyor3 = Conveyor("CONVEYOR3", 30000, logger, env, debug)
        self.input_subassembly_a = simpy.resources.store.Store(env)
        self.crane_input_subassembly_a = Crane("CRANE_INPUT_SUBASSEMBLY_A", 10000, logger, env, debug)
        self.manual_combine_subassembly_a = ManualStep("MANUAL_COMBINE_SUBASSEMBLY_A", 34000, logger, env, debug)
        self.conveyor4 = Conveyor("CONVEYOR4", 30000, logger, env, debug)
        self.input_subassembly_b = simpy.resources.store.Store(env)
        self.conveyor_input_subassembly_b = Conveyor("CONVEYOR_INPUT_SUBASSEMBLY_B", 10000, logger, env, debug)
        self.manual_combine_subassembly_b = ManualStep("MANUAL_COMBINE_SUBASSEMBLY_B", 35000, logger, env, debug)
        self.conveyor5 = Conveyor("CONVEYOR5", 30000, logger, env, debug)
        self.bowl3 = BowlFeeder("BOWL3", 5000, logger, env, debug)
        self.conveyor6 = Conveyor("CONVEYOR6", 10000, logger, env, debug)
        self.manual_add_cover_and_bolts = ManualStep("MANUAL_ADD_COVER_AND_BOLTS", 76000, logger, env, debug)
        self.conveyor7 = Conveyor("CONVEYOR7", 30000, logger, env, debug)
        self.manual_tighten_bolts1 = ManualStep("MANUAL_TIGHTEN_BOLTS1", 28000, logger, env, debug)
        self.conveyor8 = Conveyor("CONVEYOR8", 30000, logger, env, debug)
        self.input_subassembly_c = simpy.resources.store.Store(env)
        self.conveyor_input_subassembly_c = Conveyor("CONVEYOR_INPUT_SUBASSEMBLY_C", 10000, logger, env, debug)
        self.manual_combine_subassembly_c = ManualStep("MANUAL_COMBINE_SUBASSEMBLY_C", 60000, logger, env, debug)
        self.conveyor9 = Conveyor("CONVEYOR9", 21000, logger, env, debug)
        self.manual_tighten_bolts2 = ManualStep("MANUAL_TIGHTEN_BOLTS2", 16000, logger, env, debug)
        self.conveyor10 = Conveyor("CONVEYOR10", 21000, logger, env, debug)
        self.bowl4 = BowlFeeder("BOWL4", 5000, logger, env, debug)
        self.manual_add_components3 = ManualStep("MANUAL_ADD_COMPONENTS3", 11000, logger, env, debug)
        self.conveyor11 = Conveyor("CONVEYOR11", 21000, logger, env, debug)
        self.manual_tighten_bolts3 = ManualStep("MANUAL_TIGHTEN_BOLTS3", 32000, logger, env, debug)
        self.output = simpy.resources.store.Store(env)

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
        
        # Set up observation and action space sizes
        self.observation_size = 5
        self.action_size = 5
        
        # Set up State Space dictionary
        self.resource_state = {}
        resource_id = ['conveyor', 'machine1', 'machine2']
        resource_states = {'status': '', 'last_maintannance': '', 'anomaly': ''}
        for i in range (0, len(resource_id)): self.resource_state.update({resource_id[i]: dict(resource_states)})
        # Machine States (index of machine)
        #   – Machine status (work in progress, failure, wait, maintenance) – Remaining time for current job
        #   – Time for remaining operations in the queue
        #   – elapsed time since last maintenance
        #   – Indication of anomaly (sound, vibration)
        
        self.line_state = {}
        self.line_state['prod_volume_instant'] = 0
        self.line_state['Prod_volume_accum'] = 0
        self.line_state['last_maintannance'] = 0
        # Global (System) States
        #   – instant production volume
        #   – accumulated production volume
        #   – elapsed time since last maintenance
        
        # Set up Action space
        self.actions = []
        # Action space (discreet)
        #   – Schedule Planned Maintenance (for several components)
        #   – Schedule urgent maintenance for specific machine


    def get_events(self):
        return functools.reduce(lambda a, b: a + b, [module.get_events() for module in self.all_modules], [])
    
    
    # Pass resource (machine) to be run and update State
    def run_resource(self, resource):
      print("RUN RESOURCE")
      resource.update_state() # self.resource_state[] = resource.get_state()
      # production line update state
      #self.line_state = 
      return resource.spawn()
      # add update State
      
    # RL Interfacing Methods
    # render:
    #   Display state 
    def render(self):
      print("render not implemented")
      
    #  reset:
    #    Initialise environment
    #    Return first state observations
    def reset(self):
      print("reset not implemented")
      
    #  step:
    #  * A step method takes and passes an action to the environment and returns:
      #  1. the state new observations (update state, returns observation...)
      #  2. reward
      #  3. whether state is terminal (A way to recognise and return a terminal state (end of episode))
      #  4. additional information
    def step(self):
      print("step not implemented")
