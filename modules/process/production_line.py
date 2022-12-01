import functools
import simpy
import random
from numpy.random import normal

from modules.components.bowl_feeder import BowlFeeder
from modules.components.clock import Clock
from modules.components.conveyor import Conveyor
from modules.components.crane import Crane
from modules.components.manual_step import ManualStep
from simulator.logger import Logger
from modules.faults.wear_and_tear import WearAndTear
from modules.faults.retry_delay import RetryDelay


# Models one production line
class ProductionLine:
    def __init__(self, debug=True):
        self.env = simpy.Environment()
        self.all_modules = []
        self.process = []
        self.logger = Logger(self.env)
        self.debug = debug
        clock = Clock(self.env, self.logger)
        clock.spawn()
        self.step_no = 0
        self.sim_duration = 100000
        
        # Line State Space dictionary
        self.line_states = {'state': 0, 'time': 0, 'prod_volume_acc': 0, 'last_maint': 0, 'next_maint': 60}
        self.actions = [-20, -10, -5, 0, 5, 10, 20]
        self.action_size = len(self.actions)
        self.observation_size = 0

        # Machine States (index of machine)
            #   OK Machine State ('state': 0.waiting, 1.running, 2.faulted)
            #          - action: if machine is faulted, the whole line is stopped
            #   OK Accumulated running time ('run_acc': 0)
            #   OK last repair time ('last_repair': day no/date) OR (elapsed time since last repair (days acc) / run_acc)
            #   OK Indication of anomaly, sound, vibration ('anomaly': 0) yes/no
            #   OK hidden state: Accumulated wear

        # Production Line States
            #   OK Line State ('state': 0.stopped, 1.running, 2.line maintenance/machine repair)
            #   OK Last maintenance time ('last_maintenance': day no/date) OR (elapsed time since last maintenance (days acc))
            #   OK Planned maintenance time ('next_maintenance':day no/date) OR (time until next maintenance (days))
            #   OK accumulated production volume 'prod_volume_accum'
            #   OK current day/date 'day_acc'

        # Action space (discreet)
            #   – increase/decrease time delay to next maintenance [-10, -5, 0, 5, 10]

        # Reward Policy:
            # 1. If the line is running, the reward is 1 for each step
            # number of produced units
            # 2. If the resource is in a faulted state, the reward is -50
            # 3. if the change to next maintenance is done in less than 15 days before due, the reward is -5
            # 4. if the change to next maintenance is done in more than 10 days before due, the reward is -15


    def _create_resources(self, env, logger, debug):
                # The modules of the assembly line are created here in order, but wired together in the process definition.
        # All these are simpy resources with a limited capacity.
        self.input = simpy.resources.store.Store(self.env)
        self.crane1 = Crane("CRANE1", 30, logger, env, debug)
        self.manual_inspection = ManualStep("MANUAL_INSPECTION", 37, logger, env, debug)
        self.conveyor1 = Conveyor("CONVEYOR1", 30, logger, env, debug)
        self.bowl1 = BowlFeeder("BOWL1", 5, logger, env, debug)
        self.manual_add_components1 = ManualStep("MANUAL_ADD_COMPONENTS1", 21, logger, env, debug)
        self.conveyor2 = Conveyor("CONVEYOR2", 30, logger, env, debug)
        self.bowl2 = BowlFeeder("BOWL2", 10, logger, env, debug)
        self.manual_add_components2 = ManualStep("MANUAL_ADD_COMPONENTS2", 34, logger, env, debug)
        self.conveyor3 = Conveyor("CONVEYOR3", 30, logger, env, debug)
        self.input_subassembly_a = simpy.resources.store.Store(self.env)
        self.crane_input_subassembly_a = Crane("CRANE_INPUT_SUBASSEMBLY_A", 10, logger, env, debug)
        self.manual_combine_subassembly_a = ManualStep("MANUAL_COMBINE_SUBASSEMBLY_A", 34, logger, env, debug)
        self.conveyor4 = Conveyor("CONVEYOR4", 30, logger, env, debug)
        self.input_subassembly_b = simpy.resources.store.Store(self.env)
        self.conveyor_input_subassembly_b = Conveyor("CONVEYOR_INPUT_SUBASSEMBLY_B", 10, logger, env, debug)
        self.manual_combine_subassembly_b = ManualStep("MANUAL_COMBINE_SUBASSEMBLY_B", 35, logger, env, debug)
        self.conveyor5 = Conveyor("CONVEYOR5", 30, logger, env, debug)
        self.bowl3 = BowlFeeder("BOWL3", 5, logger, env, debug)
        self.conveyor6 = Conveyor("CONVEYOR6", 10, logger, env, debug)
        self.manual_add_cover_and_bolts = ManualStep("MANUAL_ADD_COVER_AND_BOLTS", 76, logger, env, debug)
        self.conveyor7 = Conveyor("CONVEYOR7", 30, logger, env, debug)
        self.manual_tighten_bolts1 = ManualStep("MANUAL_TIGHTEN_BOLTS1", 28, logger, env, debug)
        self.conveyor8 = Conveyor("CONVEYOR8", 30, logger, env, debug)
        self.input_subassembly_c = simpy.resources.store.Store(self.env)
        self.conveyor_input_subassembly_c = Conveyor("CONVEYOR_INPUT_SUBASSEMBLY_C", 10, logger, env, debug)
        self.manual_combine_subassembly_c = ManualStep("MANUAL_COMBINE_SUBASSEMBLY_C", 60, logger, env, debug)
        self.conveyor9 = Conveyor("CONVEYOR9", 21, logger, env, debug)
        self.manual_tighten_bolts2 = ManualStep("MANUAL_TIGHTEN_BOLTS2", 16, logger, env, debug)
        self.conveyor10 = Conveyor("CONVEYOR10", 21, logger, env, debug)
        self.bowl4 = BowlFeeder("BOWL4", 5, logger, env, debug)
        self.manual_add_components3 = ManualStep("MANUAL_ADD_COMPONENTS3", 11, logger, env, debug)
        self.conveyor11 = Conveyor("CONVEYOR11", 21, logger, env, debug)
        self.manual_tighten_bolts3 = ManualStep("MANUAL_TIGHTEN_BOLTS3", 32, logger, env, debug)
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

    def _add_process(self):
        yield self.crane1.spawn()
        yield self.manual_inspection.spawn()
        yield self.conveyor1.spawn()
        yield self.bowl1.spawn()
        yield self.manual_add_components1.spawn()
        yield self.conveyor2.spawn()
        yield self.bowl2.spawn()
        yield self.manual_add_components2.spawn()
        yield self.conveyor3.spawn()
        yield self.crane_input_subassembly_a.spawn()
        yield self.manual_combine_subassembly_a.spawn()
        yield self.conveyor3.spawn()
        yield self.conveyor_input_subassembly_b.spawn()
        yield self.manual_combine_subassembly_b.spawn()
        yield self.conveyor4.spawn()
        yield self.bowl3.spawn()
        yield self.conveyor5.spawn()
        yield self.manual_add_cover_and_bolts.spawn()
        yield self.conveyor6.spawn()
        yield self.manual_tighten_bolts1.spawn()
        yield self.conveyor7.spawn()
        yield self.conveyor_input_subassembly_c.spawn()
        yield self.manual_combine_subassembly_c.spawn()
        yield self.conveyor8.spawn()
        yield self.manual_tighten_bolts2.spawn()
        yield self.conveyor9.spawn()
        yield self.bowl4.spawn()
        yield self.manual_add_components3.spawn()
        yield self.conveyor10.spawn()
        yield self.manual_tighten_bolts3.spawn()
        yield self.conveyor11.spawn()
        print("---PROCESS COMPLETED---")

    def add_wear_and_tear_fault(self, env, production_line):
        yield env.timeout(0)
        # Simulated WearAndTear faults only apply to Conveyors.
        conveyor_to_fail = random.sample(production_line.conveyors, 1)[0]
        conveyor_to_fail.add_fault(
            WearAndTear(env, conveyor_to_fail, False));

    def add_retry_delay_fault(self, env, production_line):
        yield env.timeout(0)
        # Simulated RetryDelay faults only apply to BowlFeeders.
        bowl_feeder_to_fail = random.sample(production_line.bowl_feeders, 1)[0]
        bowl_feeder_to_fail.add_fault(
            RetryDelay(env, bowl_feeder_to_fail, False));

    def delay(duration, percentage_variation):
        stdev = percentage_variation / 100.0 * duration
        random_additive_noise = normal(0, stdev)
        return max(0, int(duration + random_additive_noise))

    def get_events(self):
        return functools.reduce(lambda a, b: a + b, [module.get_events() for module in self.all_modules], [])


    def get_resource_states(self):
        resource_states = {}
        for i in range (0, len(self.all_modules)):
            resource_states.update({self.all_modules[i].name: self.all_modules[i].states})
        return resource_states

    def get_line_state(self):
        return self.line_states

    def get_observation(self):
        # Put state dictionary items into observations list
        observation = []
        observation.extend([v for k,v in self.line_states.items()])
        for module in self.all_modules:
            observation.extend([v for k,v in module.states.items()])
        return observation


    def calculate_reward(self):
        reward = 0
        # for module in self.all_modules:
        #     reward += module.calculate_reward()

        #Reward Policy:
        # 1. If the line is running, the reward is 1 for each step
        # 2. If the resource is in a faulted state, the reward is -50
        # 3. if the change to next maintenance is done in less than 15 days before due, the reward is -5
        # 4. if the change to next maintenance is done in more than 10 days before due, the reward is -15

        return reward

    def render(self):
        #  Display state 
        print(self.get_resource_states())
        print(self.get_line_state())
      
    def reset(self):
        # Instantiate Process and add it to the SimPy environment
        self._create_resources(self.env, self.logger, self.debug)
        self.process = self.env.process(self._add_process())
        self.observation_size = len(self.get_observation())
        # Return first state observation
        obs = self.get_observation()
        return obs

    def step(self):

        # test RL algorithm if it works in case env.step() is done 10/100/1000 times for each RL step
        #FOR EACH STEP        
        # Check action is legal (raise exception if not):
            # if legal, perform action -> update maintenance schedule

        # Check if any faults are active (modules)
            # if so, update line status (stop prod)
            # conduct repair of module (resource), machine fault reset

        #check if its time for maintenance
            # if so, perform maintannance, update line status, (reset run_acc for machines)

        # update time step
        self.step_no += 1
        self.line_states['time'] = self.env.now//60
        # update time related line states
            # increase last maintenance time
            # decrease next maintenance time
        # self.line_states['next_maintenance'] -= self.time_step
        # self.line_states['last_maintenance'] += self.time_step
        # STEP SIMULATION
        #self.env.run(until=self.next_time_stop)
        self.env.step()

        # get new observations
        observations = self.get_observation()

        # check if terminal state is reached
        terminal = True if self.env.now >= self.sim_duration else False

        # Get reward
        reward = self.calculate_reward()

        # Return tuple of observations, reward, terminal
        return (observations, reward, terminal)



# TO DO
# OK design and add resource states and line states (scheduled maintannance)
# (???) add sound and vibration state
# failure states and logic to machines
# add maintannance and repair function in line (with maintannance delay time and updating state of line, reward)

# encode passing time, new day iterator event
# add time step (line state?)
# sim duration

# OK design action space

# design Rewards

# (Nice to have) alternative assambly line process (with gates and parallel machines)