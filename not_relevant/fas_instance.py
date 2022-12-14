import simpy
from modules.process.random_delay import delay

# Models one assembly process instance
class FASInstance:
    def __init__(self, env, production_line, logger):
        self.env = env
        self.pl = production_line
        self.logger = logger
        self.input = env.event()
        
    # def _step_(self):
    #     pass

    def process(self):
        yield self.pl.run_resource(self.pl.crane1)
        
        #yield self.pl.crane1.spawn()
        yield self.pl.manual_inspection.spawn()
        yield self.pl.conveyor1.spawn()
        yield self.pl.bowl1.spawn()
        yield self.pl.manual_add_components1.spawn()
        yield self.pl.conveyor2.spawn()
        yield self.pl.bowl2.spawn()
        yield self.pl.manual_add_components2.spawn()
        yield self.pl.conveyor3.spawn()
        yield self.pl.crane_input_subassembly_a.spawn()
        yield self.pl.manual_combine_subassembly_a.spawn()
        yield self.pl.conveyor3.spawn()
        yield self.pl.conveyor_input_subassembly_b.spawn()
        yield self.pl.manual_combine_subassembly_b.spawn()
        yield self.pl.conveyor4.spawn()
        yield self.pl.bowl3.spawn()
        yield self.pl.conveyor5.spawn()
        yield self.pl.manual_add_cover_and_bolts.spawn()
        yield self.pl.conveyor6.spawn()
        yield self.pl.manual_tighten_bolts1.spawn()
        yield self.pl.conveyor7.spawn()
        yield self.pl.conveyor_input_subassembly_c.spawn()
        yield self.pl.manual_combine_subassembly_c.spawn()
        yield self.pl.conveyor8.spawn()
        yield self.pl.manual_tighten_bolts2.spawn()
        yield self.pl.conveyor9.spawn()
        yield self.pl.bowl4.spawn()
        yield self.pl.manual_add_components3.spawn()
        yield self.pl.conveyor10.spawn()
        yield self.pl.manual_tighten_bolts3.spawn()
        yield self.pl.conveyor11.spawn()
        print("---PROCESS COMPLETED---")

    def spawn(self):
        return self.env.process(self.process())
