import simpy
from modules.process.production_line import ProductionLine
from modules.process.fas_instance import FASInstance
from modules.components.clock import Clock
from simulator.logger import Logger

# Initializing
env = simpy.Environment()
logger = Logger(env)
production_line = ProductionLine(env, logger)
clock = Clock(env, logger)
clock.spawn()

last_item = None
# Putting in 10 items, waiting for them to be done.
for i in range(0, 10):
    fas_instance = FASInstance(env, production_line, logger)
    last_item = fas_instance.spawn()

env.run(last_item)
print("Done.")
print(logger.getLoglines())
# f = open("data/output_easy.json", "w")
# f.write(logger.getLoglines())
# f.close()
