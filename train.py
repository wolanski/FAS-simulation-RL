from modules.process.production_line import ProductionLine


# Initializing
production_line = ProductionLine()

# Putting in 10 items, waiting for them to be done.
# for i in range(0, 1):
#   fas_instance = FASInstance(env, production_line, logger)
#   last_item = fas_instance.spawn()

# env.run(last_item)

#print(logger.getLoglines())

production_line.reset()

production_line.step()
production_line.step()
production_line.step()
