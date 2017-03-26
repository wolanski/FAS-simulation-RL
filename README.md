# FAS Simulator
A modular benchmark simulator for flexible assembly systems.

Licence: WTFPL

This project includes a SimPy discrete event simulator simulating a plausible Flexible Assembly System
documented in the [documentation/FAS-Simulator.pdf](https://github.com/keskival/FAS-Simulator/raw/master/documentation/FAS-Simulator.pdf).

It contains several pre-configured scripts for different kinds of production runs, named: `run*.py`

Example: Running a simple simulation with a simulated wear and tear fault:
`./run_easy_with_fault.py`

Running the simulations produces output to the STDOUT, but the actual output is written as JSON to [output.json](https://github.com/keskival/FAS-Simulator/blob/master/output.json).

The output contains a sequence of events with timestamps.

Additionally, there are several `*.m` files and `*.sh` files to create different kinds of visualizations
out of this JSON output using Octave and ffmepg.

The `data.mat` file is created using `./output_to_octave.py`
from `output.json` to `data.mat`.

There is also a related FAS-Tensorflow project (private at the moment) that is an implementation that extracts process model
features from inputs
generated by this project using deep learning methods, and uses those for anomaly detection.

TODO: Should clean up the structure of this repository.
