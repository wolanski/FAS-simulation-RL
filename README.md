# FAS Simulator
A modular benchmark simulator for flexible assembly systems.

Licence: WTFPL

This project includes a SimPy discrete event simulator simulating a plausible Flexible Assembly System
documented in the [documentation/FAS-Simulator.pdf](https://github.com/keskival/FAS-Simulator/raw/master/documentation/FAS-Simulator.pdf).

See Executive summary poster in: [documentation/ExecutiveSummary.pdf](https://github.com/keskival/FAS-Simulator/raw/master/documentation/ExecutiveSummary.pdf).

It contains several pre-configured scripts for different kinds of production runs, named: `run*.py`

See Python dependencies from `dependencies.txt`. For plotting things you need Octave and epstool.

Example: Running a simple simulation with a simulated wear and tear fault:
`./run_easy_with_wear_and_tear_fault.py`

Running the simulations produces output to the STDOUT, but the actual output is written as JSON to [output.json](https://github.com/keskival/FAS-Simulator/blob/master/data/output_easy_with_wear_and_tear_fault.json).

The output contains a sequence of events with timestamps.

Additionally, there are several `*.m` files and `*.sh` files in `utils` directory to create different kinds of visualizations
out of this JSON output using Octave and ffmepg.

The `data.mat` file is created using `./utils/output_to_octave.py`
from `output.json` to `data.mat`.

There is also a related FAS-Tensorflow project (private at the moment) that is an implementation that extracts process model
features from inputs
generated by this project using deep learning methods, and uses those for anomaly detection.

## The Default ASsembly Process Being Simulated


Step | Description | Duration | Log messages
--- | --- | --- | ---
1 | Crane | 30 s | Going forward, stopping, going back, stopping

2 | Manual inspection | 37 s | OK pressed, queue alarm

3 | Conveyor | 30 s | To station, stop, to next station

4 | Bowl feeder gives components | 5 s | Given

5 | Add components | 21 s | OK pressed, queue alarm

6 | Conveyor | 30 s | To station, stop, to next station

7 | Bowl feeder gives components | 10 s | Given

8 | Add components | 34 s | OK pressed, queue alarm

9 | Conveyor | 30 s | To station, stop, to next station

10 | Crane with subassembly A | 10 s | Going forward, stopping, going back, stopping

11 | Combine with subassembly A | 34 s | OK pressed, queue alarm

12 | Conveyor | 30 s | To station, stop, to next station

13 | Conveyor with subassembly B | 10 s | To station, stop

14 | Combine with subassembly B | 35 s | OK pressed, queue alarm

15 | Conveyor | 30 s | To station, stop, to next station

16 | Bowl feeder gives components | 5 s | Given

17 | Conveyor with cover | 10 s | To station, stop

18 | Add cover and bolts | 76 s | OK pressed, queue alarm

19 | Conveyor | 30 s | To station, stop, to next station

20 | Tighten the bolts | 28 s | OK pressed, queue alarm

21 | Conveyor | 30 s | To station, stop, to next station

22 | Conveyor with subassembly C | 10 s | To station, stop

23 | Combine with subassembly C | 60 s | OK pressed, queue alarm

24 | Conveyor | 21 s | To station, stop, to next station

25 | Tighten the bolts | 16 s | OK pressed, queue alarm

26 | Conveyor | 21 s | To station, stop, to next station

27 | Bowl feeder gives components | 5 s | Given

28 | Add components | 11 s | OK pressed, queue alarm

29 | Conveyor | 21 s | To station, stop, to next station

30 | Tighten the bolts | 32 s | OK pressed, queue alarm

31 | Conveyor | 21 s | To output gate

