Distributionally Robust Optimization with Unscented Transform for Learning-Based Motion Control in Dynamic Environments.
====================================================

This repository includes the source code for implementing the distributionally robust UT-MPC algorithm with all the baselines presented in the paper.

## Requirements
- CARLA simulator
- Python (>= 3.5)
- Forces Pro
- GPy
- Python packages, such as numpy, scipy, matplotlib, etc.
- casadi

## Usage

To run the experiments, first execute the CARLA simulator:
```
cd path/to/carla/root
./CarlaUE4.sh
```

Then, in another terminal, call the `Town10HD` map, which is used in our simulations.
```
cd path/to/carla/root/PythonAPI/util/config.py --map Town10HD
```

Finally, call the main script
```
python run.py
```

The simulation parameters can be change by adding additional command-line arguments:
- `--with_obs` - add an obstacle to the simulations.
- `--method` - chooses the MPC algorithm to execute and ccepts one of the following values: 

	- `cvar` - CVaR-constrained learning-based MPC,
	- `mean` - Mean-constrained learning-based MPC,
	- `drcvar` - Distributionally robust UT-MPC.

- `--gp` - uses GPR for learing the unknown models. If not invoked, the nominal model is used for the ego vehicle, while the obstacle is assumed to be static.
- `--record`  - records the simulation snapshots from the PyGame window.

The results of simulations are saved in separate pickle files in `/data` folder.
