# OPT_TOOLS
This repo contains helper functions to interface between:
 - ROS, 
 - numerical optimization libraries (e.g. IPOPT),
 - autodiff frameworks
mostly oriented towards building online optimal control problems such as MPC.

## Feature tracker

| Feature | JAX | CasADi |
|---------|-----|--------|
| f(x,p) objective | [x] | [x]  |
| p parameters | [x] | [x] |
| g(x) ineq constraints | [x] | [x] |
| lbg/ubg bounds on g | [x] | [x] |
| lbx/ubx bounds on x | [ ] | [x] |
| sparsity |    | [x]   |
|---------|-----|--------|"
| single shoot | [x] | [ ] |
| mult shoot | [ ] | [x] |


## Getting started
This repo is structured as follows
```bash
.
+-- benchmarking.ipynb            compares JAX and CasADi for a baseline problem
+-- casadi_tools
¦   +-- human.py                  basic dynamic model with objective, constraints
¦   +-- sys.py                    generic system tools
¦   +-- sysvars.py                generic optimization variables tools
+-- jax_tools
¦   +-- ...                       same as ../casadi_tools/*, but for JAX
¦   +-- ipopt.py                  wrapper for cyipopt to roughly match CasADi syntax
+-- mpc_example/                  example of using MPC in ROS, entry point is main.py
+-- mpc_intro
    +-- MPC_intro.ipynb           exercises for MPC, needs robot.py and Pinocchio
    +-- MPC_intro_pt_2.ipynb      open-ended exercise for MPC using pure CasADi
    +-- config/                   config files needed for robot
    +-- decision_vars.py          helper functions needed for robot
    +-- robot.py                  robot kinematics/dynamics via Pinocchio
    +-- solns/                    solutions to the above exercise
```

### Dependencies
For `casadi_tools` you'll need `pip install casadi`.  To run the `MPC_intro`, you'll need Pinocchio >3.0.0, which is installed on the Converging PC (username: `ipk410`, pwd: `ipk`).

For `jax_tools` you'll need at least `pip install jax flax`. To run the `ipopt` interface you'll also need `cyipopt`. 

### Running a module
These are set up with relative imports so they work best when executed as modules, e.g. `python3 -m jax_tools.mds_sys`.

## Introduction
This section reviews numerical optimization and the two libraries we currently support. 

### Why numerical optimization?
Numerical optimization allows us to specify an engineering problem with a numerical objective (reduce time, reduce energy use), constraints (force should not exceed 20 N), and bounds on the variables we optimize.

Optimization can scale well; parameter fitting for Machine Learning is (typically unconstrained) optimization.  Optimization can also be done online --- Boston Dynamics uses onlien optimization-based control (MPC) for their lower-level control. Model-based reinforcement learning uses a form of MPC to exploit learned models for control.

What do we have to pay for this?  We need to specify our problem so it can be efficiently optimized. Above all, this means we need gradients and often Hessians of our objective and constraints.  These can (and should!) be found automatically with a form of automatic differentiation, to avoid the pain of doing it by hand. Autodiff is a key enabler for optimization, but imposes certain constraints on how we write our problem. This repo aims to make the expression and solving of optimization problems for robotics easier.

Interested in MPC? 
 - The [Rawlings book](https://sites.engineering.ucsb.edu/~jbraw/mpc/) is a good review from the controls side. Chapter 8 (Numerical Methods) covers many challenges important in robotics (choice of solver, stiff integration, etc).
 - [Pieter Abbeel's lecture notes](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/), lecture 8 and 9, provide a review from the robotics perspective.


### CasADi
[CasADi](https://web.casadi.org/) is a library for autodiff and interfacing with numerical solvers. They provide useful tools for efficient MPC (compile to C, track sparsity in our problem), but requires one to work with symbolic variables and build CasADi functions, which can be limiting.

The pip install for CasADis is fine, but if you need faster solutions it's worth installing the HSL linear solvers. A good guide is [here](https://github.com/ami-iit/ami-commons/blob/master/doc/casadi-ipopt-hsl.md), it requires a bit of CMake and linker magic, they are already installed on the Converging PC in the lab.

We can also get robot kinematics and dynamics in CasADi by loading a URDF file via [Pinocchio](https://github.com/stack-of-tasks/pinocchio). However, Pinocchio >3.0.0 is required, which is not currently released on pip, etc. Building Pinocchio from source requires a fair bit of CMake magic as of 11.2023, but is already installed on the Converging PC in the lab.

### JAX
[JAX](https://github.com/google/jax) is a autodiff and parallelization library oriented towards machine learning. Compared with CasADi, it is native Python, can autodiff through your custom data structures, and doesn't require symbolic variables or special functions to be built. On the downsides, it doesn't track sparsity, and CPU-based optimization libraries are not prioritized (although available, e.g. [CyIPOPT](https://github.com/mechmotum/cyipopt)). 


