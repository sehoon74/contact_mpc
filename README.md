# Contact MPC 
This repo contains code to build estimators and MPC problems which contain
robot dynamics and compliant contact models. 

It is associated with [this paper](https://arxiv.org/abs/2303.17476). 

## Getting started

### Dependencies
The known dependencies are CasADi (standard install via pip should be OK) and Pinocchio.

Unfortunately, Pinocchio 3 is required, so you'll need to build from the branch `pinocchio3-preview` in order to get the robot dynamics which are differentiable in CasADi. 

The interfaces with the robot are done in ROS1, and using the standard `franka_ros` should be OK. 

### Using the repo
For each experiment setup, I make a folder in `config` which holds the params of observer/MPC/etc for reproduceability.

The file `contact.yaml` specifies the contact modes and individual contact models (i.e. a single compliant contact). The following high-level keys are needed:
- modes: This specifies what discrete contact modes are in the model, and which contacts are included in each one. The contacts listed are connected in parallel
- models: Each of these is a single contact model with `pos` (position on robot), `stiff` (stiffness in world coord), `rest` (rest position in world coord)
- est: Specifies what parameters hould be estimated for each contact. Should be empty list if nothing to estimate.

Once all the params are set, you can call `main.py` using `--config-path` to specify what folder it should take the params from. This will start an estimator and MPC problem with the corresponding params.

There was previously code which fit the parameters to a rosbag.  This worked via the argument `--opt_par`, specifying a bag with `--bag`. This code is not yet updated, and can be found in `fit_model.py`.



