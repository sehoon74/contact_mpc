import casadi as ca
import ruamel.yaml as yaml

from contact import Contact
from robot import Robot

def yaml_load(path):
    try:
        with open(path, 'r') as f:
            params = yaml.load(f, Loader = yaml.Loader)        
    except:
        print(f"Error loading yaml from path {path}")
    return params

def spawn_models(robot_path, attr_path, contact_path = None, sym_vars = []):
    robot_params = yaml_load(robot_path)
    attrs = yaml_load(attr_path)
    attrs_state = {n:attrs[n] for n in ["proc_noise", "cov_init", "meas_noise"]}
    contact_params = yaml_load(contact_path) if contact_path else {}
    contact_models = {}
    for model in contact_params.get('models'):
        contact_models[model] = Contact(name = model+'/',
                                        pars = contact_params['models'][model],
                                        attrs = attrs_state,
                                        sym_vars = sym_vars)
    modes = {}
    for mode in contact_params.get('modes'):
        modes[mode] = Robot(robot_params['urdf_path'],
                            attrs = attrs_state,
                            subsys = [contact_models[model] for model in contact_params['modes'][mode]])
    return modes

def mult_shoot_rollout(sys, H, xi0, **step_inputs):
    state = sys.get_state(H)
    res = sys.step(**step_inputs, **state.get_vars())
    continuity_constraints = []
    for st in state.get_vars().keys():
        continuity_constraints += [state[st][:, 0] - xi0[st]]
        continuity_constraints += [ca.reshape(res[st][:, :-1] - state[st][:, 1:], -1, 1)]
    return state, ca.sum2(res['cost']), continuity_constraints

def singleshoot_rollout(sys, H, x0, inp_traj, **step_inputs):
    step_inputs['xi'] = x0
    cost = 0
    for h in range(H):
        step_inputs['imp_rest'] = inp_traj[:,h]
        res = sys.step_vec(**step_inputs, **res)
        cost += res.pop['cost']
        step_inputs.update(res)
    return cost
