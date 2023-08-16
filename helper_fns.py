from contact import Contact
from robot import Robot
import ruamel.yaml as yaml

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
    attrs_state = {n:attrs[n] for n in ["proc_noise", "cov_init"]}
    contact_params = yaml_load(contact_path) if contact_path else {}
    contact_models = {}
    for model in contact_params.get('models'):
        contact_models[model] = Contact(name = model,
                                        pars = contact_params['models'][model],
                                        attrs = attrs_state,
                                        sym_vars = sym_vars)
    modes = {}
    for mode in contact_params.get('modes'):
        modes[mode] = Robot(robot_params['urdf_path'],
                            attrs = attrs_state,
                            subsys = [contact_models[model] for model in contact_params['modes'][mode]])
    return modes
