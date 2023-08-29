import casadi as c
import numpy as np
import ruamel.yaml as yaml

from contact import Contact
from robot import *
from impedance_controller import ImpedanceController

import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

names_franka = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
                'panda_joint5', 'panda_joint6', 'panda_joint7']

def map_franka_joint_state(msg):
    try:
        q = []
        v = []
        tau = []
        for jt_name in names_franka:
            ind = msg.name.index(jt_name)
            q.append(msg.position[ind])
            v.append(msg.velocity[ind])
            tau.append(msg.effort[ind])
        q = np.array(q)
        v = np.array(v)
        tau = np.array(tau)
    except:
        print("Error reading franka joint_state")
    return q, v, tau

def build_pose_msg(position = None, frame_id = 'panda_link0'):
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = rospy.Time.now()
    if position is not None:
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
    return msg

def build_jt_msg(q, dq = [], tau = [], names = []):
    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.position = q
    msg.velocity = dq
    msg.effort = tau
    msg.name = names
    return msg

def tf_to_state(msg):
    q = np.array([msg.transform.rotation.w,
         msg.transform.rotation.x,
         msg.transform.rotation.y,
         msg.transform.rotation.z])
    r = quat_to_rotvec(q)
    p = np.array([msg.transform.translation.x,
         msg.transform.translation.y,
         msg.transform.translation.z])
    return np.hstack((p.T,np.squeeze(r)))

def quat_to_rotvec(q):
    q *= ca.sign(q[0])  # multiplying all quat elements by negative 1 keeps same rotation, but only q0 > 0 works here
    th_2 = ca.acos(q[0])
    th = th_2*2.0
    return ca.vertcat(q[1]/ca.sin(th_2)*th, q[2]/ca.sin(th_2)*th, q[3]/ca.sin(th_2)*th)

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
                                        attrs = attrs,
                                        sym_vars = sym_vars)
    modes_filter = {}
    for mode in contact_params.get('modes'):
        modes_filter[mode] = LinearizedRobot(robot_params['urdf_path'],
                                             attrs = attrs_state,
                                             subsys = [contact_models[model] for model in contact_params['modes'][mode]])
        
    imp = ImpedanceController(input_vars = ['imp_rest'], attrs = attrs)
    modes_mpc = {}
    for mode in contact_params.get('modes'):
        modes_mpc[mode] = Robot(robot_params['urdf_path'],
                                attrs = attrs,
                                ctrl = imp,
                                subsys = [contact_models[model] for model in contact_params['modes'][mode]])
    return modes_filter, modes_mpc, contact_models

def mult_shoot_rollout(sys, name, H, xi0, **step_inputs):
    name = name + '/'
    state = sys.get_state(H)
    assert set(['q', 'dq']) == state.get_vars().keys(), 'Additional state elements not handled currently with named mode rollouts'
    state.namespace_var('q', name)
    state.namespace_var('dq', name)
    res = sys.step(q=state[name+'q'], dq=state[name+'dq'], **step_inputs)
    continuity_constraints = []
    for st in ['q', 'dq']:
        continuity_constraints += [state[name+st][:, 0] - xi0[st]]
        continuity_constraints += [ca.reshape(res[st][:, :-1] - state[name+st][:, 1:], -1, 1)]
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
