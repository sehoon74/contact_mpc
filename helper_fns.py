import casadi as c
import numpy as np
import ruamel.yaml as yaml

from contact import Contact
from robot import *
from impedance_controller import ImpedanceController
from mpc import MPC

import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, WrenchStamped, TransformStamped, PointStamped
from tf2_msgs.msg import TFMessage


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
            tau.append(-msg.effort[ind])
        q = np.array(q)
        v = np.array(v)
        tau = np.array(tau)
    except:
        print("Error reading franka joint_state")
    return q, v, tau

def map_wrench(msg, prev_msgs):
    if len(prev_msgs) == 0:
        prev_msgs['force'] = []
    prev_msgs['force'].append(np.hstack((msg.wrench.force.x,
                                msg.wrench.force.y,
                                msg.wrench.force.z)))
    return prev_msgs

def map_joint_state(msg, prev_msgs):
    if len(prev_msgs) == 0:
        for el in ('q', 'dq', 'tau'):
            prev_msgs[el] = []
    q,v,t = map_franka_joint_state(msg)
    prev_msgs['q'].append(q)
    prev_msgs['dq'].append(v)
    prev_msgs['tau'].append(t)
    return prev_msgs

def build_pose_msg(position = None, frame_id = 'panda_link0'):
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = rospy.Time.now()
    if position is not None:
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
    return msg

def build_point_msg(position = None, frame_id = 'panda_link0'):
    msg = PointStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = rospy.Time.now()
    if position is not None:
        msg.point.x = position[0]
        msg.point.y = position[1]
        msg.point.z = position[2]
    return msg

def build_tf_msg(position, child_frame_id, frame_id = 'panda_link0'):
    msg = TransformStamped()
    msg.header.frame_id = frame_id
    msg.child_frame_id = child_frame_id
    msg.header.stamp = rospy.Time.now()
    if position is not None:
        msg.transform.translation.x = position[0]
        msg.transform.translation.y = position[1]
        msg.transform.translation.z = position[2]
        msg.transform.rotation.w = 1.0
    return TFMessage([msg])

def build_wrench_msg(F = None, frame_id = 'panda_link0'):
    msg = WrenchStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = rospy.Time.now()
    if F is not None:
        msg.wrench.force.x = F[0]
        msg.wrench.force.y = F[1]
        msg.wrench.force.z = F[2]
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
    for model in contact_params.get('models', []):
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
                                visc_fric = robot_params['visc_fric'],
                                attrs = attrs,
                                name = mode+"/",
                                ctrl = imp,
                                subsys = [contact_models[model] for model in contact_params['modes'][mode]])
    return modes_filter, modes_mpc, contact_models

def spawn_switched_models(robot_path, attr_path, contact_path = None, mode = None, sym_vars = []):
    assert mode, "Please provide the name of the mode to use"
    robot_params = yaml_load(robot_path)
    attrs = yaml_load(attr_path)
    contact_params = yaml_load(contact_path) if contact_path else {}
    contact_models = {}
    for model in contact_params.get('models', []):
        contact_models[model] = Contact(name = model+'/',
                                        pars = contact_params['models'][model],
                                        attrs = attrs,
                                        sym_vars = sym_vars)
        
    imp = ImpedanceController(input_vars = ['imp_rest'], attrs = attrs)

    robot = SwitchedRobot(robot_params['urdf_path'],
                          visc_fric = robot_params['visc_fric'],
                          attrs = attrs,
                          name = "free/",
                          ctrl = imp,
                          subsys = [contact_models[model] for model in contact_params['modes'][mode]])
    return {'free':robot}, contact_models

def spawn_mpc(print_level = 0, switched = False):
    mpc_params = yaml_load('config/mpc_params_test.yaml')
    ipopt_options = yaml_load('config/ipopt_options.yaml')
    ipopt_options['ipopt.print_level'] = print_level

    if not switched:
        _, robots, contacts = spawn_models(robot_path = "config/franka.yaml",
                                           attr_path  = "config/attrs.yaml",
                                           contact_path = "config/contact_test.yaml",
                                           sym_vars = [])
    else:
        robots, contacts = spawn_switched_models(robot_path = "config/franka.yaml",
                                                 attr_path  = "config/attrs.yaml",
                                                 contact_path = "config/contact_test.yaml",
                                                 mode='point',
                                                 sym_vars = [])
    q0 = 1*np.ones(7)
    dq0 = 1.5*np.ones(7)
    params = {'q': q0,
              'dq': dq0,
              'belief_free':0.0,
              'belief_point':1.0,
              'imp_stiff':400*ca.DM.ones(3),}
    mpc = MPC(robots, params = params, mpc_params=mpc_params, ipopt_options=ipopt_options)
    return mpc, params
