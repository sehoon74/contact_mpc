import argparse
import pickle
import time
import casadi as ca
import numpy as np

import rospy
import tf2_ros as tf
import dynamic_reconfigure.client
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped, PoseStamped

from observer import *
from robot import *
from helper_fns import *
from mpc import *

class ContactMPC():
    """ This handles the ros interface, loads models, etc.
        An observer runs on callback for joint_states, updating any estimate on robot state
        The MPC runs in ::control(), using the current state of observer
    """
    def __init__(self, config_path, est_pars = [], sim = False):
        self.mpc_params = yaml_load(config_path+'mpc_params.yaml')
        self.ipopt_options = yaml_load(config_path+'ipopt_options.yaml')

        self.tf_buffer = tf.Buffer()
        self.tf_listener = tf.TransformListener(self.tf_buffer)

        self.joint_sub = rospy.Subscriber('/joint_states', JointState, self.joint_callback, queue_size=1)
        self.joint_pub = rospy.Publisher('joint_states_obs', JointState, queue_size=1)
        self.bel_pub = rospy.Publisher('belief_obs', JointState, queue_size=1)
        self.F_pub = rospy.Publisher('est_force', JointState, queue_size=1)
        self.tcp_pub = rospy.Publisher('tcp_pos', JointState, queue_size=1)
        self.imp_rest_pub = rospy.Publisher('cartesian_impedance_example_controller/equilibrium_pose', PoseStamped, queue_size=1)  # impedance rest point publisher

        robots_obs, robots_mpc, self.contacts = spawn_models(robot_path = config_path+"franka.yaml",
                                                  attr_path  = config_path+"attrs.yaml", 
                                                  contact_path = config_path+"contact.yaml",
                                                  sym_vars = est_pars)
        self.nq = robots_obs['free'].nq

        self.observer = EKF_bank(robots_obs, step_size = 1.0/250.0 )

        self.contact_env_pub = {c:rospy.Publisher(c+'/env', PoseStamped, queue_size=1) for c in self.contacts}
        self.contact_rob_pub = {c:rospy.Publisher(c+'/rob', PoseStamped, queue_size=1) for c in self.contacts}

        # Set up robot state and MPC state
        self.rob_state = {'imp_stiff':None}
        self.rob_state.update(self.observer.get_statedict())
        if sim:
            self.rob_state['imp_stiff'] = np.array([200, 200, 200])
        else:
            self.par_client = dynamic_reconfigure.client.Client( "/cartesian_impedance_example_controller/dynamic_reconfigure_compliance_param_node")

        self.init_orientation = self.tf_buffer.lookup_transform('panda_link0', 'panda_EE', rospy.Time(0),
                                                                rospy.Duration(1)).transform.rotation
        # Set up MPC
        self.mpc = MPC(robots_mpc,
                       mpc_params=self.mpc_params,
                       ipopt_options=self.ipopt_options)

        self.timelist = [] # Performance profiling

    def joint_callback(self, msg):
        """ To be called when the joint_state topic is published with joint position and torques """
        q_m, _, tau_m = map_franka_joint_state(msg)

        if hasattr(self, 'observer'):
            self.observer.step(q_meas = q_m, tau_meas = tau_m)
            self.publish_observer()

    def publish_contacts(self):
        for contact in self.contacts:
            msg_env = build_env_stiff(contact['rest'],contact['stiff'])
            msg_rob = build_rob_contact(contact['pos'], self.tcp)
            self.contact_env_pub.publish(msg_env)
            self.contact_rob_pub.publish(msg_rob)
            
            # TODO publish force estimates

    def publish_observer(self):
        odict = self.observer.get_ext_state()
        
        msg_jt    = build_jt_msg(q=odict['q'], dq=odict['dq'])
        msg_F_est = build_jt_msg(q=odict.get('F_ext', np.zeros(3))) 
        msg_tcp   = build_jt_msg(q=odict['p'], dq=odict['dx'])
        msg_bel   = build_jt_msg(q=odict['belief'].values(),
                                 names=odict['belief'].keys())

        if not rospy.is_shutdown():
            self.joint_pub.publish(msg_jt)
            self.F_pub.publish(msg_F_est)
            self.tcp_pub.publish(msg_tcp)
            self.bel_pub.publish(msg_bel)
            
    def publish_imp_rest(self, imp_rest):
        #des_pose_w = compliance_to_world(self.rob_state['pose'], action_to_execute, only_position=True)
        msg_imp_xd = build_pose_msg(position=imp_rest, frame_id='panda_link0')    # get desired rest pose in world frame
        msg_imp_xd.pose.orientation = self.init_orientation
        if not rospy.is_shutdown():
            self.imp_rest_pub.publish(msg_imp_xd)

    def update_state_async(self):
        #pose_msg = self.tf_buffer.lookup_transform('panda_link0', 'panda_EE', rospy.Time(0), rospy.Duration(0.05))
        #self.rob_state['pose'] = msg_to_state(pose_msg)

        if hasattr(self, 'par_client'):
            imp_pars = self.par_client.get_configuration()   # set impedance stiffness values
            self.rob_state['imp_stiff'] = np.array((imp_pars['translational_stiffness_x'],
                                                    imp_pars['translational_stiffness_y'],
                                                    imp_pars['translational_stiffness_z']))
            
    def control(self):
        if any(el is None for el in self.rob_state.values()) or rospy.is_shutdown(): return
        
        self.rob_state.update(self.observer.get_statedict())
        start = time.time()
        _, mpc_result = self.mpc.solve(self.rob_state)
        self.timelist.append(time.time() - start)

        self.publish_imp_rest(mpc_result['imp_rest'])  # publish impedance optimized rest pose --> to be sent to franka impedance interface

    def shutdown(self):
        print("Shutting down node")
        if len(self.timelist) > 1:
            t_stats = np.array(self.timelist)
            print(f"Cold Start: {t_stats[0]}, Mean: {np.mean(t_stats[1:])}, Min: {min(t_stats[1:])}, Max: {max(t_stats[1:])}")

def start_node(config_path, est_pars, sim):
    rospy.init_node('contact_mpc')
    node = ContactMPC(config_path = config_path, est_pars = est_pars, sim = sim)
    rospy.on_shutdown(node.shutdown)  # Set shutdown to be executed when ROS exits
    rospy.sleep(1e-1)  # Sleep so ROS can init
    while not rospy.is_shutdown():
        node.update_state_async()
        node.control()
        rospy.sleep(1e-8)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", default="", help="Optimize params on this bag")
    parser.add_argument("--opt_par", default=False, action='store_true',
                        help="Optimize the parameters")
    parser.add_argument("--est_par", default="", help="Estimate this param online")
    parser.add_argument("--config_path", default="config/", help="Directory with config files")
    parser.add_argument("--sim", default=False, action='store_true',
                        help="If using a bag file")
    args = parser.parse_args()

    if args.opt_par:
        if args.bag == "": rospy.signal_shutdown("Need bag to optimize params from")
        generate_traj(args.bag, est_pars)
        param_fit(args.bag)
    else:
        start_node(config_path = args.config_path,
                   est_pars = args.est_par,
                   sim = args.sim)
