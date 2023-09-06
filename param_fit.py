import numpy as np
import casadi as ca
import time
import rosbag

from helper_fns import *

def loss_fn(states, torques, robot, contact, num_pts = 1000):
    ''' States is a trajectory as list.
        Param are the parameters to optimize
        disc_dyn is a function for next state as fn of state, input, param '''
    loss = 0
    num_obs = states['q'].shape[1]
    if num_pts > num_obs:
        print(f"Traj only has {num_obs} points, less than requested pts. Still doing it")
        num_pts = num_pts
    skip_size = int(num_obs/num_pts)

    p = contact._state.get_vars()
    for i in range(0, num_obs, skip_size):
        p['q'] = states['q'][:,i]
        p['dq'] = states['dq'][:,i]
        tau_pred = robot.tau_ext_fn.call(p)['tau_ext']
        loss += 1.0*ca.norm_2(torques[:,i] - tau_pred)
        statedict = robot.get_ext_state(p)
        loss += 100*ca.norm_2(statedict['contact_1/disp'])

    #del param['xi']
    for k,v in p.items():
        if 'stiff' in k:
            loss += 1e-12*ca.sqrt(ca.norm_1(v))
        elif 'pos' in k:
            loss += 100000.*v.T@v #+ 100*v[0]*v[0]
    return loss

def optimize(states, torques, robot, contact):
    loss = loss_fn(states, torques, robot, contact)
    
    x, lbx, ubx, x0 = contact._state.get_vectors('sym', 'lb', 'ub', 'init')
    
    nlp = {'x':x, 'f': loss}

    opts = {'expand':False,
            'ipopt.print_level':5}

    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    print('________________________________________')
    print(' ##### Optimizing offline params ######' )

    solve_time = -time.time()
    res = solver(x0=x0, lbx=lbx, ubx=ubx)
    status = solver.stats()['return_status']
    obj = res['f']
    solve_time += time.time()

    res_dict = contact._state.dictize(res['x'])
    
    print("Solve time:  %s - %f sec" % (status, solve_time))
    print("Final obj:  {}".format(obj))
    return res_dict
    
def param_fit(bag, contact_to_fit, pars_to_fit = [], contact_yaml = 'contact.yaml'):
    msgs = bag_loader(bag, map_joint_state, topic_name = '/joint_states')
    states = {'q':msgs['q'], 'dq':msgs['dq']}
    tau_ms = msgs['tau']
    time_step = np.mean(np.diff(msgs['t']))
    print(f"Loaded {len(msgs['t'])} msgs from {bag} with time step {time_step} sec")
    
    robots_obs, _, contacts = spawn_models(robot_path = "config/franka.yaml",
                                    attr_path  = "config/attrs.yaml", 
                                    contact_path = "config/"+contact_yaml,
                                    sym_vars = pars_to_fit)

    contact_params = yaml_load("config/"+contact_yaml)
    for mode in contact_params['modes']:
        if contact_to_fit in contact_params['modes'][mode]:
            robot_to_fit = mode
            print(f"Using robot {mode} which has contacts {contact_params['modes'][mode]}")

    robots_obs[robot_to_fit].build_step(time_step)

    optimized_par = optimize(states, tau_ms,
                             robots_obs[robot_to_fit],
                             contacts[contact_to_fit])

    for k,v in optimized_par.items():
        print(f'{k}:{v}')

def bag_loader(path, map_and_append_msg, topic_name = 'joint_state'):
    bag = rosbag.Bag(path)
    num_obs = bag.get_message_count(topic_name)
    if num_obs == 0:
        topic_name = '/'+topic_name
        num_obs = bag.get_message_count(topic_name)
    print('Loading ros bag {}  with {} msgs on topic {}'.format(path, num_obs, topic_name))

    msgs = {}
    t = []
    for _, msg, t_ros in bag.read_messages(topics=[topic_name]):
        t.append(t_ros.to_sec())
        map_and_append_msg(msg, msgs)
    t = [tt-t[0] for tt in t]
    msgs_in_order = {}
    for key in msgs.keys():
        t_in_order, el_in_order = zip(*sorted(zip(t,msgs[key])))
        msgs_in_order[key] = np.array(el_in_order).T
    msgs_in_order['t'] = np.array(t_in_order)
    
    return msgs_in_order
