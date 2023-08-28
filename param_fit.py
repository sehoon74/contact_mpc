
def generate_traj(bag, est_pars = {}):
    print('Generating trajectory from {}'.format(bag))
    
    msgs = bag_loader(bag, map_joint_state, topic_name = '/joint_states')
    force_unaligned = bag_loader(bag, map_wrench, topic_name = '/franka_state_controller/F_ext')
    force = get_aligned_msgs(msgs, force_unaligned)

    #robot = Robot(p, est_pars = est_pars)

    robots = RobotDict("config_files/franka.yaml", ["config_files/contact.yaml", "config_files/free_space.yaml"], est_pars).param_dict
    observer = ekf(robots['contact'])
    #observer = ekf(robots['free-space'])
    #observer = HybridParticleFilter(robots)
    num_msgs = len(msgs['pos'].T)

    sd_initial = observer.get_statedict()
    results = {k:np.zeros((v.shape[0], num_msgs)) for k,v in sd_initial.items()}
    results['q_m'] = msgs['pos']
    results['dq_m'] = msgs['vel']
    results['f_ee'] = force['force']
    results['tau_m'] = msgs['torque']
    print("Results dict has elements: {}".format(results.keys()))
    
    update_freq = []

    for i in range(num_msgs):
        #if i == 1 or i == 1000 or i == 3000:
        #    print(observer.cov)

        tic = time.perf_counter()
        res = observer.step(q = msgs['pos'][:,i], tau = msgs['torque'][:,i])
        toc = time.perf_counter()
        update_freq.append(1/(toc-tic))
        #print(msgs['torque'][:,i])
        statedict = observer.get_statedict()
        for k,v in statedict.items():
            results[k][:,[i]] = v

    average_freq = (sum(update_freq)/num_msgs)/1000
    print("Average update frequency is {} kHz".format(average_freq))
    fname = bag[:-4]+'.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(results, f)
    print('Finished saving state trajectory of length {}'.format(num_msgs))

def param_fit(bag):
    fname = bag[:-4]+'.pkl'
    if not exists(fname):
        generate_traj(bag)
    print("Loading trajectory for fitting params")
    with open(fname, 'rb') as f:
        results = pickle.load(f)
    #states = results['xi']
    #print(np.mean(results['q_m'],axis=1))
    #print(np.mean(results['dq_m'], axis=1))
    #print(np.std(results['dq_m'], axis=1))

    states = np.vstack((results['q_m'], results['dq_m']))
    tau_ms = results['tau_m']
    print(min(tau_ms[1,:]))
    print(max(tau_ms[1,:]))

    p_to_opt = {}
    p_to_opt['contact_1_pos'] = ca.SX.sym('pos',3)
    p_to_opt['contact_1_stiff'] = ca.SX.sym('stiff',3)
    p_to_opt['contact_1_rest'] = ca.SX.sym('rest',3)

    robots = RobotDict("config_files/franka.yaml", ["config_files/contact.yaml", "config_files/free_space.yaml"], est_pars)
    p = robots.raw_param_dict['contact']

    prediction_skip = 1
    p['h'] *= prediction_skip
    rob = Robot(p, opt_pars = p_to_opt)
    optimized_par = optimize(states.T, tau_ms.T, p_to_opt, rob, prediction_skip)
    for k,v in optimized_par.items():
        print(f'{k}:{v}')
        #rospy.set_param('contact_1_'+k, v.full().tolist())
