import casadi as ca
from sys import float_info
from robot import Robot
from scipy.stats import multivariate_normal

sym = ca.SX.sym
# ~3x speedup from JIT

jit_opts = {}#{'jit':True, 'compiler':'shell', "jit_options":{"compiler":"gcc", "flags": ["-Ofast"]}}

def build_step_fn(robot):
    # Build a static KF update w arguments of mu, sigma, tau and measured q
    mu, proc_noise, meas_noise= robot.get_ekf_info()

    M_inv = sym('M_inv', robot.nq, robot.nq)

    A, C, mu_next, y = robot.linearized(mu, ca.DM.zeros(robot.nq), M_inv)  # get linearized state and observation matrices wrt states

    tau_meas = sym('tau_meas', robot.nq)
    q_meas = sym('q_meas', robot.nq)
    y_meas = ca.vertcat(q_meas, tau_meas)

    cov = sym('cov', mu.shape[0], mu.shape[0])
    cov_next = A@cov@(A.T) + proc_noise

    L = cov_next@C.T@ca.inv(C@cov_next@(C.T) + meas_noise)  # calculate Kalman gain
    S_hat = C@cov_next@(C.T) + meas_noise
    [Q, R] = ca.qr(S_hat)  # QR decomposition for evaluating determinant efficiently
    det_S_t = ca.trace(R)  # determinant of original predicted measurement covariance is just the product of diagonal elements of R --> block triangular
    log_likelihood = -0.5 * (ca.log(det_S_t) + (y_meas - y).T@ca.inv(S_hat)@(y_meas - y) + robot.nq*ca.log(2*ca.pi))
    likelihood = det_S_t**(-1/2)*ca.exp(-0.5*ca.transpose(y_meas-y) @ ca.inv(S_hat) @ (y_meas-y))

    mu_next_corr = mu_next + L@(y_meas - y)
    cov_next_corr = (ca.SX.eye(robot.nx)-L@C)@cov_next # corrected covariance
    
    fn_dict = {'mu':mu, 'cov':cov, 'q_meas':q_meas, 'tau_meas':tau_meas, 'M_inv':M_inv,
               'mu_next':mu_next_corr, 'cov_next':cov_next_corr, 'likelihood': likelihood,}
    return ca.Function('ekf_step', fn_dict,
                       ['mu', 'cov', 'q_meas', 'tau_meas', 'M_inv'], # inputs to casadi function
                       ['mu_next', 'cov_next', 'likelihood'], jit_opts)

class EKF():
    """ This defines an EKF observer """
    def __init__(self, robot, step_size):
        robot.build_step(step_size)
        xi_init, cov_init = robot.get_ekf_init()
        self.x = {'mu':xi_init, 'cov':ca.diag(cov_init)}
        self.step_fn = build_step_fn(robot)
        self.robot = robot
        self.nq = robot.nq

    def step(self, q_meas, tau_meas, M_inv = None):
        self.x['q_meas'] = q_meas
        self.x['tau_meas'] = tau_meas
        if M_inv is not None:
            self.x['M_inv'] = M_inv
        else:
            self.x['M_inv'] = self.robot.inv_mass_fn(q_meas)
        res = self.step_fn.call(self.x)
        self.x['mu'] = res['mu_next']
        self.x['cov'] = res['cov_next']
        self.likelihood = res['likelihood'] if res['likelihood'] != 0 else float_info.epsilon
        #print(f"exp: {res['y'][7:]}, \nmeas: {tau_meas}")

    def get_statedict(self):
        return self.robot.get_statedict(self.x['mu'])

    def get_ext_state(self):
        d = self.robot._state.dictize(self.x['mu'])
        return self.robot.get_ext_state(d)

class EKF_bank():
    def __init__(self, robots, step_size):
        self.ekfs = {k:EKF(v, step_size) for k,v in robots.items()}
        self.x = {}
        self.x['belief'] = {k:1.0/len(robots) for k in robots}

    def step(self, q_meas, tau_meas, M_inv = None):
        for robot, ekf in self.ekfs.items():
            ekf.step(q_meas, tau_meas, M_inv)
            self.x['belief'][robot] = ekf.likelihood#ca.exp(ekf.likelihood)
            
        likelihood_sum = sum(ekf.likelihood for ekf in self.ekfs.values())#sum(ca.exp(ekf.likelihood) for ekf in self.ekfs.values())
        for robot, ekf in self.ekfs.items():
            self.x['belief'][robot] *= 1/likelihood_sum

    def get_statedict(self):
        # Return the estimated state as dict, plus the belief
        d = {'belief_'+robot:self.x['belief'][robot] for robot in self.ekfs.keys()}
        for ekf in self.ekfs.values():
            d.update(ekf.get_statedict())
        return d

    def get_ext_state(self):
        d = {'belief':self.x['belief']}
        for rob in self.ekfs.keys():
            if rob != 'free': d.update(self.ekfs[rob].get_ext_state())
        d.update(self.ekfs['free'].get_ext_state())
        return d

class Particle: #TODO: inherit EKF directly?
    def __init__(self, belief, mu, cov, weight):
        self.belief = belief
        self.mu = mu
        self.cov = cov
        self.weight = weight

class HybridParticleFilter():
    def __init__(self, robots, step_size):
        assert 'free' in robots, "Need at least a free-space model with key: free"
        self.num_particles = 80
        self.trans_matrix = np.array([[0.8, 0.2], [0.2, 0.8]])

        for robot in robots.values():
            robot.build_step(step_size)
        self.step_fn = {k:build_step_fn(v) for k,v in robots.items()}
        
        xi_init, cov_init = robots['free'].get_ekf_init()
        
        self.x = {'mu':xi_init, 'cov':ca.diag(cov_init)}
        self.x['belief'] = {k:1.0/len(robots) for k in robots}

        self.particles = [Particle(self.x['belief'], self.x['mu'], self.x['cov'], 1/self.num_particles) for _ in range(self.num_particles)]
        
    def step(self, q_meas, tau_meas, M_inv = None):
        self.propogate(q_meas, tau_meas, M_inv)
        self.calc_weights()
        if self.N_eff < self.num_particles/5.0:
            self.stratified_resampling()
        self.estimate_state()

    def propogate(self, q_meas, tau_meas, M_inv):
        step_args = dict(q_meas=q_meas, tau_meas=tau_mes, M_inv=M_inv)
        for particle in enumerate(self.particles):
            particle.belief = np.matmul(particle.bel, self.trans_matrix)
            particle.mode = np.random.choice(self.x['belief'].keys(), p=particle.belief)
            step_args['mu'] = particle.mu
            step_args['cov'] = particle.cov
            res = self.step_fn[sampled_mode].call(step_args)
            particle.mu = res['mu']
            particle.cov = res['cov']
            particle.weight = res['likelihood'] if res['likelihood'] != 0 else float_info.epsilon

    def calc_weights(self):
        summation = sum(p.weight for p in self.particles)
        weightsum = sum(np.exp(p.weight-np.log(summation))**2 for p in self.particles)        
        self.N_eff = 1/weightsum
        for mode in self.x['belief']:
            self.x['belief'][mode] = sum(np.exp(p.weight) for p in self.particles)
        for particle in particles:
            particle.belief = self.x['belief'].values()
        
    def estimate_state(self):
        return [(p.mode, p.mu) for p in self.particles]

    def get_belief(self):
        return self.x['belief']
        
    
class MomentumObserver():
    ''' Mostly following https://elib.dlr.de/129060/1/root.pdf '''
    
    def __init__(self, par, q0):
        self.K = par['mom_obs_K']
        self.h = par['h']
        self.r = np.zeros(q0.size)
        
    def step(self, p, tau_err, F = None):      
        self.r += self.K*(p-self.h*(self.r-tau_err))
