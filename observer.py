import casadi as ca
import numpy as np
from robot import Robot
from scipy.stats import multivariate_normal
import copy

def build_step_fn(robot):
    # Build a static KF update w arguments of mu, sigma, tau and measured q
    mu, proc_noise, meas_noise= robot.get_ekf_info()
    
    M_inv = ca.SX.sym('M_inv', robot.nq, robot.nq)
    
    A, C, mu_next, y = robot.linearized(mu, ca.DM.zeros(robot.nq), M_inv)  # get linearized state and observation matrices wrt states
    tau_meas = ca.SX.sym('tau_meas', robot.nq)
    q_meas = ca.SX.sym('q_meas', robot.nq)
    y_meas = ca.vertcat(q_meas, tau_meas)

    cov = ca.SX.sym('cov', mu.shape[0], mu.shape[0])
    cov_next = A@cov@(A.T) + proc_noise
    
    L = cov_next@C.T@ca.inv(C@cov_next@(C.T) + meas_noise)  # calculate Kalman gain
    S_hat = C@cov_next@(C.T) + meas_noise
    [Q, R] = ca.qr(S_hat)  # QR decomposition for evaluating determinant efficiently
    det_S_t = ca.trace(R)  # determinant of original predicted measurement covariance is just the product of diagonal elements of R --> block triangular
    log_likelihood = -0.5 * (np.log(det_S_t) + (y_meas - y).T @ ca.inv(S_hat) @ (y_meas - y) + robot.nq * np.log(2 * np.pi))
    temp1 = det_S_t**(-1/2)
    temp2 = ca.exp(-0.5*(y_meas-y).T @ ca.inv(S_hat) @ (y_meas-y))
    likelihood = (2*np.pi)**(-robot.nq/2)*temp1*temp2
    
    mu_next_corr = mu_next + L@(y_meas - y)
    cov_next_corr = (ca.SX.eye(robot.nx)-L@C)@cov_next # corrected covariance
    
    fn_dict = {'mu':mu, 'cov':cov, 'q_meas':q_meas, 'tau_meas':tau_meas, 'M_inv':M_inv,
               'mu_next':mu_next_corr, 'cov_next':cov_next_corr, 'S_hat': S_hat,
               'likelihood': log_likelihood, 'cov_next_pre': cov_next}
    step_fn = ca.Function('ekf_step', fn_dict,
                          ['mu', 'cov', 'q_meas', 'tau_meas', 'tau_input', 'M_inv'], # inputs to casadi function
                          ['mu_next', 'cov_next', 'S_hat', 'likelihood', 'cov_next_pre'])   # outputs of casadi function
    return step_fn

class EKF():
    """ This defines an EKF observer """
    def __init__(self, robot):
        xi_init, cov_init = robot.get_init()
        self.x = {'mu':xi_init, 'cov':ca.diag(cov_init)} 
        self.step_fn = build_step_fn(robot)
        self.dyn_sys = robot
        self.nq = robot.nq

    def step(self, q, tau, M_inv = None):
        self.x['q_meas'] = q
        self.x['tau_meas'] = tau
        if M_inv is not None:
            self.x['M_inv'] = M_inv
        else:
            self.x['M_inv'] = self.dyn_sys.inv_mass_fn(q)
        res = self.step_fn.call(self.x)
        
        self.x['mu'] = res['mu_next']
        self.x['cov'] = res['cov_next']

    def get_statedict(self):
        return self.dyn_sys.get_statedict(self.x['mu'])
    
    def likelihood(self, obs):
        return NotImplemented

class MomentumObserver():
    ''' Mostly following https://elib.dlr.de/129060/1/root.pdf '''
    
    def __init__(self, par, q0):
        self.K = par['mom_obs_K']
        self.h = par['h']
        self.r = np.zeros(q0.size)
        
    def step(self, p, tau_err, F = None):      
        self.r += self.K*(p-self.h*(self.r-tau_err))
