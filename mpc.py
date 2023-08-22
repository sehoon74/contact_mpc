from robot import Robot
import casadi as ca
from decision_vars import *
from helper_fns import mult_shoot_rollout as rollout

class MPC:
    def __init__(self, robots, mpc_params, nlpopts = {}, icem_params = {}):
        self.robots = robots

        self.nlpopts = nlpopts
        self.icem_params = icem_params

        for r in robots.values():
            r.build_step(mpc_params['dt'])
            
        self.H  = mpc_params['H']   # number of mpc steps
    
    def solve(self, params):
        r = self.robots['free']
        params['M_inv'] = r.inv_mass_fn(params['q'])

        if not hasattr(self, "solver"): self.build_solver(params)
            
        self.__args['p'] = self.__pars.vectorize(d = params)

        if self.icem_params:   # warm start nlp with iCEM
            best_traj, best_input = self.icem_warmstart(params_icem)
            # TODO move to icem_warmstart
            self.__vars.set_x0('q_free', best_traj)
            self.__vars.set_x0('q_contact', best_traj)
            self.__vars.set_x0('imp_rest', best_input)
            self.__args['x0'] = self.vars.get_x0()

        sol = self.solver(**self.__args)

        if not self.icem_params: self.__args['x0'] = sol['x']
        self.__args['lam_x0'] = sol['lam_x']
        self.__args['lam_g0'] = sol['lam_g']

        
        return self.__vars.dictize(sol['x'].full())

    def traj_cost(self, robot, traj):
        J = 0
        for h in range(self.H):
            d = robot.get_ext_state(traj[:,h])
            J += ca.sumsqr(d['p']-ca.DM([0.3,0.5,0.8])) + 0.05*ca.sumsqr(d['dx'])
        return J
    
    def build_solver(self, params0):
        self.__pars = ParamSet(params0)
        # TODO: adjust so we're pulling state/input/params from dyn sys
        J = 0
        g = []
        self.__vars = DecisionVarSet(attr_names = ['lb', 'ub'])
        self.__vars += self.robots['free'].get_input(self.H)

        step_inputs = self.__vars.get_vars()
        step_inputs['imp_stiff'] = self.__pars['imp_stiff']
        step_inputs['M_inv'] = self.__pars['M_inv']
        xi0 = dict(q = params0['q'], dq = params0['dq'])
        for name, rob in self.robots.items():
            traj, cost, cont_const = rollout(rob,
                                              self.H,
                                              xi0,
                                              **step_inputs)
            self.__vars += traj
            J += cost
            g += cont_const

        self.g = ca.vertcat(*g)
        self.lbg = ca.DM.zeros(self.g.shape[0])
        self.ubg = ca.DM.zeros(self.g.shape[0])
        
        #self.add_max_force_constraint(self.robots[mode].force_sym(self.vars['q_' + mode][:self.nq, 0]), self.vars['q_' + mode][:self.nq, 0])

        x, lbx, ubx, x0 = self.__vars.get_vectors('sym', 'lb', 'ub', 'init')
        prob = dict(f=J, x=x, g=self.g, p=self.__pars.vectorize())
        self.__args = dict(x0=x0, lbx=lbx, ubx=ubx, lbg=self.lbg, ubg=self.ubg)
        self.solver = ca.nlpsol('solver', 'ipopt', prob, self.nlpopts)

    def add_max_force_constraint(self, tau_ext, q):
        H = self.H
        p_inv_jac = self.pinv_jac(q)
        F_ext = p_inv_jac @ tau_ext

        self.g += [ca.reshape(F_ext[2], 1, 1)]
        self.lbg += [-30] * 1
        self.ubg += [np.inf] * 1
    
    def icem_static(self, rob, H, step_inputs, icem_params):
        # (mu, std, noise, x0) -> (mu, std, best_ctrl, best_inp)
        num_samp = icem_params['num_samples']
        mu = sym('mu', H, rob.nu)
        std = sym('std', H, rob.nu)
        noise = sym('noise', num_samp, H, rob.nu)

        x0 = sym('x0', num_samp, rob.nx)
        
        for i in range(num_samp):
            inp = mu + ca.times(std, noise[i, :, :])
            cost = singleshoot_rollout(rob, H, x0, inp, step_inputs)
