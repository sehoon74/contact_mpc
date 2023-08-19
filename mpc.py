from robot import Robot
import casadi as ca
from decision_vars import *
from helper_fns import mult_shoot_rollout

class MPC:
    def __init__(self, robots, mpc_params, ipopt_params, icem_params = {}):
        self.robots = robots
        self.mpc_params = mpc_params  # mpc parameters
        self.icem_params = icem_params
        self.nlpopts = ipopt_params
        
        self.H  = mpc_params['H']  # number of mpc steps
        self.dt = mpc_params['dt']  # sampling time
        self.dim_samples = (mpc_params['N_p'], self.H)  # (nq,H)
        
    def solve(self, params):
        if not hasattr(self, "solver"):
            self.build_solver(params)

        self.__args['p'] = self.pars.update(params)  # update parameters for the solver

        if self.icem_params:   # warm start nlp with iCEM
            best_traj, best_input = self.icem_warmstart(params_icem)
            # TODO move to icem_warmstart
            self.__vars.set_x0('q_free', best_traj)
            self.__vars.set_x0('q_contact', best_traj)
            self.__vars.set_x0('imp_rest', best_input)
            self.__args['x0'] = self.vars.get_x0()

        sol = self.solver(**self.args)

        if not self.icem_params: self.__args['x0'] = sol['x']
        self.__args['lam_x0'] = sol['lam_x']
        self.__args['lam_g0'] = sol['lam_g']

        res = self.__vars.dictize(sol['x'])
        return res['imp_rest']

    def build_solver(self, params0):
        nq = self.nq
        self.__pars = ParamSet(params0)

        J = 0
        g = []
        self.__vars = self.robots['free'].get_inputs(self.H)
        imp_stiff = self.pars['imp_stiff']
        for name, rob in self.robots:
            traj, obj, cont_const = mult_shoot_rollout(rob,
                                                       self.H,
                                                       self.__params['init_state'],
                                                       **self.__vars)
            self.__vars += traj
            J += self.pars['belief_'+name]*obj
            g += cont_const

        self.g = ca.vertcat(*g)
        self.lbg = ca.DM.zeros(g.shape[0])
        self.ubg = ca.DM.zeros(g.shape[0])
        
        self.add_max_force_constraint(self.robots[mode].force_sym(self.vars['q_' + mode][:self.nq, 0]), self.vars['q_' + mode][:self.nq, 0])

        x, lbx, ubx, x0 = self.__vars.get_vectors(['sym', 'lb', 'ub', 'init'])
        prob = dict(f=J, x=x, g=ca.vertcat(*self.g), p=self.pars.vectorize())
        self.__args = dict(x0=x0, lbx=lbx, ubx=ubx, lbg=self.lbg, ubg=self.ubg)
        self.solver = ca.nlpsol('solver', 'ipopt', prob, self.options)

    def add_max_force_constraint(self, tau_ext, q):
        H = self.H
        p_inv_jac = self.pinv_jac(q)
        #print(p_inv_jac.shape)
        F_ext = p_inv_jac @ tau_ext

        #print(ca.norm_2(F_ext))
        self.g += [ca.reshape(F_ext[2], 1, 1)]
        self.lbg += [-30] * 1
        self.ubg += [np.inf] * 1
