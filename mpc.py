from robot import Robot
import casadi as ca
from decision_vars import *

class MPC:
    def __init__(self, robots, mpc_params, ipopt_params, icem_params = {}):
        self.robots = robots
        self.mpc_params = mpc_params  # mpc parameters
        self.icem_params = icem_params
        self.nlpopts = ipopt_params

        self.H  = mpc_params['H']  # number of mpc steps
        self.dt = mpc_params['dt']  # sampling time
        self.dim_samples = (mpc_params['N_p'], self.H)  # (nq,H)
        
    def solve(self, params_mpc, params_icem):
        if not hasattr(self, "solver"):
            self.build_solver(params_mpc)

        self.__args['p'] = self.pars.update(params_mpc)  # update parameters for the solver

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
        N_p = self.N_p

        self.__pars = ParamSet(params0)

        J = 0       # objective function
            
        self.__vars = self.robots.values()[0].get_mpc_vars(self.H, ['imp_rest'])
        for rob in self.robots:
            self.__vars += robots.get_mpc_vars(self.H, ['q', 'dq']) 

        imp_stiff = self.pars['imp_stiff']

        self.build_constraints()
        for name, robot in self.robots.items():
            dyn_next = robot.step(xi=self.__vars[name+'/'+],
                                  imp_rest=self.vars['imp_rest'],
                                  imp_stiff=imp_stiff,
                                  des_pose=self.pars['des_pose'])
            self.add_continuity_constraints(dyn_next['xi_next'], self.vars['q_' + mode])
            #self.add_max_force_constraint(self.robots[mode].force_sym(dyn_next['xi_next'][:self.nq, -1]), dyn_next['xi_next'][:self.nq, -1])
            self.add_max_force_constraint(self.robots[mode].force_sym(self.vars['q_' + mode][:self.nq, 0]), self.vars['q_' + mode][:self.nq, 0])
            
            J += self.pars['belief_' + mode] * ca.sum2(dyn_next['cost'])

        x, lbx, ubx, x0 = self.__vars.get_vectors(['sym', 'lb', 'ub', 'init'])
        
        self.__args = dict(x0=x0, lbx=lbx, ubx=ubx, lbg=self.lbg, ubg=self.ubg)

        prob = dict(f=J, x=x, g=ca.vertcat(*self.g), p=self.pars.vectorize())
    
        self.solver = ca.nlpsol('solver', 'ipopt', prob, self.options)

    def build_constraints(self):
        # General NLP constraints, not including continuity constraints
        self.g = []  # constraints functions
        self.lbg = []  # lower bound on constraints
        self.ubg = []  # upper-bound on constraints
        #self.g += [self.vars.get_deviation('imp_stiff')]
        #self.lbg += [-self.mpc_params['delta_K_max']] * self.N_p
        #self.ubg += [self.mpc_params['delta_K_max']] * self.N_p

    def add_max_force_constraint(self, tau_ext, q):
        H = self.H
        p_inv_jac = self.pinv_jac(q)
        #print(p_inv_jac.shape)
        F_ext = p_inv_jac @ tau_ext



        #print(ca.norm_2(F_ext))
        self.g += [ca.reshape(F_ext[2], 1, 1)]
        self.lbg += [-30] * 1
        self.ubg += [np.inf] * 1



    def add_continuity_constraints(self, x_next, x):
        nx = self.nx
        H = self.H

        self.g += [ca.reshape(self.pars['init_state'] - x[:, 0], nx, 1)]
        self.g += [ca.reshape(x_next[:, :-1] - x[:, 1:], nx * (H-1), 1)]
        self.lbg += [self.mpc_params['constraint_slack']] * nx * H
        self.ubg += [-self.mpc_params['constraint_slack']] * nx * H

    def build_dec_var_constraints(self):
        ub = {}
        lb = {}
        lb['imp_rest'] = -self.mpc_params['delta_xd_max']
        ub['imp_rest'] = self.mpc_params['delta_xd_max']
        lb['imp_stiff'] = self.mpc_params['K_min']
        ub['imp_stiff'] = self.mpc_params['K_max']

        return ub, lb

    def cartesian_force(self, tau_ext, q):
        H = self.H
        p_inv_jac = self.pinv_jac(q)
        F_ext = p_inv_jac @ tau_ext
        return F_ext

    def get_tcp(self, q, dq):
        x_tcp = self.robots['free'].get_tcp_motion(q=q, dq=dq)[0]
        x_tcp_pos = x_tcp[0]
        return x_tcp_pos
