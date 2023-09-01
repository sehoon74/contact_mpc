# Filippo Rozzi, 2023

from robot import Robot
import casadi as ca
from decision_vars import *
from helper_fns import mult_shoot_rollout as rollout
from colorednoise import powerlaw_psd_gaussian

class MPC:
    def __init__(self, robots, mpc_params, ipopt_options = {}, icem = False):
        assert 'free' in robots, "Need at least the free-space model w/ key: free!"
        self.robots = robots

        self.mpc_params = mpc_params
        self.ipopt_options = ipopt_options
        
        for r in robots.values():
            cost_fn = self.build_cost_fn(r)
            r.build_step(step_size = mpc_params['dt'], cost_fn = cost_fn)

        self.H  = mpc_params['H']   # number of mpc steps
        self.nu = robots['free'].nu

        self.icem = icem
        if icem: self.icem_init(mpc_params)
        
    def solve(self, params):
        r = self.robots['free']
        params['M_inv'] = r.inv_mass_fn(params['q'])

        if not hasattr(self, "solver"): self.build_solver(params)

        self.__args['p'] = self.__pars.vectorize_dict(d = params)

        sol = self.solver(**self.__args)

        self.__args['x0'] = sol['x']
        self.__args['lam_x0'] = sol['lam_x']
        self.__args['lam_g0'] = sol['lam_g']

        res = self.__vars.dictize(sol['x'])
        res.update(params)
        return sol['f'], res

    def build_cost_fn(self, robot):
        st = robot.get_step_args()
        ext_st = robot.get_ext_state(st)
        cost = 0
        cost += ca.sumsqr(ext_st['p'] - self.mpc_params['des_pose'])
        cost += self.mpc_params['vel_cost']*ca.sumsqr(ext_st['dx'])
        cost += self.mpc_params['imp_cost']*ca.sumsqr(ext_st['p']-st['imp_rest'])
        st_cost = ca.Function('st_cost', [*st.values()], [cost], [*st.keys()], ['cost'])
        return st_cost
    
    def build_solver(self, params0):
        self.__pars = ParamDict(params0)

        J = 0
        g = []
        self.__vars = DecisionVarDict(attr_names = ['lb', 'ub'])
        self.__vars += self.robots['free'].get_input(self.H)

        step_inputs = self.__vars.get_vars()
        step_inputs['imp_stiff'] = self.__pars['imp_stiff']
        step_inputs['M_inv'] = self.__pars['M_inv']
        xi0 = dict(q = params0['q'], dq = params0['dq'])
        for name, rob in self.robots.items():
            traj, cost, cont_const = rollout(rob, self.H, xi0, **step_inputs)
            self.__vars += traj
            J += self.__pars['belief_'+name]*cost
            g += cont_const

        self.g = ca.vertcat(*g)
        self.lbg = ca.DM.zeros(self.g.shape[0])
        self.ubg = ca.DM.zeros(self.g.shape[0])
        
        #self.add_max_force_constraint(self.robots[mode].force_sym(self.vars['q_' + mode][:self.nq, 0]), self.vars['q_' + mode][:self.nq, 0])

        x, lbx, ubx, x0 = self.__vars.get_vectors('sym', 'lb', 'ub', 'init')
        prob = dict(f=J, x=x, g=self.g, p=self.__pars.vectorize_attr())
        self.__args = dict(x0=x0, lbx=lbx, ubx=ubx, lbg=self.lbg, ubg=self.ubg)
        self.solver = ca.nlpsol('solver', 'ipopt', prob, self.ipopt_options)

    def add_max_force_constraint(self, tau_ext, q):
        H = self.H
        p_inv_jac = self.pinv_jac(q)
        F_ext = p_inv_jac @ tau_ext

        self.g += [ca.reshape(F_ext[2], 1, 1)]
        self.lbg += [-30] * 1
        self.ubg += [np.inf] * 1

    def icem_init(self):
        self.mu = np.zeros((self.nu, self.H))
        self.std = 1*np.ones((self.nu, self.H))
        for r in self.robots.values():
            r.build_rollout(self.mpc_params['H'], self.mpc_params['num_samples'])

    def icem_warmstart(self, params, num_iter = None):
        if num_iter: self.mpc_params['num_iter'] = num_iter
        #res = self.__vars.dictize(self.__args['x0'])
        #self.mu = res['imp_rest'].full()
        _, res = self.icem_solve(params)
        self.__args['x0'] = self.__vars.vectorize_dict(d = res)
                    
    def icem_solve(self, params):
        """ Solve from initial state xi0 for """
        r = self.robots['free']
        args= dict(xi0 = r.get_statevec({r.name+k:v for k,v in params.items()}),
                   M_inv = r.inv_mass_fn(params['q']),
                   imp_stiff=params['imp_stiff'])
        elite_samples = powerlaw_psd_gaussian(self.mpc_params['beta'],
                                        size=(self.nu, self.mpc_params['num_elites'], self.H))
        ns = self.mpc_params['num_samples']-self.mpc_params['num_elites']
        lb, ub = r._u.get_vectors('lb', 'ub')
        lb = lb.full()
        ub = ub.full()
        for i in range(self.mpc_params['num_iter']):
            samples = powerlaw_psd_gaussian(self.mpc_params['beta'],
                                            size=(self.nu, ns, self.H))
            samples = np.append(elite_samples, samples, axis=1)
            u = samples*self.std[:, np.newaxis, :] + self.mu[:, np.newaxis, :]
            u = u.reshape(self.nu, -1)
            u.clip(lb, ub, out = u)
            args['u_traj'] = u
            cost = sum(self.robots[name].rollout_map(**args)['cost']*params['belief_'+name] for name in self.robots)
            elite_indices = np.argsort(cost)[0, :self.mpc_params['num_elites']]
            elite_samples = samples[:, elite_indices, :]
            new_mu  = np.mean(elite_samples, axis = 1)
            new_std = np.var(elite_samples, axis = 1)
            
            self.mu  = self.mpc_params['alpha_mu']*self.mu  + (1-self.mpc_params['alpha_mu'])*new_mu
            self.std = self.mpc_params['alpha_std']*self.std + (1-self.mpc_params['alpha_std'])*new_std
        best_cost = cost[elite_indices[0]]
        best_sample = samples[:, elite_indices[0], :]
        best_u = np.clip(best_sample*self.std + self.mu, lb, ub)
        args['u_traj'] = best_u
        res = r.get_inputdict(best_u)
        for rob in self.robots.values():
            best_xi = rob.rollout_xi(**args)['xi']
            res.update(rob.get_statedict(best_xi))
        res.update({k:v for k,v in params.items() if not k in ['q', 'dq']})
        return best_cost, res
