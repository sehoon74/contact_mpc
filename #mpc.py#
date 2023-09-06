# Filippo Rozzi, 2023

from robot import Robot
import casadi as ca
from decision_vars import *
from colorednoise import powerlaw_psd_gaussian

class MPC:
    def __init__(self, robots, params, mpc_params, ipopt_options = {}):
        assert 'free' in robots, "Need at least the free-space model w/ key: free!"
        self.robots = robots

        self.mpc_params = mpc_params
        self.ipopt_options = ipopt_options

        for r in robots.values():
            cost_fn = self.build_cost_fn(r)
            r.build_step(step_size = mpc_params['dt'], cost_fn = cost_fn, jit = mpc_params['jit'])
            r.build_rollout(H = mpc_params['H'], num_samples = mpc_params['num_samples'], jit = mpc_params['jit'])
            
        self.H  = mpc_params['H']   # number of mpc steps
        self.nu = robots['free'].nu

        self.build_solver(params)
        self.icem_init()
    
    def reset_warmstart(self):
        for arg in ['x0', 'lam_x0', 'lam_g0']:
            if arg not in self.__args: continue
            self.__args[arg] = ca.DM.zeros(self.__args[arg].shape)
        self.icem_init()

    def solve(self, params):
        r = self.robots['free']
        params['M_inv'] = r.inv_mass_fn(params['q'])

        self.__args['p'] = self.__pars.vectorize_dict(d = params)

        sol = self.solver(**self.__args)
        if not ca.DM.is_regular(sol['f']): # true if inf or nan
            raise RuntimeError(f'Error solving MPC with: \n params {params} \n\n mpc params {self.mpc_params} \n\n ipopt {self.ipopt_options}')

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

        for k in ext_st:
            if 'contact' in k and k[-1] == 'F':
                pass
#                print(f'Adding contact setpoint cost for {k}')
#                cost += self.mpc_params['force_cost']*ca.sumsqr(self.mpc_params['force_setpoint']-ext_st[k])
        st_cost = ca.Function('st_cost', [*st.values()], [cost], [*st.keys()], ['cost'])
        return st_cost

    def build_solver(self, params0):
        params0['M_inv'] = self.robots['free'].inv_mass_fn(params0['q'])
        self.__pars = ParamDict(params0)

        J = 0
        self.g = []
        self.__vars = DecisionVarDict(attr_names = ['lb', 'ub'])
        self.__vars += self.robots['free'].get_input(self.H)

        step_inputs = self.__vars.get_vars()
        step_inputs['imp_stiff'] = self.__pars['imp_stiff']
        step_inputs['M_inv'] = self.__pars['M_inv']
        xi0 = dict(q = params0['q'], dq = params0['dq'])
        for name, rob in self.robots.items():
            traj, cost, cont_const = mult_shoot_rollout(rob, self.H, xi0, **step_inputs)
            self.__vars += traj
            J += self.__pars['belief_'+name]*cost
            self.g += cont_const

        self.g = ca.vertcat(*self.g)
        self.lbg = ca.DM.zeros(self.g.shape[0])
        self.ubg = ca.DM.zeros(self.g.shape[0])   
            
        #self.add_imp_force_const()
     
        x, lbx, ubx, x0 = self.__vars.get_vectors('sym', 'lb', 'ub', 'init')
        
        prob = dict(f=J, x=x, g=self.g, p=self.__pars.vectorize_attr())
        self.__args = dict(x0=x0, lbx=lbx, ubx=ubx, lbg=self.lbg, ubg=self.ubg)
        self.solver = ca.nlpsol('solver', 'ipopt', prob, self.ipopt_options)
        
    def add_imp_force_const(self):
        args = self.__vars.get_vars()
        args.update(self.__pars.get_vars())
        ext_st = self.robots['free'].get_ext_state(args)
        for i in range(self.H):
            self.g += [ca.sumsqr(ext_st['F_imp'][:,i])]
        self.lbg += [0]*self.H
        self.ubg += [self.mpc_params['max_imp_force']**2]*self.H
        
    def icem_init(self):
        self.mu = np.zeros((self.nu, self.H))
        self.std = 0.1*np.ones((self.nu, self.H))

    def icem_warmstart(self, params, num_iter = None):
        if num_iter is not None: self.mpc_params['num_iter'] = num_iter
        res = self.__vars.dictize(self.__args['x0']) # seemed to be hurting somehow?
        self.mu = res['imp_rest'].full()
        best_cost, res = self.icem_solve(params)
        self.__args['x0'] = self.__vars.vectorize_dict(d = res)
        return best_cost, res 

    def icem_solve(self, params):
        """ Solve from initial state xi0 for """
        r = self.robots['free']
        args= dict(xi0 = r.get_statevec({r.name+k:v for k,v in params.items()}),
                   M_inv = r.inv_mass_fn(params['q']),
                   imp_stiff=params['imp_stiff'])
        elite_samples = powerlaw_psd_gaussian(self.mpc_params['beta'],
                                        size=(self.nu, self.mpc_params['num_elites']-1, self.H))
        elite_samples = np.append(np.zeros((self.nu, 1, self.H)), elite_samples, axis=1)
        ns = self.mpc_params['num_samples']-self.mpc_params['num_elites'] # num new samples
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

    
def mult_shoot_rollout(sys, H, xi0, **step_inputs):
    name = sys.name
    state = sys.get_state(H)
    res = sys.step(**state, **step_inputs)
    continuity_constraints = []
    for st in ['q', 'dq']:
        continuity_constraints += [state[name+st][:, 0] - xi0[st]]
        continuity_constraints += [ca.reshape(res[st][:, :-1] - state[name+st][:, 1:], -1, 1)]
    return state, ca.sum2(res['cost']), continuity_constraints
