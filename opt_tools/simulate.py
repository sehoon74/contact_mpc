"""This class rolls out a sys, using a solver to do MPC on it."""
import logging
from dataclasses import dataclass, field, asdict, replace
from functools import partial
from itertools import product
from typing import Callable, Any

import pickle
import matplotlib.pyplot as plt
import jax
#jax.config.update('jax_numpy_dtype_promotion', 'strict')
import jax.numpy as jnp

from jax_tools.sys import AbstractSys
from jax_tools.sysvars import to_traj
from jax_tools.mds_sys import MDSSys, CostParams, StepParams
from jax_tools.brax_sys import BraxSys

key = jax.random.PRNGKey(0)
 
                 

@dataclass
class Results:
    actions: list = field(default_factory=list)
    states: list  = field(default_factory=list)
    costs: list   = field(default_factory=list)

    def postproc(self):
        self.states = to_traj(self.states)
        self.actions = to_traj(self.actions)
        self.total_cost = jnp.sum(jnp.array(self.costs))

@dataclass
class Problem:
    env_name: str
    mpc_name: str
    mpc_spawner: Callable[[BraxSys],Any]
    sim_horizon: int = 500
    save_render: bool = False
    save_results: bool = True
    print_online: bool= False

def rollout(sys: BraxSys,
            step: Callable[['Input', 'State'], 'State'],
            cost: Callable[['Input', 'State'], float],
            mpc: Any,
            sim_horizon: int,
            print_online = False) -> Results:
    res = Results()
    state = sys.reset()
    action = None

    logging.getLogger().info(f"Simulating {type(sys)} with solver {type(mpc)} from x0 {state}")
    print('',  end="")
    for i in range(sim_horizon):
        action = mpc(x0=state, u0=action)
        res.actions.append(action[0])
        res.states.append(sys.from_bstate(state))

        state = step(action[0], state)
        res.costs.append(cost(action[0], state))

        if print_online: print(f'\r Iteration: {i}, Cost: {res.costs[-1]}', end="")
    print('')
    res.postproc()
    return res


def run_env(p: Problem):
    try:
        sys, step, cost = get_brax_sys(p.env_name, backend='generalized')
        mpc = p.mpc_spawner(sys)
        results = rollout(sys, step, cost, mpc,
                          sim_horizon=p.sim_horizon,
                          print_online=p.print_online)
        if p.save_render:
            sys.render(results.states, p.mpc_name+'_'+p.env_name, height=480, width=640)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(e)
        print(f'Error running {p.mpc_name} on {p.env_name}')
    return results


def run_envs(p: Problem):
    envs = ['ant',
            'halfcheetah',
            'hopper',
            'humanoid',
            #'inverted_double_pendulum',
            #'inverted_pendulum',
            'reacher',
            'pusher',
            'swimmer',
            'walker2d',
            ]

    results = {}
    for env in envs:
        r = run_env(replace(p, env_name=env))
        results[env] = r.total_cost
    print(results)
    if p.save_results:
        pickle.dump(results, open(p.mpc_name+'.p', 'wb'))
    return results

def get_brax_sys(env_name, **kwargs):
    sys = BraxSys(env_name, **kwargs)
    step = jax.jit(lambda u, x: sys._env.step(x, u.u))
    cost = jax.jit(sys.cost)
    return sys, step, cost

def get_sss(sys, H=20):
    # unfortunately, the imports  need to be here b/c they change the Float64 config
    from solvers.single_shooting import SingleShootingSolver
    from jax_tools.ipopt import nlpsol
    return SingleShootingSolver(sys, nlpsol, H=H,
                           ipopt_opts={'hessian_approximation':'limited-memory',
                                       'print_level':0,
                                       'max_iter': 10})

def get_cem(sys, H=20, dtype=jax.numpy.float32, **cem_param_args):
    from solvers.cem import CEM, CEMParams
    return CEM(sys, H=H,
               cost_params=CostParams(),
               step_params=StepParams(),
               cem_params=CEMParams(**cem_param_args),
               dtype=dtype
               )

def get_cemmpc(sys, H=20):
    mpc = get_sss(sys, H=H)
    cem = get_cem(sys, H=H, num_iter=5, num_samples=150, dtype=jax.numpy.float64)

    def solver(x0, u0=None):
        u0 = cem(x0, u0)
        return mpc(x0, u0)

    return solver

if __name__ == "__main__":
    #logging.basicConfig(level=logging.WARN)
    logging.basicConfig(level=logging.ERROR)
    prob = Problem(env_name='reacher',
                   mpc_name='cem',
                   mpc_spawner=get_cem,
                   save_render=True,
                   save_results=True,
                   print_online=True)

    #prob = replace(prob, mpc_name='mpc', mpc_spawner=get_sss)
    prob = replace(prob, mpc_name='cemmpc', mpc_spawner=get_cemmpc)
    
    run_envs(prob)

    
    #res = run_env(prob)
    #res.states.plot()
