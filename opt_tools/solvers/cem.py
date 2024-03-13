from dataclasses import dataclass
from typing import Tuple, Any
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import tree_leaves
#from jax.tree_util import tree_map # has issues with class and metadata
from jax_tools.sysvars import tree_map_relaxed as tree_map
from jax_tools.sysvars_helpers import cast_leaves, get_dtype
from flax.struct import dataclass as flax_dataclass

from jax_tools.sys import AbstractSys
from jax_tools.sysvars import SysVarSet, Batch, Traj, TrajBatch, Array
from jax_tools.sysvars_helpers import is_float
Input = SysVarSet
State = SysVarSet
CostParams = dataclass()
StepParams = dataclass()

@dataclass
class CEMParams:
    """Parameters of the CEM planner"""
    alpha_mu: float = 0.8  #: iteration smoothing for mean
    alpha_std: float = 0.4  #: iteration smoothing for covariance
    init_std: float = 1.
    num_samples: int = 100  #: number of input trajectories to roll out per iter
    num_iter: int = 80  #: number of iterations to do
    num_elites: int = 10  #: best samples to keep until next iteration

@flax_dataclass
class CEMState:
    """State of the CEM planner"""
    x0: State
    mu: Traj  #: mean of the sample distribution
    std: Traj  #: covariance of the sample distribution
    elite_u_trajs: TrajBatch  #: best input trajectories

class CEM:
    """Cross-entropy method planner, which plans an input
    trajectory for a system by sampling inputs according to
    a distribution, and iteratively updating the distribution
    to give better performing trajectories"""

    def __init__(self,
                 sys: AbstractSys,
                 H: int,
                 cem_params: CEMParams = CEMParams(),
                 step_params: Any = None,
                 cost_params: Any = None,
                 dtype = jax.numpy.float32):
        """Initialize a CEM planner with planning horizon H"""
        self._H = H  # number of planning horizon steps
        self._sys = sys  # Abstract System
        self._cem_params = cem_params

        key = jax.random.PRNGKey(0)
        self._dtype = dtype
        
        self._step_red = partial(self._step, sp=step_params, cp=cost_params, key=key)
        self._state = self._init_state()

    def _init_state(self, **kwargs) -> CEMState:
        """Initialize a CEMState, updating attr according to **kwargs."""
        kw = dict(
            x0=None,
            mu=self._sys.Input().generate(horizon=self._H),
            std=self._sys.Input().generate(horizon=self._H,
                                           factory=lambda s, l: jnp.full(s, self._cem_params.init_std)),
            elite_u_trajs=self._sys.Input().generate(
                batch_size=self._cem_params.num_elites,
                horizon=self._H,
                factory=lambda s, l: jnp.full(s, 0.))
        )
        kw.update(kwargs)
        state = CEMState(**kw)
        state = cast_leaves(state, self._dtype)
        return state

    def _generate_u_trajs(self, state: CEMState, key) -> Batch:
        """ Generate a set of random input trajectories
            defined by the current mean and covariance."""

        ns = self._cem_params.num_samples - self._cem_params.num_elites  # num new samples

        def sampler(leaf_mean, leaf_cov):
            sample = jax.random.normal(key, (ns, *leaf_mean.shape))
            new_traj = jnp.multiply(sample, leaf_cov) + leaf_mean
            return new_traj

        new_u_trajs = tree_map(sampler, state.mu, state.std, typ=TrajBatch)
        new_u_trajs = cast_leaves(new_u_trajs, self._dtype)
        return state.elite_u_trajs + new_u_trajs

    def _update_mean_cov(
        self, state: CEMState, elite_u_trajs: Batch
    ) -> Tuple[Array, Array]:
        """Update the mean and std in `state` according to
        the updated `state.elite_u_trajs` and smoothing params"""

        def update_mean(mean_leaf, elite_mean_leaf):
            return (1 - self._cem_params.alpha_mu) * mean_leaf + self._cem_params.alpha_mu * elite_mean_leaf

        def update_cov(cov_leaf, elite_std_leaf):
            return (1 - self._cem_params.alpha_std) * cov_leaf + self._cem_params.alpha_std * elite_std_leaf

        elite_mean = elite_u_trajs.reduce_batch(jnp.mean)
        elite_std = elite_u_trajs.reduce_batch(jnp.var)

        new_mu = tree_map(update_mean, state.mu, elite_mean)
        new_std = tree_map(update_cov, state.std, elite_std)

        new_mu = cast_leaves(new_mu, self._dtype)
        new_std = cast_leaves(new_std, self._dtype)
        
        return new_mu, new_std

    def _step(self, sp: StepParams, cp: CostParams, state: CEMState, key):
        """Individual iteration of the CEM algorithm."""
        u_trajs = self._generate_u_trajs(state, key)
        costs = self._sys.get_rollout(u_trajs, state.x0, sp=sp, cp=cp)
        
        # get best num_elites trajectories with best reward function (minimal cost)
        sorted_indices = jnp.argsort(costs)
        elite_indices = sorted_indices[:self._cem_params.num_elites]
        elite_u_trajs = tree_map(lambda l: l[elite_indices, ...], u_trajs)
        # update mean and covariance according to elite_trajectories
        new_mu, new_std = self._update_mean_cov(state, elite_u_trajs)
        return CEMState(x0=state.x0,
                        mu=new_mu,
                        std=new_std,
                        elite_u_trajs=elite_u_trajs)

    @partial(jax.jit, static_argnums=(0,))
    def solve(self, state: CEMState):
        """Find optimal u_traj from x0."""
        state = jax.lax.fori_loop(0,
                                  self._cem_params.num_iter,
                                  lambda i, state: self._step_red(state=state),
                                  state)
        best_trajectory = tree_map(lambda l: l[0, ...],
                                   state.elite_u_trajs,
                                   typ=Traj)
        return best_trajectory

    def __call__(self, x0: State, u0: Traj = None):
        """
        Call solve, where kwargs can contain
         x0: initial state
         u0: used to initialize mu
        """
        kwargs = dict(x0=x0)
        if u0 is not None:
            kwargs['mu'] = u0
        state = self._init_state(**kwargs)
        return self.solve(state)
