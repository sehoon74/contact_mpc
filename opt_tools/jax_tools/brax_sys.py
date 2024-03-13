# @Kevin Haninger 02.2024, initial prototype
import logging
from typing import Any, Union, Tuple, Callable
from dataclasses import field
from types import MethodType
import os
if not os.environ.get('MUJOCO_GL', None):
    logging.getLogger().info("No env var MUJOCO_GL detected,"
                           "using software rendering (ie slow)")
    os.environ['MUJOCO_GL'] = 'osmesa'

import mediapy as media

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from brax import envs
from brax.envs.base import State as BState

from .sysvars import SysVarSet, SysVar, Array, Traj, Batch
from .sysvars_helpers import get_dtype, cast_leaves
from .sys import AbstractSys, Input

class BraxSys(AbstractSys):
    """Shim for Brax environments."""
    class Input(SysVarSet):
        u: Array = jnp.zeros(1)

    class State(SysVarSet):
        q: Array = jnp.zeros(1)
        qd: Array = jnp.zeros(1)

    class MPCParams(State):
        pass

    def __init__(self, env_name, backend='generalized', key = jax.random.PRNGKey(0)):
        self._key = key
        self.setup_env(env_name, backend)

        self.Input = self.Input.update_defaults(u=jnp.zeros(self._env.action_size))
        self.State = self.State.update_defaults(q=jnp.zeros(self._env.sys.q_size()),
                                                qd=jnp.zeros(self._env.sys.qd_size()))

    def setup_env(self, env_name: str, backend: str):
        env = envs.get_environment(env_name, backend=backend)
        # We monkeypatch pipeline step to drop the inner lax.scane, which somehow kills jit
        def pipeline_step(cls, pipeline_state: Any, action: jax.Array) -> BState:
            return cls._pipeline.step(cls.sys, pipeline_state, action, cls._debug)
        env.pipeline_step = MethodType(pipeline_step, env)

        self._env = env

    def reset(self) -> State:
        return self._env.reset(self._key)

    def _step(self, u: Input, x: BState, sp: Any = None) -> State:
        return self._env.step(x, u.u)
    
    def _cost(self, u: Input, x: BState, cp: Any = None):
        return jax.lax.cond(x.done,
                            lambda: 0.*jnp.array([x.reward]), #if done==true
                            lambda: -jnp.array([x.reward])) # else

    def from_bstate(self, bstate: BState) -> State:
        return self.State(q=bstate.pipeline_state.q,
                          qd=bstate.pipeline_state.qd)

    def render(self, x_traj: Traj, name='render', **kwargs) -> None:
        bstate_list = [x for x in x_traj]
        frames = self._env.render(bstate_list, **kwargs)
        media.write_video(name+'.mp4', frames, fps=1./self._env.sys.dt)

    def get_rollout(self,
                    u: Union[Traj, Batch],
                    x0: State,
                    cast: Callable[[Any], Any] = lambda x: x,
                    **static_args) -> Array:
        """Roll out system from x0, applying the trajectory of u.

        We overwrite the AbstractSys get_rollout b/c the cost is in state for
        brax, which allows us to be _slightly_ more efficient.
        
        IN:
          u: Input, either trajectory or trajbatch
          x0: Initial state
          cast: Cast a pytree to the desired float precision
          **static_args are used to pass any static transforms, currently ignored
        OUT:
          Total cost as array of size (batch_size,)"""
        
        dtype = get_dtype(x0)
        x0 = cast_leaves(x0, dtype)
        u = cast_leaves(u, dtype)
        step = lambda u, x: cast_leaves(self._env.step(x, u.u), dtype)
        cost = self.cost
        batch_dim = None
        traj_dim = None
        if isinstance(u, Batch):
            batch_dim = 0
            step = jax.vmap(step, in_axes = (u.batch_dim, u.batch_dim))
            cost = jax.vmap(cost, in_axes = (u.batch_dim, u.batch_dim))
            x0 = tree_map(lambda l: jax.numpy.full((len(u), *l.shape), l), x0)
            u = tree_map(lambda l: jax.numpy.swapaxes(l, 0, 1), u)
        if isinstance(u, Traj):  # type: ignore
            traj_dim = 0 + int(batch_dim==0)
            def scan_fn(x_: BState, u_: Input) -> Tuple[BState, Array]:
                x_next = step(u_, x_)
                return cast(x_next), cost(u_, x_next)
            x_last, cost = jax.lax.scan(scan_fn, x0, u)
        else:
            raise TypeError(f"input u_traj should be Traj, got {type(u_traj)}")
        return jnp.squeeze(jnp.sum(cost, axis=batch_dim))
