from __future__ import annotations  # automatic forward declaration of types
from typing import Dict, Union, Callable, overload, Tuple, Any
from typing_extensions import Self
from functools import partial
from dataclasses import asdict

import jax
from jax.tree_util import tree_map
from jax import Array

from .sysvars import SysVarSet, Batch, Traj

Input = SysVarSet
State = SysVarSet
MPCParams = SysVarSet

class AbstractSys:
    """Abstract class for dynamic systems to standarize function signatures."""
    
    def _step(self, u: Input, x: State) -> State:
        """Discrete-time dynamic update equation for singleton input and state."""
        return NotImplementedError(f"Write your _step for derived class {type(self)}")

    def step(self,
             u: Union[Input, Traj[Input], Batch[Input], Batch[Traj[Input]]],
             x: State,
             #aux: Any = None, # was going to add to handle pipeline_state from mjx.
             **kwargs
             ) -> Union[State, Traj[State], Batch[State], Batch[Traj[State]]]:
        """Discrete-time upate by applying input at state.
        In:
          u: Input, can be a singleton, trajectory, batch, or batch of trajectories
          x: State, can be a singleton or a batch
          aux: Additional state which should get carried along traj

        The function is overloaded to allow batches, trajectories, and batches
        of trajectories as input. When multiple batches are applied over the input,
        the batches are stacked in the output into a single batch."""

        step = partial(self._step, **kwargs)
        batch_dim = None
        traj_dim = None
        if isinstance(u, Batch):
            batch_dim = 0
            step = jax.vmap(step, in_axes = (u.batch_dim, u.batch_dim))
            x = tree_map(lambda l: jax.numpy.full((len(u), *l.shape), l), x)
            if isinstance(u, Traj):
                u = tree_map(lambda l: jax.numpy.swapaxes(l, 0, 1), u)
        if isinstance(u, Traj):  # type: ignore
            traj_dim = 0 + int(batch_dim==0)
            def scan_fn(x_: Self.State, u_: Self.Input) -> Tuple[Self.State, Self.State]:
                x_next = step(u_, x_)
                return (x_next, x_next)
            _, x_ret = jax.lax.scan(scan_fn, x, u)
            if isinstance(u, Batch):
                x_ret = tree_map(lambda l: jax.numpy.swapaxes(l, 0, 1), x_ret)
        else:
            x_ret = step(u, x)
        ret_dict = asdict(x_ret)
        ret_dict.update(dict(batch_dim=batch_dim, traj_dim=traj_dim))
        return type(x_ret)(**ret_dict)

    def _cost(self, u: Input, x: State, **kwargs) -> float:
        """Cost for a singleton input and state."""
        return NotImplementedError(f"Write your _cost for derived class {type(self)}")

    def cost(self, u: Union[Input, Traj[Input], Batch[Input]],
                   x: Union[State, Traj[State], Batch[State]], **kwargs
             ) -> Union[float, Batch[float]]:
        red_cost = partial(self._cost, **kwargs)
        if isinstance(u, Batch) and isinstance(x, Batch) and isinstance(u, Traj):
            #traj_cost = lambda u_, x_: jax.numpy.sum(jax.vmap(red_cost, in_axes=(u_.traj_dim, x_.traj_dim))(u_, x_))
            traj_cost = lambda u_, x_: jax.numpy.sum(jax.vmap(red_cost, in_axes=(0, 0))(u_, x_))
            return jax.vmap(traj_cost, in_axes=(u.batch_dim, x.batch_dim))(u, x)
        if isinstance(u, Traj) and isinstance(x, Traj) and not isinstance(u, Batch) and not isinstance(x, Batch):
            return jax.numpy.sum(jax.vmap(red_cost, in_axes=(u.traj_dim,x.traj_dim))(u, x))
        return red_cost(u, x)

    def _output(self, u: Input, x: State, **kwargs) -> Output:
        """Output for a singleton input and state."""
        return NotImplementedError(f"Write your _output for derived class {type(self)}")

    def output(self, u: Union[Input, Traj[Input], Batch[Input]],
                     x: Union[State, Traj[State], Batch[State]], **kwargs
               ) -> Union[Output, Traj[Output], Batch[Output]]:
        output = partial(self._output, **kwargs)
        batch_dim = None
        traj_dim = None
        if isinstance(u, Batch) and isinstance(x, Batch):
            output = jax.vmap(output, in_axes=(u.batch_dim, x.batch_dim))
            batch_dim = 0
        if isinstance(u, Traj) and isinstance(x, Traj):
            traj_dim = 0 + int(batch_dim==0)
            output = jax.vmap(output, in_axes=(u.traj_dim, x.traj_dim))
        ret = output(u, x)
        ret_dict = asdict(ret)
        ret_dict.update(dict(batch_dim=batch_dim, traj_dim=traj_dim))
        return type(ret)(**ret_dict)


    def get_vec_rollout(self, u_traj: Traj[Input],
                        x0: State,
                        **static_args
                        ) -> Callable[[Array], float]:
        _, u_unraveller = jax.flatten_util.ravel_pytree(u_traj)
        _, param_unraveller = jax.flatten_util.ravel_pytree(x0)

        def vec_rollout(flat_opt_args, flat_param_args):
            _u_traj = u_unraveller(flat_opt_args)
            _mpc_param = param_unraveller(flat_param_args)
            return self.get_rollout(_u_traj, _mpc_param, **static_args)
            #x_traj = self.step(_u_traj, _mpc_param, **static_args)
            #return self.cost(_u_traj, x_traj, **static_args)
        return vec_rollout, u_unraveller

    def get_rollout(self,
                    u_traj: Traj[Input] | TrajBatch[Input],
                    x0: State,
                    **static_args
                    ) -> Array:
        """Return cost of a rollout of u_traj from x0."""
        x_traj = self.step(u_traj, x0, sp=static_args.get('sp', None))
        return self.cost(u_traj, x_traj, cp=static_args.get('cp', None))
