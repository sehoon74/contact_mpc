"""Mass-damper dynamics with mean and covariance of state."""

from dataclasses import dataclass, asdict, field
from typing import Union, Callable, Any, Dict, Tuple

from numpy.random import normal
import numpy as np
import jax.numpy as jnp
import jax
from jax.flatten_util import ravel_pytree

from .sysvars import SysVar, SysVarSet, Array
from .sys import AbstractSys

nq = 3  # human DOF
nx = 3  # dim of goal space

@dataclass
class CostParams:
    width: Array = np.full(nx, 1e-2)
    gamma: float = 0.99
    dist_cost: float = 2e-2
    ctrl_cost: float = 1e-5
    vel_cost: float = 1e-3
    jerk_cost: float = 5e-1


@dataclass
class StepParams:
    M: float = 2.
    D: float = 0.3
    sigma_tau: float = 100.
    eps: float = 0.
    effort_angle: float = 0.1
    h: float = 1./50.



class Human(AbstractSys):
    class State(SysVarSet):
        mu: Array = jnp.zeros((2*nq,))

    class Input(SysVarSet):
        tau: Array = jnp.zeros((nq,))
    
    class MPCParams(SysVarSet):
        goal: Array = jnp.zeros((nx,))
        xi0: 'Human.State' = field(default_factory=lambda: Human.State())

    def _cost(self, u: Input, x: State,
              cost_params: CostParams, mpc_params: MPCParams) -> Dict[str, float]:
        mp, cp = mpc_params, cost_params # shorthand
        err = x.mu[...,:nq] - mp.goal
        
        c = jnp.sum(err**2)
        c += cp.ctrl_cost*jnp.sum(u.tau**2)
        c += cp.vel_cost*jnp.sum(x.mu[nq:]**2)

        return c

    def _step(self, u: Input, x: State,
              step_params: StepParams) -> State:
        sp = step_params
        Iq = np.eye(nq)  # shorthand

        # Define dynamics
        A = np.eye(2*nq)
        A[:nq, nq:] = sp.h*Iq
        A[nq:, nq:] -= sp.h*sp.D/sp.M*Iq
        B = np.zeros((2*nq, nq))
        B[nq:, :] = sp.h/sp.M*Iq

        # Build transform
        T = np.eye(nq)
        T[:2, :2] = np.array([[np.cos(sp.effort_angle), -np.sin(sp.effort_angle)],
                              [np.sin(sp.effort_angle),  np.cos(sp.effort_angle)]])

        # Take step w/ noise
        mu_next = A@x.mu + B@u.tau 

        return Human.State(mu=mu_next)
