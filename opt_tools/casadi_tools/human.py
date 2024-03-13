"""Mass-damper dynamics with mean and covariance of state."""

from dataclasses import dataclass, asdict, field
from typing import Union, Callable, Any, Dict, Tuple
from typing_extensions import Self

import casadi as ca
from numpy.random import normal
import numpy as np

from .sysvars import SysVar, SysVarSet, Array, Sym
from .sys import AbstractSys

nq = 3  # human DOF
nx = 3  # dim of goal space

nlpopts = {"print_time": False,
           "verbose": False,
           "ipopt.print_level": 0, }
# "ipopt.tol":1e-16}


@dataclass
class CostParams:
    width: Array = np.full(nx, 1e-2)
    gamma: float = 0.99
    dist_cost: float = 2e-0
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
    h: float = 1./10.



class Human(AbstractSys):
    class State(SysVarSet):
        mu: SysVar = np.zeros(2*nq)

    class Input(SysVarSet):
        tau: SysVar = np.zeros(nq)

    class MPCParams(State):
        goal: SysVar = np.zeros(nx)
        
    def _cost(self, u: Input, x: State,
              mpc_params: MPCParams, cost_params: CostParams, **kwargs
              ) -> Dict[str, Union[Sym, float]]:
        mp, cp = mpc_params, cost_params # shorthand

        
        err = x.mu[:nq] - mp.goal
        
        c = ca.sumsqr(err)
        c += cp.ctrl_cost*ca.sumsqr(u.tau)
        c += cp.vel_cost*ca.sumsqr(x.mu[nq:])

        return {'c':c}

    def _step(self, u: Input, x: State,
              step_params: StepParams, **kwargs) -> State:
        sp = step_params # shorthand
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
        mu_next = A@x.mu + B@u.tau + \
            B@T@(np.eye(nq) + np.diag([sp.eps]*nq))@u.tau

        return Human.State(mu=mu_next)

