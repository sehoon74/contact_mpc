# @Kevin Haninger 10.2023, initial prototype
from typing_extensions import Annotated
from dataclasses import asdict, dataclass

import jax.numpy as np
from jax import Array  # Recommended as return type
from jax.typing import ArrayLike  # More permissive, rec for input type
from flax import struct

from .sysvars import SysVarSet, SysVar, Array
from .sys import AbstractSys

@struct.dataclass
class StepParams:
    env_stiff: float = 50.0
    env_rest_pos: float = 0.3

@dataclass
class CostParams:
    des_pos: float = 0.5

class MDSSys(AbstractSys):
    """Mass, damper, spring system"""

    # We define the system variables within the class so
    # they can be accessed as attributes of the system
    # wherever the system gets passed to.

    class State(SysVarSet):
        pos: SysVar = np.zeros(1)
        vel: SysVar = np.zeros(1)

    class Input(SysVarSet):
        force: SysVar = np.zeros(1)

    class Output(SysVarSet):
        spring_force: SysVar = np.zeros(1)

    def __init__(self, h: float):
        self.h = h  # step size in seconds
        self.D = 0.25  # damping coefficient, normalized by sqrt stiffness

    def _step(self, u: Input, x: State, sp: StepParams = StepParams()) -> State:
        """Discrete-time step update of dynamics by
        semi-implicit integration, mass=1."""
        # Semi-implicit means we calculate the damping force
        # based on the next speed. This is typically more stable
        # at higher stiffnesses.
        effective_damping = 1.0 / (1.0 + np.sqrt(sp.env_stiff) * self.D * self.h)
        vel = effective_damping * x.vel + self.h * (u.force + self._get_force(x, sp))
        pos = x.pos + self.h * vel

        return x.replace(pos=pos, vel=vel) # need replace to keep type otherwise JAX freaks out

    def _get_force(self, x: State, sp: StepParams) -> float:
        """Get spring force, 0 if not in contact"""
        return -sp.env_stiff * np.maximum(0, x.pos - sp.env_rest_pos)

    def _cost(self, u: Input, x: State, cp: CostParams = CostParams()) -> float:
        """Get the stage cost"""
        return (
            np.square(x.pos - cp.des_pos)
            + 1e-2 * np.square(x.vel)
           + 1e-5 * np.square(u.force)
        )

    def _output(self, u: Input, x: State, sp: StepParams, noise:float=0.0) -> Output:
        """Get the output"""
        return self.Output(spring_force=self._get_force(x, sp)+noise)

def test_elem():
    sys = MDSSys(h=0.05)
    x0 = sys.State(pos=1.0, vel=0.0)
    u0 = sys.Input(force=3 * np.ones((1,)))
    sp = StepParams()#env_stiff=0.)
    cp = CostParams()
    res = sys.step(u0, x0, sp = sp)
    cost = sys.cost(u0, x0, cp = cp)
    out = sys.output(u0, x0, sp = sp)
    assert res.pos.shape == (1,), "State update has wrong shape"
    assert cost.shape == (1,), "Cost has wrong shape"
    assert out.spring_force.shape == (), "Output has wrong shape"

def test_traj():
    sys = MDSSys(h=0.05)
    num_steps = 25
    x0 = sys.State(pos=np.ones(1,), vel=np.ones(1,))
    sp = StepParams()
    us = sys.Input().generate_traj(force=3*np.ones(1), horizon=num_steps)

    xs = sys.step(us, x0, sp = sp)
    cost = sys.cost(us, xs, cp = CostParams())
    out = sys.output(us, xs, sp = sp)

    assert xs.pos.shape == (num_steps, 1), "Error with trajectory rollout"
    assert cost.shape == (), "Cost should be scalar"
    assert isinstance(xs, Traj), "Result of step should be traj"
    assert isinstance(out, Traj), "Result of output shoul be traj"
    
    
def test_traj_batch():
    sys = MDSSys(h=0.05)
    num_steps = 5
    num_batch = 3
    x0 = sys.State(pos=np.ones(1,), vel=np.ones(1,),)
    sp = StepParams()
    
    u_traj = sys.Input(force=np.ones(1)).generate_traj(horizon=num_steps)
    u_traj_batch = u_traj.generate_batch(batch_size = num_batch)
    x_batch = sys.step(u_traj_batch, x0, sp=sp)
    y_batch = sys.output(u_traj_batch, x_batch, sp=sp)
    assert x_batch.pos.shape == (num_batch, num_steps, 1), "Error with the batch rollout"
    assert isinstance(x_batch, Batch), 'x_batch should be Batch'
    assert isinstance(y_batch, Batch), 'y_batch should be Batch'
    assert isinstance(y_batch, Traj), 'y_batch should also be a Traj'
    #c_batch = sys.cost(us, x_batch, p)
    #assert c_batch.shape == (num_batch,1)


if __name__ == "__main__":
    from .sysvars import Traj, Batch
    tests = [
        test_elem,
        test_traj,
        test_traj_batch
    ]
    for fn in tests:
        print(f"Testing {fn.__name__}")
        fn()
    print("Done with tests")
