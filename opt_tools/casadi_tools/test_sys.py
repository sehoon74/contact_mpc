from dataclasses import dataclass
import casadi as ca
from .sys import vectorize_fn_eval
from .sysvars import SysVarSet, SysVar

class State(SysVarSet):
    pos: SysVar = 0.
    vel: SysVar = 0.

@dataclass
class StepParams:
    h: float = 0.1
    M: float = 3.
    
def step(x: State):
    return State(pos=x.pos+0.1*x.vel, vel=x.vel+0.2)

def step2(x: State, sp: StepParams):
    return State(pos=x.pos+sp.h*x.vel, vel=x.vel+0.2)
    
def test_vectorize():
    st = State.generate_traj(horizon=5)
    x_traj = vectorize_fn_eval(step, traj_args={'x': st})
    print(x_traj.pos.shape)
    assert x_traj.pos.shape == (1,5), "Error in traj shape"

def test_vec_with_static():
    st= State.generate_traj(horizon=5)
    x_traj = vectorize_fn_eval(step2, {'x':st}, sp=StepParams())
    print(x_traj.pos)
    

test_vectorize()
test_vec_with_static()
