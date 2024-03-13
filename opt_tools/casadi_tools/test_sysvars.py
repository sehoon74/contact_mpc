from .sysvars import SysVar, SysVarSet, Sym
import numpy as np
import casadi as ca


class State(SysVarSet):
    pos: SysVar = np.array([3, 1])
    vel: SysVar = SysVar(x0=5., lb=-1, ub=1)


class Input(SysVarSet):
    tau: SysVar = np.ones((3, 1))


def test_vectorization():
    st = State(vel=1)
    assert isinstance(st.pos, Sym)
    print(st.vel)
    print(st.x0)
    print(st.ub)
    print(st.lb)

def test_add():
    st = State(vel=1)
    var = st + Input()
    print(isinstance(var, SysVarSet))
    print(var.pos)
    print(var.x0)
    print(var.lb)

def test_traj():
    st_traj = State.generate_traj(horizon=3)
    print(st_traj.pos)

def test_fromvec():
    print(State.from_vec(np.array([1, 1, 2])))

def test_sym():
    v = ca.SX.sym('vel_test', 3)
    stsym = State(vel=v)
    print(stsym.symvec)

def test_num():
    st = State.skip_sym(pos=np.array([2, 5]), vel=3.)
    assert st.pos[0] == 2
    assert st.vel.shape == (1,1)

    
test_vectorization()
test_add()
test_traj()
test_fromvec()
test_sym()
test_num()
