from dataclasses import asdict


import jax.numpy as jnp
from jax.tree_util import tree_flatten

from .sysvars import SysVarSet, SysVar, Traj, Batch, TrajBatch, Array, tree_map_relaxed

#### Tests! ####
class St(SysVarSet):
    pos: SysVar = SysVar(x0=jnp.ones(2), ub=4)
    vel: SysVar = SysVar(x0=3., lb = 2.)
    acc: SysVar = 1.0

class St2(SysVarSet):
    pos: SysVar = jnp.zeros(2)
    
def test_vec():
    st = St()
    assert jnp.allclose(st.pos, jnp.ones(2))
    assert jnp.allclose(st.x0, jnp.array([1, 1, 3, 1]))
    assert jnp.allclose(st.lb, jnp.array([-jnp.inf, -jnp.inf, 2., -jnp.inf]))
    assert jnp.allclose(st.ub, jnp.array([4, 4, jnp.inf, jnp.inf]))

def test_traj():
    stt = St().generate_traj(horizon=4)
    assert isinstance(stt, St)
    assert isinstance(stt, Traj)
    assert not isinstance(stt, Batch)
    assert jnp.allclose(stt.pos, jnp.ones((4,2)))
    assert jnp.allclose(stt.x0, jnp.array([*[1]*8, *[3]*4, *[1]*4]))
    assert jnp.allclose(stt.lb, jnp.array([*[-jnp.inf]*8, *[2]*4, *[-jnp.inf]*4]))
    assert(stt.traj_dim == 0)
    assert(stt.batch_dim is None)
    assert(isinstance(stt, Traj))
    st = stt.reduce_traj(jnp.mean)
    assert jnp.allclose(st.pos, jnp.ones(2))
    assert not isinstance(st, Traj)

    # Test direct def from initialize
    stt2 = St(pos=jnp.ones((4,2)), vel=jnp.zeros((4,1)), acc=jnp.ones((4,1)), traj_dim=0)
    assert isinstance(stt2, Traj)
    stt2 = St(traj_dim = 0)
    st2 = St2()
    assert not isinstance(st2, Traj)
    assert not isinstance(st2, Batch)

def test_batch():
    stb = St().generate_batch(batch_size=5)
    assert len(stb) == 5
    assert jnp.allclose(stb.pos, jnp.ones((5,2)))
    assert jnp.allclose(stb.x0, jnp.array([*[1]*10, *[3]*5, *[1]*5]))
    assert jnp.allclose(stb.lb, jnp.array([*[-jnp.inf]*10, *[2]*5, *[-jnp.inf]*5]))
    assert(stb.traj_dim is None)
    assert(stb.batch_dim == 0)
    assert(isinstance(stb, Batch))
    st = stb.reduce_batch(jnp.mean)
    assert jnp.allclose(st.pos, jnp.ones(2))
    assert not isinstance(st, Batch)

def test_batch_add():
    stb1 = St().generate_batch(batch_size=2)
    stb2 = St().generate_batch(batch_size=3)
    stb = stb1+stb2
    assert stb.pos.shape == (5,2)
    assert sum(1 for _ in (iter(stb))) == 5, 'Iterator didnt return 5 elements'
    
def test_traj_batch():
    stt = St().generate_traj(horizon=3)
    sttb = stt.generate_batch(batch_size=4)
    assert sttb.pos.shape == (4,3,2)
    assert isinstance(sttb, Batch)
    assert isinstance(sttb, Traj)
    assert isinstance(sttb, TrajBatch)

    cls = type(sttb)
    assert issubclass(cls, TrajBatch)
    assert issubclass(cls, Traj)
    assert issubclass(cls, Batch)
    
    st2tb = St2(batch_dim = 0, traj_dim=1)
    assert isinstance(st2tb, TrajBatch)
    st = St()
    assert not isinstance(st, Traj)
    assert not isinstance(st, Batch)
    st2 = St2()
    assert not isinstance(st2, Traj)
    assert not isinstance(st2, Batch)
    stb = St(batch_dim=0)
    assert not isinstance(stb, Traj)

    st2t = St2(traj_dim=0)
    assert not isinstance(st2t, Batch)

    stt = St().generate_traj(horizon=3)
    assert not isinstance(stt, Batch)

def test_treemap():
    """Issues with treemapping over trajs/batchs."""
    st = St()
    stt = st.generate_traj(horizon=3)
    stb = st.generate_batch(batch_size=4)
    sttb = stt.generate_batch(batch_size=4)
    assert st._treedefs == stt._treedefs
    assert st._treedefs == stb._treedefs
    assert st._treedefs == sttb._treedefs
    
    stt = tree_map_relaxed(lambda l: l, st, typ = Traj)
    stb = tree_map_relaxed(lambda l: l, st, typ = Batch)
    sttb = tree_map_relaxed(lambda l: l, st, typ = TrajBatch)
    assert st._treedefs == stt._treedefs
    assert st._treedefs == stb._treedefs
    assert st._treedefs == sttb._treedefs

    
    st2 = tree_map_relaxed(lambda l: jnp.ones(3), st)
    stt = tree_map_relaxed(lambda l: jnp.ones(3), st, typ=Traj)
    assert isinstance(stt, Traj)
    stb = tree_map_relaxed(lambda l: jnp.ones(3), st, typ=Batch)
    assert isinstance(stb, Batch)
    sttb = tree_map_relaxed(lambda l: jnp.ones(3), st, typ=TrajBatch)
    assert isinstance(sttb, TrajBatch)

    sttb2 = tree_map_relaxed(lambda l: l, sttb)
    assert isinstance(sttb2, TrajBatch)

    st = tree_map_relaxed(lambda l: l, sttb, typ=SysVarSet)
    assert not isinstance(st, Traj)
    assert not isinstance(st, Batch)

def test_treedef_construction():
    """Issues with setting _treedefs only in the first custom class."""

    st = St()
    stt = tree_map_relaxed(lambda l: l, st, typ=Traj)
    stb = tree_map_relaxed(lambda l: l, st, typ=Batch)
    sttb = tree_map_relaxed(lambda l: l, st, typ=TrajBatch)
    assert not isinstance(st, Traj)
    assert not isinstance(st, Batch)
    assert isinstance(stt, Traj)
    assert isinstance(stb, Batch)
    assert isinstance(sttb, TrajBatch)

    stt = St(traj_dim=0, pos=jnp.ones(2))
    st = tree_map_relaxed(lambda l: l, stt, typ=SysVarSet)
    stb = tree_map_relaxed(lambda l: l, stt, typ=Batch)
    sttb = tree_map_relaxed(lambda l: l, stt, typ=TrajBatch)
    assert not isinstance(st, Traj)
    assert not isinstance(st, Batch)
    assert isinstance(stt, Traj)
    assert isinstance(stb, Batch)
    assert isinstance(sttb, TrajBatch)


def test_update_defaults():
    St2 = St.update_defaults(pos=jnp.ones(4))
    st = St2()
    stt = St2().generate_traj(horizon=5)
    assert jnp.allclose(st.pos, jnp.ones(4))
    assert jnp.allclose(st.x0, jnp.array([1, 1, 1, 1, 3, 1]))
    assert jnp.allclose(stt.pos, jnp.ones((5,4)))
    assert jnp.allclose(st.lb, jnp.array([*[-jnp.inf]*4, 2., -jnp.inf]))

if __name__ == "__main__":
    tests = [
        test_vec,
        test_traj,
        test_batch,
        test_traj_batch,
        test_batch_add,
        test_treemap,
        test_treedef_construction,
        test_update_defaults
    ]
    for fn in tests:
        print(f"Testing {fn.__name__}")
        fn()
    print("Done with tests")
