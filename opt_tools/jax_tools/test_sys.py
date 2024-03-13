import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)  # Enable 64 bit floating point precision
config.update('jax_platform_name', 'cpu')  # CPU for IPOPT

from dataclasses import asdict
from functools import partial

from .human import Human, CostParams, StepParams, nq
from .sysvars import Traj, Batch

hum = Human()
sp = StepParams()
cp = CostParams()
mp = hum.MPCParams()

H = 10
B = 5

u0 = hum.Input(tau=jnp.ones((nq,)))
x0 = hum.State()
u_traj = u0.generate_traj(horizon=H)
u_batch = u0.generate_batch(batch_size=B)

def test_step():
    x_next = hum.step(u0, x0, step_params=sp)
    assert x_next.mu.shape == (2*nq,)
    assert all(x_next.mu[:nq]>=0)
    assert not isinstance(x_next, Traj)
    assert not isinstance(x_next, Batch)

    x_traj = hum.step(u_traj, x0, step_params=sp)
    assert x_traj.mu.shape == (H, 2*nq)
    assert isinstance(x_traj, Traj)
    assert not isinstance(x_traj, Batch)

def test_step_batch():
    x_batch = hum.step(u_batch, x0, step_params=sp)
    assert x_batch.mu.shape == (B, 2*nq)
    assert not isinstance(x_batch, Traj)
    assert isinstance(x_batch, Batch)

    u_batchtraj = u_traj.generate_batch(batch_size=B)
    x_batchtraj = hum.step(u_batchtraj, x0, step_params=sp)
    assert x_batchtraj.mu.shape == (B, H, 2*nq)
    assert isinstance(x_batchtraj, Traj)
    assert isinstance(x_batchtraj, Batch)
    
def test_cost():
    cost = hum.cost(u0, x0, cost_params=cp, mpc_params=mp)
    assert cost.shape == ()
    
    x_traj = hum.step(u_traj, x0, step_params=sp)
    cost = hum.cost(u_traj, x_traj, cost_params=cp, mpc_params=mp)
    assert cost.shape == ()

def test_dtypes():
    step = jax.jit(partial(hum.step, step_params=sp))
    u0 = hum.Input(tau=jnp.ones((nq,),dtype=jnp.float64))
    u_traj = u0.generate_traj(horizon=30)
    x0 = hum.State()
    x = step(u0, x0)
    assert x.mu.dtype == jnp.float64
    x = step(u_traj, x0)
    assert x.mu.dtype == jnp.float64
    u_tb  = u_traj.generate_batch(batch_size=3)
    assert step(u_tb, x0).mu.dtype == jnp.float64

def test_cost_batch():
    pass
    

if __name__ == "__main__":
    tests = [
        test_step,
        test_step_batch,
        test_cost,
        test_dtypes
    ]
    for fn in tests:
        print(f"Testing {fn.__name__}")
        fn()
    print("Done with tests")
