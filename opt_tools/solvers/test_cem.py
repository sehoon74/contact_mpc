import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

from jax_tools.mds_sys import MDSSys, CostParams, StepParams
from jax_tools.brax_sys import BraxSys

from solvers.cem import CEM, CEMParams

def test_mds_sys(step_size: float, pos: np.ndarray, time_horizon: int, force_zero: bool):
    sys = MDSSys(h=step_size)
    sp = StepParams()
    cp = CostParams()
    x0 = sys.State(pos=pos)
    cem_sys = CEM(sys, time_horizon, step_params=sp, cost_params=cp)
    best_traj = sys.Input().generate_traj(horizon=time_horizon)

    if not force_zero:
        best_traj = cem_sys(x0)

    x_step = sys.step(best_traj, x0, sp=sp)

    cost = sys.cost(best_traj, x_step, cp=cp)
    print("cost", cost)

    x = np.linspace(0, time_horizon * step_size, num=time_horizon)

    ax = plt.figure()
    sub_1 = ax.add_subplot(131)

    sub_1.plot(x, best_traj.force.flatten())
    sub_1.set_xlabel('time')
    sub_1.set_ylabel('force')
    sub_1.set_title("Best Input over time")

    sub_2 = ax.add_subplot(132)
    sub_2.plot(x, x_step.pos.flatten(), label="pos", color='r')
    # sub_2.plot(x, vel, label="vel", color='b')
    sub_2.set_xlabel('time')
    sub_2.set_title("Position")
    sub_2.legend()

    sub_3 = ax.add_subplot(133)
    # y = sys.output(best_traj, x_step, sp=StepParams()).spring_force.flatten()
    sub_3.plot(x, x_step.vel.flatten(), label="vel", color='b')
    sub_3.set_xlabel('time')
    sub_3.set_title("Velocity")
    plt.show()

def test_brax():
    sp = None
    cp = None
    cem_params = CEMParams(alpha_std=0.5,
                           init_std=0.1,
                           num_samples=150,
                           num_iter=30)
    
    sys = BraxSys('reacher')
    x0 = sys.reset()

    cem = CEM(sys, H=30, cem_params=cem_params, cost_params=cp, step_params=sp)
    res = cem(x0)

    print(res)
    #print(np.squeeze(res.u))
    #traj = sys.step(res, x0)
    #cost = sys.cost(res, traj)
    #fig, axs = plt.subplots(3,1)
    #axs[0].plot(jnp.squeeze(traj.q))
    #axs[1].plot(jnp.squeeze(traj.qd))
    #axs[2].plot(cost)
    #plt.show()

def benchmark():
    from timeit import timeit
    sp = None
    cp = None
    cem_params = CEMParams(alpha_std=0.5, num_samples=50, num_iter=30)
    
    sys = BraxSys('reacher')
    x0 = sys.reset()

    my_cem = CEM(sys, H=10, cem_params=cem_params, cost_params=cp, step_params=sp)
    fn = lambda: my_cem(x0)[0].u.block_until_ready()
    print(f'Compiling took {timeit(fn, number=1)}')
    print(f'Step took {timeit(fn, number=10)/10.}') 

    """
    Notes:
    - compiling time arnd 5 seconds
    - exec time propto num_samples
    - passing state as arg to solve speeds up 50ms per solve
    """

if __name__ == "__main__":
    #test_mds_sys(step_size=1e-1, pos=np.array([0.1]), time_horizon=100, force_zero=True)
    #test_mds_sys(step_size=0.05, pos=np.array([0.1]), time_horizon=100, force_zero=False)
    test_brax()
    #benchmark()
