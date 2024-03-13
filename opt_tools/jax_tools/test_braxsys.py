from brax_sys import *

def test_braxsys():
    sys = BraxSys('inverted_double_pendulum')
    x0 = sys.reset()
    u0 = sys.Input(u=jnp.full(sys._env.action_size, 0.01))
    res = sys.step(u0, x0)
    cost = sys.cost(u0, x0)

    assert res.q.shape == (sys._env.sys.q_size(),), "State update has wrong shape"
    assert cost.shape == (1,), "Cost has wrong shape"

def test_braxsys_traj():
    sys = BraxSys('inverted_double_pendulum')
    x0 = sys.reset()
    u0 = sys.Input(u=jnp.full(sys._env.action_size, 3.1)).generate(horizon=20)
    #res = sys.step(u0,x0)
    #print(res.q)
    #print(sys.cost(u0, res).shape)
    print(sys.get_rollout(u0, x0))
    
    u0 = sys.Input(u=jnp.full(sys._env.action_size, 3.1)).generate(horizon=20, batch_size=1)
    u0 += sys.Input(u=jnp.full(sys._env.action_size, -3.1)).generate(horizon=20, batch_size=1)
    #res = sys.step(u0, x0)
    #cost = sys.cost(u0, res)
    #cost.block_until_ready()
    #print(res.q)
    #print(cost)
    print(sys.get_rollout(u0, x0))

def test_braxsys_grads():
    # Notes w/ double_inverted_pendulum:
    # _step vs step: minimal overhead
    # _pstep vs _step:
    #   pstep compiles 5x faster, execs 2x faster
    #   monkeypatching pipeline_step to remove the lax.scan makes them about ==
    # generalized vs mjx:
    #   compiling is 3x-6x slower (worse at longer traj)
    #   exec for single step is ==, traj5 is 10x slower!!!
    # fwd vs rev mode:
    #   on the generalized backend, step_fwd compiles 5x faster, executes 3x faster
    #   on the mjx backend, step_rev doesn't even compile (while loop in mjx solve!)
    # generalized vs mjx: on step_fwd, generalized compiles 3x faster, but same exec
    # traj:
    #   horizon=5  compiles 3x slower and calls 2x slower
    #   horizon=15 compiles 3x slower and calls 4x slower
    # cost seems linear in exec costs
    
    sys = BraxSys('inverted_double_pendulum')
    sys_mjx = BraxSys('inverted_double_pendulum', backend='mjx')
    x0 = sys.reset()
    x_5 = x0.generate(horizon=5)
    u0 = sys.Input(u=jnp.full(sys._env.action_size, 3.1))
    u_5 = sys.Input(u=jnp.full(sys._env.action_size, 3.1)).generate(horizon=5)
    u_15 = sys.Input(u=jnp.full(sys._env.action_size, 3.1)).generate(horizon=15)
    
    step_fwd = jax.jit(jax.jacfwd(sys.step))
    mjxstep_fwd = jax.jit(jax.jacfwd(sys_mjx.step))
    step_rev = jax.jit(jax.jacrev(sys.step))
    pstep_fwd  = jax.jit(jax.jacfwd(sys._pstep))
    cost_fwd = jax.jit(jax.jacfwd(sys.cost))
    
    fns = dict(
        step_fwd = lambda: step_fwd(u0, x0),
        mjxstep_fwd = lambda: mjxstep_fwd(u0, x0),
        #pstep_fwd = lambda: pstep_fwd(u0, x0),
        #step_rev = lambda: step_rev(u0, x0),
        traj5_fwd = lambda: step_fwd(u_5, x0),
        mjxtraj5_fwd = lambda: mjxstep_fwd(u_5, x0),
        #traj15_fwd = lambda: step_fwd(u_15, x0),
        #cost_fwd = lambda: cost_fwd(u0, x0),
        #cost5_fwd = lambda: cost_fwd(u_5, x_5)
    )

    for n, fn in fns.items():
        print(f'Compiling {n} took {timeit(fn, number=1)}')

    for n, fn in fns.items():
        print(f'Step {n} took {timeit(fn, number=100)}') 

def test_render():
    sys = BraxSys('inverted_double_pendulum')
    x0 = sys.reset()
    u0 = sys.Input(u=jnp.full(sys._env.action_size, 0.1)).generate(horizon=200)
    res = sys.step(u0, x0)
    sys.render(res)

        
def benchmark():
    
    test = BraxSys()
    x = test.reset()
    u = test.Input()
    batch_size = 100
    u_batch = test.Input().generate_batch(batch_size=batch_size)
    
    xbrax = sys._env.reset(jax.random.PRNGKey(0))

    step = jax.jit(test.step)
    step_red = jax.jit(test._step)
    stepbrax = jax.jit(sys._env.step)
    step_batch = jax.jit(test.step)
    step_raw_batch = jax.jit(jax.vmap(test._step, in_axes=(0, None)))
    #envmjx = envs.get_environment(env_name, backend='mjx')
    #stepmjx = jax.jit(envmjx.step)

    fns = dict(
        braxsys = lambda: step(u, x).q.block_until_ready(),
        braxsys_red = lambda: step_red(u, x).q.block_until_ready(),
        brax = lambda: stepbrax(xbrax, u.u).pipeline_state.q.block_until_ready(),
        batch_braxsys = lambda: step_batch(u_batch, x).q.block_until_ready(),
        batch_raw_braxsys = lambda: step_raw_batch(u_batch, x).q.block_until_ready(),
        #mjx = lambda: stepmjx(xbrax, u.u).pipeline_state.q.block_until_ready(),
    )

    for n,fn in fns.items():
        print(f'Compiling {n} took {timeit(fn, number=1)}')

    for n, fn in fns.items():
        print(f'Step {n} took {timeit(fn, number=100)}') 


if __name__ == "__main__":
    from timeit import timeit
    #test_braxsys()
    #test_render()
    test_braxsys_traj()
    #test_braxsys_grads()
    #benchmark()
