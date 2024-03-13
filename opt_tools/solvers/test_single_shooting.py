import logging
logging.basicConfig(level=logging.WARN)

from solvers.single_shooting import SingleShootingSolver

def jax_test():
    import jax
    from jax_tools.human import Human, CostParams, StepParams
    from jax_tools.ipopt import nlpsol

    H = 15
    sp = StepParams()
    cp = CostParams()
    mp = Human.MPCParams(goal=jax.numpy.array([0.3,0,0]))
    hum = Human()

    sss = SingleShootingSolver(hum, nlpsol, H=H, step_params=sp, cost_params=cp)
    print(sss(mp))
    #print(sss(mp))

def brax_test():
    import jax.numpy as jnp
    from jax_tools.brax_sys import BraxSys
    from jax_tools.ipopt import nlpsol

    H = 5
    sp = None
    cp = None
    mp = None

    sys = BraxSys('inverted_double_pendulum',
                  #backend='mjx',
                  #float_dtype=jnp.float64
                  )
    sss = SingleShootingSolver(sys, nlpsol, H=H,
                               ipopt_opts={'hessian_approximation':'limited-memory',})
    print(sss(sys.reset()))
    print(sss(sys.reset()))

def casadi_test():    
    from casadi_tools.human import Human, CostParams, StepParams
    from casadi import nlpsol
    import numpy as np

    H = 5
    sp = StepParams()
    cp = CostParams()
    mp = Human.MPCParams(goal=np.array([0.3,0,0]))
    hum = Human()

    sss = SingleShootingSolver(hum, nlpsol, sp, cp, H=H,)
    print(sss(mp))
    print(sss(mp))

#jax_test()
brax_test()
#casadi_test()
