from typing import Any, Callable, Dict, Tuple
from types import MethodType
from dataclasses import dataclass
import logging
logger = logging.getLogger()

from cyipopt import Problem
from jax import jit, grad, jacrev, jacfwd
import jax.numpy as np
from jax import config
config.update("jax_enable_x64", True)  # Enable 64 bit floating point precision
config.update('jax_platform_name', 'cpu')  # CPU for IPOPT

Array = Any

@dataclass
class OptProblem:
    f: Callable[[Array, Array], float]  # Scalar objective function, takes arguments of x and p
    g: Callable[[Array, Array], Array] = None  # Constraints, takes arguments of x and p
    x: Array = None  # Decision vars
    p: Array = None  # Parameters which can change btwn optimization calls


class JAXProblem(Problem):
    """Wrapper to cyipopt Problem, which is a cython compiled class."""

    def __init__(self, prob: OptProblem, lb, ub, cl, cu):
        """Initialize an IPOPT problem with JAX for autodiff.
        IN:
          prob: Define fns for objective/constraints, dec vars and params
          lb/ub: lower/upper bounds on x
          cl/cu: lower/upper bounds on g
          skip_hess: avoid building hessian
        """
        n = len(prob.x)
        m = 0

        logger.info(f"Starting JIT compilation with variables {n}")
        self.__obj = jit(prob.f)
        self.__grad = jit(jacfwd(prob.f))  # objective gradient
        logger.info(f'Test eval obj  {self.__obj(np.zeros(n, dtype=np.float64), prob.p)}')
        logger.info(f'Test eval grad {self.__grad(np.zeros(n, dtype=np.float64), prob.p)}')

        self.__obj_hess = jit(jacfwd(jacrev(prob.f)))  # objective hessian
        logger.info("Jit finished")

        self.__p = prob.p # Parameter, can change btwn IPOPT calls

        if prob.g:
            m = len(prob.g(prob.x, prob.p))
            logger.info(f'Adding {m} constraints')
            self.__con = jit(prob.g)  # constraints as array
            self.__con_jac = jit(jacfwd(prob.g))  # constraint jacobian
            self.__con_hess = jit(jacrev(jacfwd(prob.g)))  # constraint hessian
        else:
            self.__con_hess = lambda x,p: np.array([])
            
        super().__init__(n, m, None, lb, ub, cl, cu)
        
        
    def set_p(self, p):
        self.__p = p

    def objective(self, x):
        return self.__obj(x, self.__p)

    def gradient(self, x):
        return self.__grad(x, self.__p)

    def hessian(cls, x, lagrange, obj_factor):
        H = obj_factor * cls.__obj_hess(x, cls.__p)
        H += np.tensordot(lagrange, cls.__con_hess(x, cls.__p), axes=1)
        return H[np.tril_indices(x.size)]

    
    def constraints(self, x):
        return self.__con(x, self.__p)

    def jacobian(self, x):
        return self.__con_jac(x, self.__p)


class nlpsol:
    """Wrapper for Cyipopt matching the CasADi syntax."""

    def __init__(self, name='solver',
                 solver='ipopt',
                 prob: Dict[str, Array] = {},
                 options: Dict[str, Any] = {'print_level': 5}):
                                            #'check_derivatives_for_naninf':'yes',#}):
                                            #'derivative_test':'second-order',}):
                                            #'hessian_approximation':'exact'}):
        self._prob = OptProblem(**prob)  # re-format to dataclass
        self.options = options

    def __call__(self, x0: Array,
                 p: Array = None,
                 lbx: Array = None,
                 ubx: Array = None,
                 lbg: Array = None,
                 ubg: Array = None,) -> Tuple[Array, Any]:
        if not hasattr(self, "_nlp"):
            # We need to build on the first call b/c the numerical vectors are
            #  built into the problem with CyIPOPT, so we need to wait until these avail.
            
            self._nlp = JAXProblem(self._prob,
                                    lb=lbx,
                                    ub=ubx,
                                    cl=lbg,
                                    cu=ubg)

            for option, value in self.options.items():
                try:
                    self._nlp.add_option(option, value)
                except TypeError as e:
                    msg = 'Invalid option for IPOPT: {0}: {1} (Original message: "{2}")'
                    raise TypeError(msg.format(option, value, e))

        if p is not None:
            logger.info(f'Setting params in nlpsol to {p}')
            self._nlp.set_p(p)

        x, info = self._nlp.solve(x0)

        return dict(x=x, info=info)
