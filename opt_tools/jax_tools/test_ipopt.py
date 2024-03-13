"""Transcribing jax example from Cyipopt.""" 
import jax.numpy as np

from time import perf_counter
from ipopt import nlpsol

def obj(x, p):
    return x[0]*p[0]*np.sum(x[:3])+x[2]

def g(x, p):
    return np.array([np.prod(x)*p[0] - 25, np.prod(x)*p[0]+10])


x0 = np.array([1.0, 3.0, 5.0])
p = np.array([1.0])

lbx = np.full(3, 1)
ubx = np.full(3, 5)

lbg = np.full(2, 0)
ubg = np.full(2, np.inf)

prob = dict(f=obj, g=g, x=x0, p=p)


def opt_param():
    solver = nlpsol(prob=prob)
    print(g(x0, p))

    tic = perf_counter()
    soln = solver(x0=x0, p=np.array([1.0]), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    print(f"first solve in {-tic+perf_counter()}, soln: {soln['x']}")
    tic = perf_counter()
    soln = solver(x0=x0, p=np.array([1.0]), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    print(f"second solve in {-tic+perf_counter()}, soln: {soln['x']}")

def check_derivs():
    solver_check = nlpsol(prob=prob, options={'derivative_test':'second-order'})
    solver_check(x0=x0, p=np.array([1.0]), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    
def opt():
    def obj(x, p):
        return x[0]*x[3]*np.sum(x[:3])+x[2]

    def g(x, p):
        return np.array([np.prod(x)*x[3] - 25])
    x0 = np.array([1.0, 3.0, 5.0, 1.])

    lbx = np.full(4, 1)
    ubx = np.full(4, 5)

    lbg = np.full(1, 0)
    ubg = np.full(1, np.inf)

    prob = dict(f=obj, g=g, x=x0)
    solver = nlpsol(prob=prob)
    print(g(x0, None))
    tic = perf_counter()
    soln = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    print(f"first solve in {-tic+perf_counter()}, soln: {soln['x']}")
    tic = perf_counter()
    soln = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    print(f"second solve in {-tic+perf_counter()}, soln: {soln['x']}")


opt()
opt_param()
check_derivs()


