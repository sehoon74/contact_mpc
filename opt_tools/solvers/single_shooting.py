import logging
logger = logging.getLogger()

from jax.numpy import float64
from jax.flatten_util import ravel_pytree
from jax_tools.sysvars_helpers import cast_leaves


class SingleShootingSolver:
    def __init__(self,
                 sys,
                 solver,
                 H: int,
                 ipopt_opts = {},
                 **static_args):

        u_traj = cast_leaves(sys.Input().generate_traj(horizon=H), float64)
        x0 = cast_leaves(sys.reset(), float64)

        obj, self.u_traj_unflatten = sys.get_vec_rollout(u_traj,
                                                         x0,
                                                         cast=lambda x: cast_leaves(x, float64))

        prob = dict(f=obj,
                    x=u_traj.symvec.astype(float64),
                    p=ravel_pytree(x0)[0].astype(float64))

        self.args = dict(x0=u_traj.x0.astype(float64),
                         lbx=u_traj.lb,
                         ubx=u_traj.ub)

        self.solver = solver('solver', 'ipopt', prob, ipopt_opts)

    def __call__(self, x0 = None, u0 = None):
        if x0: self.args['p'] = ravel_pytree(x0)[0].astype(float64)
        if u0: self.args['x0'] = u0.x0.astype(float64)
        soln = self.solver(**self.args)
        return self.u_traj_unflatten(soln['x'])
