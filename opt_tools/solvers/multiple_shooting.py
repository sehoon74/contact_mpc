"""Multiple shooting solver, currently only tested w/ casadi_tools."""
import casadi as ca
# from casadi_tools.human import Human, CostParams, StepParams
from casadi_tools.robot import Robot, CostParams, RobotParams
from casadi import nlpsol
import numpy as np


class MultipleShootingSolver:
    def __init__(self, sys, solver, step_params, cost_params, H: int):
        """Builds a multiple-shooting MPC problem."""

        # Build decision variables
        mp = sys.MPCParams()  # get symbolic mpc params
        x = sys.State.generate_traj(horizon=H)
        u = sys.Input.generate_traj(horizon=H)
        self.vars = x + u

        # Evaluate self._step over the trajectory
        # x_next = sys._step(u=u, x=x, step_params=step_params)
        x_next = sys._step(u=u, x=x, rp=step_params)

        # Continuity constraints
        g = x.continuity_constraint(x0=mp, x_next=x_next)

        # Objective function
        # J = sys._cost(u=u, x=x, mpc_params=mp, cost_params=cost_params)['c']
        J = sys._cost(u=u, x=x, mp=mp, cp=cost_params)['c']

        # Set up dictionary of arguments to solve
        bnd_g = np.zeros(g.shape)
        self.args = dict(x0=self.vars.x0,
                         lbx=self.vars.lb,
                         ubx=self.vars.ub,
                         lbg=bnd_g,
                         ubg=bnd_g)
        self.prob = dict(f=J,
                         x=self.vars.symvec,
                         g=g,
                         p=mp.symvec)
        self.solver = solver('solver', 'ipopt', self.prob, {})

    def __call__(self, mpc_params):
        # get the numerical vector from mpc_params
        self.args['p'] = mpc_params.x0
        sol = self.solver(**self.args)
        # print(self.solver.stats())
        res = self.vars.from_vec(sol['x'])
        return res

# '''human test'''
# cp = CostParams()
# sp = StepParams()
# hum = Human()

# MSS = MultipleShootingSolver(hum, nlpsol, sp, cp, H=15)
# print(MSS(hum.MPCParams(goal=ca.DM([0.3, 0, 0]))))

'''robot test'''
cp = CostParams()
rp = RobotParams()
rob = Robot(rp)

MSS = MultipleShootingSolver(rob, nlpsol, step_params=rp, cost_params=cp, H=15)
print(MSS(rob.MPCParams(des_pose=ca.DM([0.3, 0, 0]))))
