from dataclasses import dataclass

import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin
import numpy as np

from .sysvars import SysVarSet, SysVar, Array

nq = 7

@dataclass
class RobotParams:
    urdf_path: str = 'franka.urdf'
    ee_frame_name: str = 'fr3_link8'
    step_size: float = 1/50.

@dataclass
class CostParams:
    K: float = 1.
    
@dataclass
class TCPMotion:
    pos: Array
    rot: Array
    vel: Array
    jac: Array

class Robot:
    """ This class handles the loading of robot dynamics/kinematics, the discretization/integration, and linearization.
        Rough design principles:
          This class is _stateless_, meaning
             - it produces only pure functions
             - actual state should be stored elsewhere
          The state and input are divided into
             - _state[nx]: the *minimal* dynamic state of the system, with contributions from subsys
             - _input[nu]: the input of the dynamic system
             - _param: any symbolic parameters which stay fixed over the planning horizon
    """
    class State(SysVarSet):
        q: SysVar = np.zeros(nq)
        dq: SysVar = np.zeros(nq)

    class Input(SysVarSet):
        ddq: SysVar = np.zeros(nq)

    class MPCParams(State): # we inherit state because initial state x0 is an MPC param
        des_pose: SysVar = np.zeros(3)

    def __init__(self, rp: RobotParams):
        """ IN: robot_params with configuration variables"""
        self.fwd_kin_fn, self.jac_kin_fn = load_urdf(rp.urdf_path,
                                                     rp.ee_frame_name)

    def fwd_kin(self, state: State) -> TCPMotion:
        """Return the TCPMotion based on joint state."""
        x_ee = self.fwd_kin_fn(state.q)
        jac = self.jac_kin_fn(state.q[:,0])
        return TCPMotion(pos=x_ee[0],
                         rot=x_ee[1],
                         vel=jac@state.dq,
                         jac=jac)

    def _step(self, u: Input, x: State, rp: RobotParams) -> State:
        """Kinematic step function."""
        dq_next = x.dq + rp.step_size*u.ddq
        q_next = x.q + rp.step_size*dq_next
        return Robot.State.skip_sym(q=q_next, dq=dq_next)
    
    def _cost(self, u: Input, x: State, mp: MPCParams, cp: CostParams) -> float:
        tcp_motion = self.fwd_kin(x)
        return {'c':ca.sumsqr(cp.K@(mp.des_pose - tcp_motion.pos))}

def load_urdf(urdf_path, ee_frame_name):
    model = pin.buildModelsFromUrdf(urdf_path, verbose=True)[0]
    __cmodel = cpin.Model(model)
    __cdata = __cmodel.createData()

    q = ca.SX.sym('q', model.nq)
    ee_ID = __cmodel.getFrameId(ee_frame_name)

    cpin.forwardKinematics(__cmodel, __cdata, q)
    cpin.updateFramePlacement(__cmodel, __cdata, ee_ID)
    ee = __cdata.oMf[ee_ID]
    
    fwd_kin = ca.Function('p', [q], [ee.translation, ee.rotation])
    jac_kin = ca.Function('J', [q], [ca.jacobian(ee.translation, q)])
    return fwd_kin, jac_kin


if __name__ == '__main__':
    rp = RobotParams()
    r = Robot(rp)
    st = Robot.State.skip_sym(q=np.ones(nq), dq=np.zeros(nq))
    print(st)
    print(r.fwd_kin(st))

    u=Robot.Input.skip_sym(ddq=np.ones(nq))
    
    next_st = r._step(u=u,
                      x=st,
                      rp=rp)
    print(next_st)
    cost = r._cost(u=u, x=st, cp=CostParams(), mp=Robot.MPCParams.skip_sym())
    print(cost)
