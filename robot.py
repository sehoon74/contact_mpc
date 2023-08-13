import pinocchio as pin
import pinocchio.casadi as cpin
import casadi as ca

from decision_vars import DecisionVarSet

from typing import Union
Vector = Union[ca.SX, ca.MX, ca.DM]
SymVector = Union[ca.SX, ca.MX]

class DynSys():
    """ Minimal, abstract class to standardize interfaces """
    def __init__(self):
        return NotImplementedError
    
    def get_statedict(self, xi:Vector) -> dict:
        return NotImplementedError

    def get_dec_vars(self) -> DecisionVarSet:
        return NotImplementedError
        
    def build_vars(self) -> Vector:
        return NotImplementedError
    
class Robot(DynSys):
    """ This class handles the loading of robot dynamics/kinematics, the discretization/integration, and linearization.
        Rough design principles:
          This class is _stateless_, meaning
             - produces only pure functions (same result for same input)
             - actual state should be stored elsewhere
          The state variable (xi) is _minimal_, meaning
             - all derived quantities (for monitoring, debugging, etc) are derived from state in ::get_statedict()
        Currently, it is assumed that all subsystems can be algebraically evaluated from _xi_, i.e. they are stateless
    """
    def __init__(self, urdf_path, ee_frame_name = 'fr3_link8', subsys = []):
        """ IN: urdf_path is the location of the URDF file
            IN: ee_frame_name is the name in the urdf file for the end-effector
            IN: subsys is a list of systems coupled to the robot
        """
        print(f"Building robot model from {urdf_path} with TCP {ee_frame_name}")
        if subsys: print(f"  with {len(subsys)} subsys")

        self.__jit_options = {} # {'jit':True, 'compiler':'shell', "jit_options":{"compiler":"gcc", "flags": ["-Ofast"]}}
        self.__vars = {}        # dictionary of state as symbolic variables
        self.__subsys = subsys  # subsystems which are coupled to the robot  
        
        self.load_pin_model(urdf_path)
        self.build_vars()
        self.build_fwd_kin(ee_frame_name)
        self.build_subsys()

    def get_statedict(self, xi):
        """ Produce all values which are derived from state.
            IN: complete state xi
        """
        d = {'q':xi[:self.nq],
             'dq':xi[self.nq:2*self.nq],
             'xi':xi}
        for sys in self.__subsys:
            d += sys.get_statedict(xi)
        return d

    def get_dec_vars(self):
        """ Gather all decision variables """
        sym = {'xi':self.__vars['xi']}
        for sys in self.__subsys:
            sym += sys.get_symdict()
        return sym

    def get_mass(self, q):
        return cpin.crba(self.__cmodel, self.__cdata, q)

    def load_pin_model(self, urdf_path):
        """ Load the Pinocchio model from the URDF file """
        self.model = pin.buildModelsFromUrdf(urdf_path, verbose = True)[0]
        self.data = self.model.createData()
        self.__cmodel = cpin.Model(self.model)
        self.__cdata = self.__cmodel.createData()
        self.nq = self.model.nq

    def build_vars(self):
        """ Build symbolic variable for this system and all subsys """
        self.__vars['q']  = ca.SX.sym('q', self.nq)     # Joint position
        self.__vars['dq'] = ca.SX.sym('dq', self.nq)    # Joint velocity
        self.__vars['xi'] = ca.vertcat(self.__vars['q'], self.__vars['dq'])
        self.nx = self.__vars['xi'].shape[0]

    def build_subsys(self):
        for sys in self.__subsys:
            new_state = sys.build_vars(self.__vars['xi'], self.fwd_kin)
            self.__vars['xi'] = ca.vertcat(self.__vars['xi'], new_state)
        self.nx = self.__vars['xi'].shape[0]

    def build_fwd_kin(self, ee_frame_name):
        """ This builds fwd kinematics, Jacobian matrix, and pseudoinv to the ee_frame_name
            as CasADi fns over __vars['q'], __vars['dq']
        """
        q   = self.__vars['q']
        dq  = self.__vars['dq']
        ddq = ca.SX.sym('ddq', self.nq) # Joint acceleration
        ee_ID = self.__cmodel.getFrameId(ee_frame_name)
        
        cpin.forwardKinematics(self.__cmodel, self.__cdata, q, dq, ddq)
        cpin.updateFramePlacement(self.__cmodel, self.__cdata, ee_ID)
        ee = self.__cdata.oMf[ee_ID]
        self.fwd_kin = ca.Function('p',[q],[ee.translation, ee.rotation])

        # x_ee is TCP pose as (pos, R), where pos is a 3-Vector and R a rotation matrix
        x_ee = self.fwd_kin(q) 
        J = ca.jacobian(x_ee[0], q) # Jacobian of only position, 3 x nq
        Jd = ca.jacobian(J.reshape((3*self.nq,1)), q)@dq # Vectorize b/c jacobian of matrix tricky
        Jd = Jd.reshape(J.shape)@dq # then reshape the result into the right shape
        
        self.jac = ca.Function('jacobian', [q], [J], ['q'], ['Jac'])
        self.jacpinv = ca.Function('jac_pinv', [q], [ca.pinv(J.T)], ['q'], ['pinv']) 
        self.tcp_motion = ca.Function('tcp', [q, dq], [x_ee[0], J@dq])
        
        # The Pinocchio jacobian is in TCP frame & doesn't have CasADi sparsity :(
        #jac_exp = cpin.computeFrameJacobian(self.__cmodel, self.__cdata, q, ee_ID)
        #self.jac_pin = ca.Function('jacobian_pin', [q], [jac_exp[:3,:]], ['q'], ['jac'])

        self.djac = ca.Function('dot_jacobian',  [q, dq], [Jd])
        
        self.d_fwd_kin = ca.Function('dx', [q, dq], [J@dq], ['q', 'dq'], ['dx'])

        
    def build_ext_torques(self, q):
        """ Build expression for external torques
            IN: q, a symbolic variable for joint poisitions
        """
        tau_ext = 0
        F_ext = 0
        
        for sys in self.__subsys:
            tau_ext += -sys.get_contact_torque(q)  # get model contact torque
            F_ext += -sys.get_contact_force(q)     # get model contact force in world coord

        self.__vars['tau_ext'] = tau_ext  # make contact torque an independent variable
        self.__vars['F_ext'] = F_ext      # make contact force an independent variable
        
    def build_disc_dyn(self, step_size, visc_fric = 30):
        """ Build the variations of discrete dynamics
            IN: step_size, length in seconds
            IN: visc_fric, the viscious friction in joint space
        """
        self.disc_dyn_core = self.build_disc_dyn_core(step_size, visc_fric)

        # Building with symbolic mass
        Msym = cpin.crba(self.__cmodel, self.__cdata, self.__vars['xi'][:self.nq])
        xi_next = self.disc_dyn_core(self.__vars['xi'], self.__vars['tau_input'], Msym)
        self.disc_dyn = ca.Function('disc_dyn',
                                    [self.__vars['xi'], self.__vars['tau_input']],
                                    [xi_next],
                                    ['xi', 'tau_input'],
                                    ['xi_next'], self.__jit_options).expand()
        #self.build_lin_dyn(Mtilde_inv, B)
        return self.disc_dyn
    
    def build_disc_dyn_core(self, step_size, visc_fric = 30):
        """ Build the dynamic update equation
            IN: step_size, length in seconds
            IN: visc_fric, the viscious friction in joint space
        """
        # Shorthand
        nq = self.nq
        nq2 = 2*self.nq
        q = self.__vars['q']
        dq = self.__vars['dq']
        self.__vars['tau_input'] = ca.SX.sym('tau_input', nq) # any additional torques which will be applied

        B = ca.diag([visc_fric]*nq) # joint damping matrix for numerical stability
        # Inverse of mass matrix, with adjustments for numerical staiblity
        M = ca.SX.sym('M', nq, nq)
        Mtilde = M+ca.diag(0.5*ca.DM.ones(self.nq))     # Adding motor inertia to mass matrix
        Mtilde_inv = ca.inv(M+step_size*B)   # Semi-implicit inverse of mass matrix
        # Have done benchmarking on using ca.solve(M, I, 'ldl') and cpin.computeMinverse
        # ca.solve: 12usec, ca.inv: 15usec, cpin: 22 usec. 

        # Gravitational torques
        self.__vars['tau_g'] = cpin.computeGeneralizedGravity(self.__cmodel, self.__cdata, q)

        # Torque residual: joint torques not compensated by inner controller
        #tau_err = -self.vars['tau_g'] # gravity not compensated (UR)
        tau_err = ca.DM.zeros(self.nq) # gravity compensated by inner controller (Franka)
        
        self.build_ext_torques(q)

        # Joint acceleration, then integrate
        ddq = Mtilde_inv@(-B@dq + tau_err + self.__vars['tau_ext'] + self.__vars['tau_input'])
        dq_next= dq + step_size*ddq
        q_next = q + step_size*dq_next
        xi_next = ca.vertcat(q_next, dq_next, self.__vars['xi'][nq2:])
        
        disc_dyn_core = ca.Function('disc_dyn', [self.__vars['xi'], self.__vars['tau_input'], M], [xi_next],
                                    ['xi', 'tau_input', 'mass matrix'],
                                    ['xi_next'], self.__jit_options).expand()
        return disc_dyn_core

    def build_lin_dyn(self, Mtilde_inv, B):
        tau_ext = self.__vars['tau_ext']
        q = self.__vars['q']
        ddelta_dq = Mtilde_inv@ca.jacobian(tau_ext, q) #ignoring derivative of Mtilde_inv wrt q, ~5x speedup
        ddelta_ddq = -Mtilde_inv@B
        ddelta_dp = Mtilde_inv@ca.jacobian(tau_ext, self.vars['xi'][nq2:]) #ignoring derivative of Mtilde_inv wrt q, ~5x speedup

        #A = ca.jacobian(self.vars['xi_next'], self.vars['xi']) # Old method which is slightly less efficient
        A = ca.SX.eye(self.nx)
        A[:nq, :nq]       += h*h*ddelta_dq
        A[:nq, nq:nq2]    += h*ca.SX.eye(nq)+h*h*ddelta_ddq
        A[:nq, nq2:]      += h*h*ddelta_dp
        A[nq:nq2, :nq]    += h*ddelta_dq
        A[nq:nq2, nq:nq2] += h*ddelta_ddq
        A[nq:nq2, nq2:]   += h*ddelta_dp
        dyn_fn_dict['A'] = A
        self.A = ca.Function('A', [xi, tau_in], [A],
                                  ['xi', 'tau_in'],['A'], self.jit_options).expand()
        C = ca.jacobian(self.__vars['y'], self.__vars['xi'])
        self.C = ca.Function('C', [xi, tau_in], [C],
                             ['xi', 'tau_in'], ['C'], self.jit_options).expand()

    def build_obs(self):
        self.__vars['y'] = ca.vertcat(self.__vars['q'], self.__vars['tau_ext'])
        self.obs = ca.Function('obs', [self.___vars['xi']], [y], ['xi'], ['y'])

    def get_tcp_motion(self, q, dq):
        x = self.fwd_kin(q)
        dx = self.d_fwd_kin(q, dq)
        return x, dx

    def get_linearized(self, xi):
        return self.A(xi), self.C(xi)
