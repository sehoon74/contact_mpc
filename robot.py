import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin

from decision_vars import DecisionVarSet

from typing import Union
Vector = Union[ca.SX, ca.MX, ca.DM]
SymVector = Union[ca.SX, ca.MX]

class DynSys():
    """ Minimal, abstract class to standardize interfaces """
    def __init__(self):
        return NotImplementedError
    
    def get_ext_state(self, xi:Vector) -> dict:
        return NotImplementedError

    def get_dec_vars(self) -> DecisionVarSet:
        return NotImplementedError
    
    
class Robot(DynSys):
    """ This class handles the loading of robot dynamics/kinematics, the discretization/integration, and linearization.
        Rough design principles:
          This class is _stateless_, meaning
             - it produces only pure functions
             - actual state should be stored elsewhere
          The state dict __state is _minimal_, meaning
             - all derived quantities (for monitoring, debugging, etc) are calculated from _xi_ in ::get_statedict()
        Currently, it is assumed that all subsystems can be algebraically evaluated from _xi_, i.e. they are stateless
    """
    def __init__(self, urdf_path, attrs = {}, subsys = [], ctrl = None, ee_frame_name = 'fr3_link8', visc_fric = 30):
        """ IN: urdf_path is the location of the URDF file
            IN: ee_frame_name is the name in the urdf file for the end-effector
            IN: attrs for variables (e.g. initial value, lower/upper bounds, process noise)
            IN: subsys is a list of systems coupled to the robot
        """
        print(f"Building robot model from {urdf_path} with TCP {ee_frame_name}")
        if subsys: print(f"  with subsys {[s.name for s in subsys]}")

        self.__state = {}        # decision var set for _state_
        self.__subsys = subsys   # subsystems which are coupled to the robot  
        self.__ctrl = ctrl       # controller which sets tau_input
        
        self.nq, self.fwd_kin, self.mass_fn = load_urdf(urdf_path, ee_frame_name)
        self.build_vars(attrs, visc_fric)
        self.build_fwd_kin(ee_frame_name)
    
    def build_vars(self, attrs, visc_fric):
        """ Build symbolic variables with attributes """
        self.__state = DecisionVarSet(list(attrs.keys()))
        zero = ca.DM.zeros(self.nq)
        self.__state.add_vars(inits=dict(q=zero, dq=zero), **attrs)
        
        self.visc_fric = visc_fric*ca.DM.eye(self.nq)
        if attrs: self.meas_noise = ca.diag(ca.vertcat(attrs['meas_noise']['q']*ca.DM.ones(self.nq),
                                                       attrs['meas_noise']['tau_ext']*ca.DM.ones(self.nq)))
        for sys in self.__subsys:
            self.__state += sys.get_dec_vars()
        self.nx = self.__state.vectorize().shape[0]

        self.__input = DecisionVarSet(attr_names=['lb', 'ub'])
        if not self.__ctrl:
            self.__input.add_vars(inits = dict(tau_input = zero))
        else:
            self.__input.update(self.ctrl.get_dec_vars())

    def build_fwd_kin(self, ee_frame_name):
        """ This builds fwd kinematics, Jacobian matrix, and pseudoinv to the ee_frame_name
            as CasADi fns over __state['q'], __state['dq']
        """
        q   = self.__state['q']
        dq  = self.__state['dq']

        x_ee = self.fwd_kin(q) 
        J = ca.jacobian(x_ee[0], q) # Jacobian of only position, 3 x nq
        self.jac = ca.Function('jacobian', [q], [J], ['q'], ['Jac'])
        self.jacpinv = ca.Function('jac_pinv', [q], [ca.pinv(J.T)], ['q'], ['pinv'])
        self.tcp_motion = ca.Function('tcp_motion', [q, dq], [x_ee[0], J@dq], ['q', 'dq'], ['x', 'dx'])
        
    def build_disc_dyn(self, step_size):
        """ Build the dynamic update equation from __state, __input to {q_next, dq_next}
            IN: step_size, time step length in seconds
        """
        self.build_disc_dyn_core(step_size)
        if self.__ctrl: self.build_ctrl()

        inp_args = self.__state.get_attr('sym')
        M_inv = self.inv_mass_fn(inp_args['q'])
        res = self.disc_dyn_core(**inp_args,
                                 M_inv = M_inv,
                                 tau_input = self.__input['tau_input'])
        
        inp_args.update(self.__input.get_attr('sym'))
        return ca.Function('disc_dyn', [*inp_args.values()], [res['q_next'], res['dq_next']],
                                       [*inp_args.keys()], ['q_next', 'dq_next']).expand()

    
    def build_disc_dyn_core(self, step_size):
        """ Build the dynamic update equation with mass as input
            IN: step_size, time step length in seconds
        """
        # Shorthand
        q = self.__state['q']
        dq = self.__state['dq']
        inp_args = self.__state.get_attr('sym')

        tau_ext = self.jac(q).T@self.get_F_ext(q, dq)
        
        inp_args['tau_input'] = ca.SX.sym('tau_input', self.nq)
        
        M = self.mass_fn(q) + 0.5*ca.DM.eye(self.nq)
        inv_mass = ca.inv(M+step_size*self.visc_fric)   # Semi-implicit inverse of mass matrix
        self.inv_mass_fn = ca.Function('inv_mass', [q], [inv_mass], ['q'], ['M_inv']).expand()
        inp_args['M_inv'] = ca.SX.sym('M_inv', self.nq, self.nq)
        
        # Joint acceleration, then integrate
        ddq = inp_args['M_inv']@(-self.visc_fric@dq + tau_ext + inp_args['tau_input'])
        dq_next = dq + step_size*ddq
        q_next = q + step_size*dq_next
        self.disc_dyn_core = ca.Function('disc_dyn', [*inp_args.values()], [q_next, dq_next], inp_args.keys(), ['q_next', 'dq_next']).expand()

        self.__xi = self.__state.vectorize()
        xi_next = ca.vertcat(q_next, dq_next, self.__xi[2*self.nq:])
        self.__y = ca.vertcat(self.__state['q'], tau_ext)
        self.__u = inp_args['tau_input']
        A = ca.jacobian(xi_next, self.__xi)
        C = ca.jacobian(self.__y, self.__xi)
        self.linearized = ca.Function('linearized', [self.__xi, self.__u, inp_args['M_inv'],], [A, C, xi_next, self.__y],
                                                    ['xi', 'tau_input', 'M_inv'], ['A', 'C', 'xi_next', 'y']).expand()

    def get_ekf_info(self):
        proc_noise = ca.diag(self.__state.vectorize(attr='proc_noise'))
        return self.__xi, proc_noise, self.meas_noise

    def get_init(self):
        return self.__state.get_vectors('init', 'cov_init')

    def build_ctrl(self):
        arg_dict = self.__ctrl.get_dec_vars().get_attr('sym')
        arg_dict['p'], arg_dict['R'] = self.fwd_kin(q)
        arg_dict['dx'] = self.tcp_motion(q, dq)[1]
        self.__input['tau_input'] = self.jac(q).T@self.ctrl.get_force(arg_dict)
    
    # Returns the force on the TCP expressed in world coordinates
    def get_F_ext(self, q, dq):
        F_ext = ca.DM.zeros(3)
        arg_dict = {k:self.__state[k] for k in self.__state if k not in ['q', 'dq']}
        arg_dict['p'], arg_dict['R'] = self.fwd_kin(q)
        for sys in self.__subsys:
            F_ext += sys.get_force(arg_dict)
        return F_ext
        
    def get_ext_state_from_vec(self, xi):
        if type(xi) == ca.DM: xi = xi.full()
        d = self.__state.dictize(xi)
        return self.get_ext_state(d)
    
    def get_ext_state(self, d):
        """ Produce all values which are derived from state.
            IN: complete state as dict
        """
        d = {k:v for k, v in d.items()}
        d['p'], d['R'] = self.fwd_kin(d['q'])
        d['dx'] = self.tcp_motion(d['q'], d['dq'])[1]
        for sys in self.__subsys:
            d.update(sys.get_ext_state(d))
        return d

def load_urdf(urdf_path, ee_frame_name):
    model = pin.buildModelsFromUrdf(urdf_path, verbose = True)[0]
    __cmodel = cpin.Model(model)
    __cdata = __cmodel.createData()
    q = ca.SX.sym('q', model.nq)
    ee_ID = __cmodel.getFrameId(ee_frame_name)
        
    cpin.forwardKinematics(__cmodel, __cdata, q)
    cpin.updateFramePlacement(__cmodel, __cdata, ee_ID)
    ee = __cdata.oMf[ee_ID]
    fwd_kin = ca.Function('p',[q],[ee.translation, ee.rotation])
    mass_fn = ca.Function('M',[q],[cpin.crba(__cmodel, __cdata, q)])
    return model.nq, fwd_kin, mass_fn

