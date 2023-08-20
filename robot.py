import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin

from decision_vars import DecisionVarSet, ParamSet

from typing import Union
Vector = Union[ca.SX, ca.MX, ca.DM]
SymVector = Union[ca.SX, ca.MX]

class DynSys():
    """ Minimal, abstract class to standardize interfaces """
    def __init__(self):
        self._state = None
        self._input = None
        
    def get_ext_state(self, xi:Vector) -> dict:
        return NotImplementedError

    def get_dec_vars(self) -> DecisionVarSet:
        return self._state

    def get_input(self, H=1):
        return self._input.extend_vars(H)

    def get_state(self, H=1):
        return self._state.extend_vars(H)

    def get_statevec(self, H=1):
        return self._state.extend_vec(H)
    
class Robot(DynSys):
    """ This class handles the loading of robot dynamics/kinematics, the discretization/integration, and linearization.
        Rough design principles:
          This class is _stateless_, meaning
             - it produces only pure functions
             - actual state should be stored elsewhere
          The state dict _state is _minimal_, meaning
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

        self.__subsys = subsys   # subsystems which are coupled to the robot  
        self.__ctrl = ctrl       # controller which sets tau_input
        
        self.nq, self.fwd_kin, self.mass_fn = load_urdf(urdf_path, ee_frame_name)
        self.build_vars(attrs, visc_fric)
        self.build_fwd_kin(ee_frame_name)
    
    def build_vars(self, attrs, visc_fric):
        """ Build symbolic variables with attributes """
        self._state = DecisionVarSet(list(attrs.keys()))
        self._state.add_vars(init=dict( q=ca.DM.zeros(self.nq),
                                       dq=ca.DM.zeros(self.nq)), **attrs)
        
        self.visc_fric = visc_fric*ca.DM.eye(self.nq)
        if attrs and 'meas_noise' in attrs:
            self.meas_noise = ca.diag(ca.vertcat(attrs['meas_noise']['q']*ca.DM.ones(self.nq),
                                                 attrs['meas_noise']['tau_ext']*ca.DM.ones(self.nq)))
        for sys in self.__subsys:
            self._state += sys.get_dec_vars()
        self.nx = self._state.vectorize().shape[0]

        # Params for the step fn
        init = dict(M_inv = ca.DM.zeros(self.nq, self.nq))
        if not self.__ctrl: init['tau_input'] = ca.DM.zeros(self.nq)
        self._param = ParamSet(init = init)

        self._input = DecisionVarSet(['lb', 'ub'])
        
    def build_inv_mass(self, step_size):
        M = self.mass_fn(self._state['q']) + 0.5*ca.DM.eye(self.nq)
        inv_mass = ca.inv(M+step_size*self.visc_fric)   # Semi-implicit inverse of mass matrix
        self.inv_mass_fn = ca.Function('inv_mass', [self._state['q']], [inv_mass], ['q'], ['M_inv']).expand()
        
    def build_fwd_kin(self, ee_frame_name):
        """ This builds fwd kinematics, Jacobian matrix, and pseudoinv to the ee_frame_name
            as CasADi fns over _state['q'], _state['dq']
        """
        q   = self._state['q']
        dq  = self._state['dq']

        x_ee = self.fwd_kin(q) 
        J = ca.jacobian(x_ee[0], q) # Jacobian of only position, 3 x nq
        self.jac = ca.Function('jacobian', [q], [J], ['q'], ['Jac'])
        self.jacpinv = ca.Function('jac_pinv', [q], [ca.pinv(J.T)], ['q'], ['pinv'])
        self.tcp_motion = ca.Function('tcp_motion', [q, dq], [x_ee[0], J@dq], ['q', 'dq'], ['x', 'dx'])         
    
    def build_step(self, step_size, cost_fn = None):
        """ Build the dynamic update equation with mass as input
            IN: step_size, time step length in seconds
        """
        self.build_inv_mass(step_size)
        if self.__ctrl: self.build_ctrl()
        
        # Shorthand
        q = self._state['q']
        dq = self._state['dq']
        inp_args = self._state.get_attr('sym')
        inp_args.update(self._input.get_attr('sym'))
        inp_args.update(self._param.get_attr('sym'))
        
        self.tau_ext = self.jac(q).T@self.get_F_ext(q, dq)

        # Joint acceleration, then integrate
        ddq = inp_args['M_inv']@(-self.visc_fric@dq + self.tau_ext + self._input['tau_input'])
        dq_next = dq + step_size*ddq
        q_next = q + step_size*dq_next
        self.step = ca.Function('step', inp_args.values(), [q_next, dq_next],
                                        inp_args.keys(), ['q', 'dq']).expand()

        self._xi = self._state.vectorize()
        inp_args = self._input.get_attr('sym')
        inp_args.update(self._param.get_attr('sym'))

        self.step_vec = ca.Function('step_vec',
                                    [self._xi, *inp_args.values()],
                                    [ca.vertcat(q_next, dq_next, self._xi[2*self.nq:])],
                                    ['xi', *inp_args.keys()],
                                    ['xi'])
                                     
    def build_linearized(self):
        inp_args = self._input.get_attr('sym')
        inp_args['xi'] = self._xi
        res = self.step_vec.call(inp_args)
        xi_next = res['xi']
        y = ca.vertcat(self._state['q'], self.tau_ext)
        u = self._input['tau_input']
        A = ca.jacobian(xi_next, self._xi)
        C = ca.jacobian(y, self._xi)
        self.linearized = ca.Function('linearized', [self._xi, u, inp_args['M_inv'],], [A, C, xi_next, y],
                                                    ['xi', 'tau_input', 'M_inv'], ['A', 'C', 'xi_next', 'y']).expand()
        
    def get_ekf_info(self):
        proc_noise = ca.diag(self._state.vectorize(attr='proc_noise'))
        return self._xi, proc_noise, self.meas_noise

    def get_init(self):
        return self._state.get_vectors('init', 'cov_init')
    
    def build_ctrl(self):
        print(f'  Building controller {self.__ctrl.name}')
        q = self._state['q']
        dq = self._state['dq']
        arg_dict = self.__ctrl.get_dec_vars().get_attr('sym')
        arg_dict['p'], arg_dict['R'] = self.fwd_kin(q)
        arg_dict['dx'] = self.tcp_motion(q, dq)[1]

        self._input += self.__ctrl.get_dec_vars()
        self._input['tau_input'] = self.jac(q).T@self.__ctrl.get_force(arg_dict)
    
    # Returns the force on the TCP expressed in world coordinates
    def get_F_ext(self, q, dq):
        F_ext = ca.DM.zeros(3)
        arg_dict = {k:self._state[k] for k in self._state if k not in ['q', 'dq']}
        arg_dict['p'], arg_dict['R'] = self.fwd_kin(q)
        for sys in self.__subsys:
            F_ext += sys.get_force(arg_dict)
        return F_ext
        
    def get_ext_state_from_vec(self, xi):
        if type(xi) == ca.DM: xi = xi.full()
        d = self._state.dictize(xi)
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

