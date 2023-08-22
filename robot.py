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

    def get_step_args(self):
        step_args = self._state.get_vars()
        if self._param: step_args.update(self._param.get_vars())
        if self._input: step_args.update(self._input.get_vars())
        return step_args

    def get_input(self, H = 1):
        return self._input.extend_vars(H)

    def get_state(self, H = 1):
        return self._state.extend_vars(H)
    
class Robot(DynSys):
    """ This class handles the loading of robot dynamics/kinematics, the discretization/integration, and linearization.
        Rough design principles:
          This class is _stateless_, meaning
             - it produces only pure functions
             - actual state should be stored elsewhere
          The state and input are divided into
             - _state[nx]: the *minimal* dynamic state of the system, with contributions from subsys
             - _param: any symbolic parameters which stay fixed over the planning horizon
             - _input[nu]: the input of the dynamic system
          The step function (i.e. discrete dynamics) is then called over
             step(**_state, **input, **param)
        Currently, it is assumed that all subsystems can be algebraically evaluated from _state, i.e. they are stateless
    """
    def __init__(self, urdf_path, attrs = {}, subsys = [], ctrl = None, ee_frame_name = 'fr3_link8', visc_fric = 30):
        """ IN: urdf_path is the location of the URDF file
            IN: attrs for variables (e.g. initial value, lower/upper bounds, process noise)
            IN: subsys is a list of systems coupled to the robot
            IN: ctrl is an optional controller which binds to tau_input and defines new _input
            IN: ee_frame_name is the name in the urdf file for the end-effector
        """
        print(f"Building robot model from {urdf_path} with TCP {ee_frame_name}")
        if subsys: print(f"  with subsys {[s.name for s in subsys]}")
        if ctrl: print(f"  with control {ctrl.name}")
        self.__subsys = subsys   # subsystems which are coupled to the robot  
        self.__ctrl = ctrl       # controller which sets tau_input
        
        self.nq, self.fwd_kin, self.mass_fn = load_urdf(urdf_path, ee_frame_name)
        self.build_vars(attrs, visc_fric)
        self.build_fwd_kin(ee_frame_name)
        self.add_subsys_and_ctrl()
    
    def build_vars(self, attrs, visc_fric):
        """ Build symbolic variables with attributes """
        self._state = DecisionVarSet(attr_names = list(attrs.keys()))
        self._state.add_vars(init=dict( q=ca.DM.zeros(self.nq),
                                       dq=ca.DM.zeros(self.nq)), **attrs)
        
        self.visc_fric = visc_fric*ca.DM.eye(self.nq)
        
        init = dict(M_inv = ca.DM.zeros(self.nq, self.nq))
        self._param = ParamSet(init = init)

    def add_subsys_and_ctrl(self):
        for sys in self.__subsys:
            self._state += sys._state
        self.nx = self._state.vectorize().shape[0]

        self._input = DecisionVarSet(['lb', 'ub'])
        if self.__ctrl:
            self._param += self.__ctrl._param
            self._input += self.__ctrl._input
            q = self._state['q']
            dq = self._state['dq']
            args_dict = self.get_step_args()
            args_dict = self.get_ext_state(args_dict) # add p, R, dx
            self._input['tau_input'] = self.jac(q).T@self.__ctrl.get_force(args_dict)
        else:
            self._input.add_vars({'tau_input':ca.DM.zeros(self.nq)})
        self.nu = self._input.vectorize().shape[0]
                
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

    def build_inv_mass(self, step_size):
        M = self.mass_fn(self._state['q']) + 0.5*ca.DM.eye(self.nq)
        inv_mass = ca.inv(M+step_size*self.visc_fric)   # Semi-implicit inverse of mass matrix
        self.inv_mass_fn = ca.Function('inv_mass', [self._state['q']], [inv_mass], ['q'], ['M_inv']).expand()

    def build_step(self, step_size, cost_fn = None):
        """ Build the dynamic update equation with mass as input
            IN: step_size, time step length in seconds
        """
        self.build_inv_mass(step_size)
        
        # Shorthand
        q = self._state['q']
        dq = self._state['dq']
        inp_args = self.get_step_args()
                
        self.tau_ext = self.jac(q).T@self.get_F_ext(q, dq)

        cost = cost_fn(inp_args) if cost_fn else 0
        
        # Joint acceleration, then integrate
        ddq = inp_args['M_inv']@(-self.visc_fric@dq + self.tau_ext + self._input['tau_input'])
        dq_next = dq + step_size*ddq
        q_next = q + step_size*dq_next

        # Build dictionary step function
        self.step = ca.Function('step', inp_args.values(), [q_next, dq_next, cost],
                                        inp_args.keys(), ['q', 'dq', 'cost']).expand()

        # Build vectorized step function
        self._xi = self._state.vectorize()
        self._u = self._input.vectorize()
        inp_args = self._param.get_vars()

        self.step_vec = ca.Function('step_vec',
                                    [self._xi, self._u, *inp_args.values()],
                                    [ca.vertcat(q_next, dq_next, self._xi[2*self.nq:]), cost],
                                    ['xi', 'u', *inp_args.keys()],
                                    ['xi', 'cost']).expand()

    def build_rollout(self, H, num_particles = 20):
        xi0 = ca.SX.sym('xi', self.nx)
        xi = xi0
        imp_rest  = ca.SX.sym('imp_rest', self.nu, H)
        imp_stiff = ca.SX.sym('imp_stiff', 3)
        M_inv =  self.inv_mass_fn(xi[:self.nq])
        step_ma = self.step_vec.mapaccum(H)
        xi_n, cost = step_ma(xi0, imp_rest, M_inv, imp_stiff)
        cost = ca.sum2(cost)
        self.rollout = ca.Function('rollout', [xi0, imp_rest, imp_stiff], [cost])
        self.rollout_map = self.rollout.map(num_particles)

    # Returns the force on the TCP expressed in world coordinates
    def get_F_ext(self, q, dq):
        F_ext = ca.DM.zeros(3)
        arg_dict = {k:self._state[k] for k in self._state if k not in ['q', 'dq']}
        arg_dict['p'], arg_dict['R'] = self.fwd_kin(q)
        for sys in self.__subsys:
            F_ext += sys.get_force(arg_dict)
        return F_ext

    def get_ext_state(self, st):
        """ Produce all values which are derived from state.
            IN: complete state as dict or vect
        """
        if type(st) == ca.DM: st = st.full()
        if type(st) is not dict: st = self._state.dictize(st)
        st = {k:v for k, v in st.items()}
        st['p'], st['R'] = self.fwd_kin(st['q'])
        st['dx'] = self.tcp_motion(st['q'], st['dq'])[1]
        for sys in self.__subsys:
            st.update(sys.get_ext_state(st))
        if self.__ctrl:
            st.update(self.__ctrl.get_ext_state(st))
        return st

class LinearizedRobot(Robot):
    def __init__(self, urdf_path, **kwargs):
        super().__init__(urdf_path, **kwargs)
        assert 'attrs' in kwargs, "Need attrs in kwargs for process and measurement noise"
        attrs = kwargs['attrs']
        assert ['meas_noise', 'proc_noise', 'cov_init'] <= list(attrs.keys()), f"Need measurement + process noise to support EKF, have {attrs.keys()}"

        self.proc_noise = ca.diag(self._state.vectorize(attr='proc_noise'))
        self.meas_noise = ca.diag(ca.vertcat(attrs['meas_noise']['q']*ca.DM.ones(self.nq),
                                             attrs['meas_noise']['tau_ext']*ca.DM.ones(self.nq)))

    def build_linearized(self):
        inp_args = self._param.get_vars()
        inp_args['u'] = self._u
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
        return self._xi, self.proc_noise, self.meas_noise

    def get_ekf_init(self):
        return self._state.get_vectors('init', 'cov_init')

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

