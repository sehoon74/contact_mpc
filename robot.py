import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin

from decision_vars import DecisionVarSet, ParamSet

class DynSys():
    """ Minimal, abstract class to standardize interfaces """
    def __init__(self):
        self._state = None # Dynamic state of the system, including all subsys, is updated by step
        self._input = None # Symbolic input variables for the step function
        self._param = None # Symbolic parameters, including all subsys, which are not optimized but needed for step

    def get_step_args(self) -> dict:
        """ Returns all symbolic arguments used for the step function in a dict """
        step_args = self._state.get_vars()
        if self._param: step_args.update(self._param.get_vars())
        if self._input: step_args.update(self._input.get_vars())
        return step_args

    def get_input(self, H = 1) -> DecisionVarSet:
        """ Returns the input variables, repeated along last dim by planning horizon H
             _input is typically determined by the __ctrl object """
        return self._input.extend_vars(H)

    def get_state(self, H = 1) -> DecisionVarSet:
        """ Returns the dynamic state, repeated along last dim by planning horizon H """
        return self._state.extend_vars(H = H)

    
class Robot(DynSys):
    """ This class handles the loading of robot dynamics/kinematics, the discretization/integration, and linearization.
        Rough design principles:
          This class is _stateless_, meaning
             - it produces only pure functions
             - actual state should be stored elsewhere
          The state and input are divided into
             - _state[nx]: the *minimal* dynamic state of the system, with contributions from subsys
             - _input[nu]: the input of the dynamic system
             - _param: any symbolic parameters which stay fixed over the planning horizon
          Both _state and _input are handled as dicts and vectors (_xi, _u respectively)
          The step function (i.e. discrete dynamics) is then called over
             step(**_state, **input, **param) or step_vec(_xi, _u, **param)
        Currently, it is assumed that all subsystems can be algebraically evaluated from _state

    """
    def __init__(self, urdf_path:str, ee_frame_name = 'fr3_link8', visc_fric = 30,
                 attrs = {}, subsys = [], ctrl = None, name = ''):
        """ IN: urdf_path, path to URDF file
            IN: attrs for variables (e.g. initial value, lower/upper bounds, process noise)
            IN: subsys is a list of systems coupled to the robot
            IN: ctrl is an optional controller which binds to tau_input and defines new _input variables
            IN: ee_frame_name is the name in the urdf file for the end-effector
        """
        print(f"Building robot {name} from {urdf_path} with TCP {ee_frame_name}")
        if subsys: print(f"  with subsys {[s.name for s in subsys]}")
        if ctrl:   print(f"  with control {ctrl.name}")
        self.__subsys = subsys   # subsystems which are coupled to the robot  
        self.__ctrl = ctrl       # controller which sets tau_input
        
        self.nq, self.fwd_kin, self.mass_fn = load_urdf(urdf_path, ee_frame_name)
        
        self.build_vars(attrs, visc_fric, name)
        self.build_fwd_kin()
        self.add_subsys_and_ctrl()
    
    def build_vars(self, attrs:dict, visc_fric:float, name:str):
        """ Build symbolic variables with attributes """
        self._state = DecisionVarSet(name = name, attr_names = list(attrs.keys()))
        self._state.add_vars(init=dict( q=ca.DM.zeros(self.nq),
                                       dq=ca.DM.zeros(self.nq)), **attrs)
        
        self.visc_fric = visc_fric*ca.DM.eye(self.nq)

        # Inverse mass is a parameter to step to allow us to calculate M_inv once and use multiple times
        self._param = ParamSet(dict(M_inv = ca.DM.zeros(self.nq, self.nq)))
                
    def build_fwd_kin(self):
        """ This builds fwd kinematics, Jacobian matrix, and pseudoinv to the ee_frame_name
            as CasADi fns over _state['q'], _state['dq'] """
        q   = self._state['q']
        dq  = self._state['dq']

        x_ee = self.fwd_kin(q)      # (Pose, orientation) of ee_frame_name in world coordinates
        J = ca.jacobian(x_ee[0], q) # Jacobian of only position, 3 x nq

        self.jac = ca.Function('jacobian', [q], [J], ['q'], ['Jac'])
        self.jacpinv = ca.Function('jacpinv', [q], [ca.pinv(J.T)], ['q'], ['pinv'])
        self.tcp_motion = ca.Function('tcp_motion', [q, dq], [x_ee[0], J@dq], ['q', 'dq'], ['x', 'dx'])                 

    def add_subsys_and_ctrl(self):
        """ Add the variables from subsystems and control to the _state and _input sets """
        for sys in self.__subsys:
            self._state += sys._state      # Subsys is added to state so it gets vectorized cleanly
        self._xi_set = self._state.vectorize_set('xi') # Symbolic variable for state as column vector
        self._xi = self._xi_set['xi']
        self.nx = self._xi.shape[0]

        self._input = DecisionVarSet(['lb', 'ub']) # Your input has lower / upper bounds, right? 
        if self.__ctrl:
            self._param += self.__ctrl._param
            self._input += self.__ctrl._input
            args_dict = self.get_step_args()
            args_dict = self.get_ext_state(args_dict) # add p, R, dx
            self._input['tau_input'] = self.jac(self._state['q']).T@self.__ctrl.get_force(args_dict)
        else:
            self._input.add_vars({'tau_input':ca.DM.zeros(self.nq)})
        self._u_set = self._input.vectorize_set('u')
        self._u = self._u_set['u']
        self.nu = self._u.shape[0]

    def build_inv_mass(self, step_size:float):
        """ Build a function for the inverse mass so it's easily accessed
            IN: step_size, time step length in seconds """
        M = self.mass_fn(self._state['q']) + 0.5*ca.DM.eye(self.nq)
        inv_mass = ca.inv(M+step_size*self.visc_fric)   # Semi-implicit inverse of mass matrix
        self.inv_mass_fn = ca.Function('inv_mass', [self._state['q']], [inv_mass], ['q'], ['M_inv']).expand()

    def build_step(self, step_size, cost_fn = None):
        """ Build the dynamic update equation with mass as input
            IN: step_size, time step length in seconds """
        self.build_inv_mass(step_size)
        
        # Shorthand
        q = self._state['q']
        dq = self._state['dq']
        inp_args = self.get_step_args()
                
        self.tau_ext = self.jac(q).T@self.get_F_ext(q, dq)  # External torques from the __subsys

        cost = cost_fn.call(inp_args)['cost'] if cost_fn else 0          # Cost for the current step
        
        # Joint acceleration, then integrate
        ddq = inp_args['M_inv']@(-self.visc_fric@dq + self.tau_ext + self._input['tau_input'])
        dq_next = dq + step_size*ddq
        q_next = q + step_size*dq_next
        
        # Build dictionary step function
        self.step = ca.Function('step', inp_args.values(), [q_next, dq_next, cost],
                                        inp_args.keys(), ['q', 'dq', 'cost']).expand()

        # Build vectorized step function
        param_args = self._param.get_vars()
        xi_next = ca.vertcat(q_next, dq_next, self._xi[2*self.nq:])
        self.step_vec = ca.Function('step_vec',
                                    [self._xi, self._u, *param_args.values()],
                                    [xi_next, cost],
                                    ['xi', 'u', *param_args.keys()],
                                    ['xi', 'cost']).expand()

    def build_rollout(self, H, num_particles = 20):
        """ Builds a rollout from an initial state size x0[nx] and input trajectory size u_traj = [nu, H, num_particles]
            to a total cost """
        xi0 = ca.SX.sym('xi', self.nx)
        inp = self.get_input(H = H)
        u = self._u_set.extend_vars(H = H).get_vars()['u']
        par = self._param.get_vars()

        step_ma = self.step_vec.mapaccum(H)

        res = step_ma(xi=xi0, u=u, **par)
        cost = ca.sum2(res['cost'])
        self.rollout = ca.Function('rollout', [xi0, u, *par.values()], [cost],
                                              ['xi0', 'u_traj', *par.keys()], ['cost'])
        self.rollout_map = self.rollout.map(num_particles)
        self.rollout_xi = ca.Function('rollout', [xi0, u, *par.values()], [res['xi']],
                                              ['xi0', 'u_traj', *par.keys()], ['xi'])
    def get_F_ext(self, q, dq):
        F_ext = ca.DM.zeros(3)
        arg_dict = {k:self._state[k] for k in self._state if k not in ['q', 'dq']}
        arg_dict['p'], arg_dict['R'] = self.fwd_kin(q)
        for sys in self.__subsys:
            F_ext += sys.get_force(arg_dict)
        return F_ext

    def get_statedict(self, vec):
        return self._state.dictize(vec)
    
    def get_ext_state(self, st):
        """ Produce all values which are derived from state.
            IN: complete state as dict or vec
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
        for k,v in st.items():
            if type(v) == ca.DM: st[k] = v.full()
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

    def build_step(self, step_size, cost_fn = None):
        super().build_step(step_size, cost_fn = cost_fn)
        self.build_linearized()
        
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
