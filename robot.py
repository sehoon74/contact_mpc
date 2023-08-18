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

class Robot(DynSys):
    """ This class handles the loading of robot dynamics/kinematics, the discretization/integration, and linearization.
        Rough design principles:
          This class is _stateless_, meaning
             - it produces only pure functions
             - actual state should be stored elsewhere
          The state variable (xi) is _minimal_, meaning
             - all derived quantities (for monitoring, debugging, etc) are calculated from _xi_ in ::get_statedict()
        Currently, it is assumed that all subsystems can be algebraically evaluated from _xi_, i.e. they are stateless
    """
    def __init__(self, urdf_path, ee_frame_name = 'fr3_link8', attrs = {}, subsys = [], visc_fric = 30):
        """ IN: urdf_path is the location of the URDF file
            IN: ee_frame_name is the name in the urdf file for the end-effector
            IN: attrs for the robot variables (e.g. initial value, lower/upper bounds)
            IN: subsys is a list of systems coupled to the robot
        """
        print(f"Building robot model from {urdf_path} with TCP {ee_frame_name}")
        if subsys: print(f"  with subsys {[s.name for s in subsys]}")

        self.__state = {}        # decision var set for _state_
        self.__subsys = subsys  # subsystems which are coupled to the robot  
        
        self.load_pin_model(urdf_path)
        self.visc_fric = visc_fric*ca.DM.eye(self.nq)
        self.build_vars(attrs)
        self.build_fwd_kin(ee_frame_name)
        self.add_subsys()

    def load_pin_model(self, urdf_path):
        """ Load the Pinocchio model from the URDF file """
        self.model = pin.buildModelsFromUrdf(urdf_path, verbose = True)[0]
        self.data = self.model.createData()
        self.__cmodel = cpin.Model(self.model)
        self.__cdata = self.__cmodel.createData()
        self.nq = self.model.nq

    def build_vars(self, attrs):
        """ Build symbolic variables with attributes """
        self.__state = DecisionVarSet(list(attrs.keys()))
        inits = dict(q=ca.DM.zeros(self.nq), dq=ca.DM.zeros(self.nq))
        self.__state.add_vars(inits=inits, **attrs)
        if attrs: self.meas_noise = ca.diag(ca.vertcat(attrs['meas_noise']['q']*ca.DM.ones(self.nq),
                                                       attrs['meas_noise']['tau_ext']*ca.DM.ones(self.nq)))
    
    def add_subsys(self):
        for sys in self.__subsys:
            self.__state += sys.get_dec_vars()
        self.nx = self.__state.vectorize().shape[0]

    def build_fwd_kin(self, ee_frame_name):
        """ This builds fwd kinematics, Jacobian matrix, and pseudoinv to the ee_frame_name
            as CasADi fns over __state['q'], __state['dq']
        """
        q   = self.__state['q']
        dq  = self.__state['dq']
        ddq = ca.SX.sym('ddq', self.nq) # Joint acceleration
        ee_ID = self.__cmodel.getFrameId(ee_frame_name)
        
        cpin.forwardKinematics(self.__cmodel, self.__cdata, q, dq, ddq)
        cpin.updateFramePlacement(self.__cmodel, self.__cdata, ee_ID)
        ee = self.__cdata.oMf[ee_ID]
        self.fwd_kin = ca.Function('p',[q],[ee.translation, ee.rotation])

        # x_ee is TCP pose as (pos, R), where pos is a 3-Vector and R a rotation matrix
        x_ee = self.fwd_kin(q) 
        J = ca.jacobian(x_ee[0], q) # Jacobian of only position, 3 x nq
        self.jac = ca.Function('jacobian', [q], [J], ['q'], ['Jac'])
        self.jacpinv = ca.Function('jac_pinv', [q], [ca.pinv(J.T)], ['q'], ['pinv'])
        self.tcp_motion = ca.Function('tcp_motion', [q, dq], [x_ee[0], J@dq], ['q', 'dq'], ['x', 'dx'])
        
    def build_disc_dyn(self, step_size):
        """ IN: step_size, time step length in seconds
        """
        self.build_disc_dyn_core(step_size)
        
        # Building with symbolic mass
        args = self.__state.get_attr('sym')
        args['tau_input'] = ca.SX.sym('tau_input', self.nq)
        inp_keys = list(args.keys())
        args_core = {k:v for k,v in args.items()}
        args_core['M_inv'] = self.inv_mass_fn(self.__state['q'])
        args.update(self.disc_dyn_core.call(args_core))
        return ca.Function('disc_dyn', args,
                           [*inp_keys],
                           ['q_next', 'dq_next']).expand()
    
    def build_disc_dyn_core(self, step_size):
        """ Build the dynamic update equation
            IN: step_size, length in seconds
        """
        # Shorthand
        nq = self.nq
        nq2 = 2*self.nq
        q = self.__state['q']
        dq = self.__state['dq']
        inp_args = self.__state.get_attr('sym')

        tau_ext = self.jac(q).T@self.get_F(q, dq)
        
        inp_args['tau_input'] = ca.SX.sym('tau_input', nq) # any additional torques which will be applied
        
        M = cpin.crba(self.__cmodel, self.__cdata, q) + 0.5*ca.DM.eye(self.nq)
        inv_mass = ca.inv(M+step_size*self.visc_fric)   # Semi-implicit inverse of mass matrix
        self.inv_mass_fn = ca.Function('inv_mass', [q], [inv_mass]).expand()
        inp_args['M_inv'] = ca.SX.sym('M_inv', nq, nq)
        
        # Joint acceleration, then integrate
        args = {k:v for k,v in inp_args.items()}
        ddq = args['M_inv']@(-self.visc_fric@dq + tau_ext + args['tau_input'])
        args['dq_next'] = dq + step_size*ddq
        args['q_next'] = q + step_size*args['dq_next']
        self.disc_dyn_core = ca.Function('disc_dyn', args, inp_args.keys(), ['q_next', 'dq_next']).expand()

        self.__xi = self.__state.vectorize()
        xi_next = ca.vertcat(args['q_next'], args['dq_next'], self.__xi[nq2:])
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
    
    # Returns the force on the TCP expressed in world coordinates
    def get_F(self, q, dq):
        F_ext = ca.DM.zeros(3)
        arg_dict = {k:self.__state[k] for k in self.__state if k not in ['q', 'dq']}
        arg_dict['p'], arg_dict['R'] = self.fwd_kin(q)
        arg_dict['dx'] = self.tcp_motion(q, dq)[1]
        for sys in self.__subsys:
            F_ext += sys.get_force(arg_dict)
        return F_ext
        
    def get_statedict_from_vec(self, xi):
        if type(xi) == ca.DM: xi = xi.full()
        d = self.__state.dictize(xi)
        return self.get_statedict(d)
    
    def get_statedict(self, d):
        """ Produce all values which are derived from state.
            IN: complete state in dictionary d
        """
        d = {k:v for k, v in d.items()}
        d['p'], d['R'] = self.fwd_kin(d['q'])
        d['dx'] = self.tcp_motion(d['q'], d['dq'])[1]
        for sys in self.__subsys:
            d.update(sys.get_statedict(d))
        return d

