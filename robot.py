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

        self.__jit_options = {} # {'jit':True, 'compiler':'shell', "jit_options":{"compiler":"gcc", "flags": ["-Ofast"]}}
        self.__vars = {}        # decision var set
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

    def build_vars(self, attrs = {}):
        """ Build symbolic variables with attributes """
        self.__vars = DecisionVarSet(list(attrs.keys()))
        inits = dict(q=ca.DM.zeros(self.nq), dq=ca.DM.zeros(self.nq))
        self.__vars.add_vars(inits=inits, **attrs)

    def add_subsys(self):
        for sys in self.__subsys:
            self.__vars += sys.get_dec_vars()

    def add_imp_ctrl(self):
        # If we want these in __vars (e.g. we´ll provide them to the MPC problem)
        #inits = dict(imp_rest = ca.DM.zeros(3),
        #             imp_stiff = ca.DM.ones(3),)
        # somehow add inits to the sym variables?

        imp_stiff = ca.SX.sym('imp_stiff', N_p)
        imp_damp = 3*ca.sqrt(imp_stiff)
        imp_rest = ca.SX.sym('imp_rest', N_p)
        
        x, dx = self.get_tcp_motion(self.__vars['q'], self.__vars['dq'])
        F_imp = ca.diag(imp_damp) @ dx + ca.diag(imp_stiff) @ (imp_rest - x0)
        tau_imp = self.jac(self.__vars['q']).T@F_imp
        
            
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
        
    def build_disc_dyn(self, step_size):
        """ Build the variations of discrete dynamics
            IN: step_size, length in seconds
            IN: visc_fric, the viscious friction in joint space
        """
        self.build_disc_dyn_core(step_size)

        # Building with symbolic mass
        args = self.__vars.get_attr('sym')
        args['tau_input'] = ca.SX.sym('tau_input', self.nq)
        inp_keys = list(args.keys())
        args_core = {k:v for k,v in args.items()}
        args_core['M_inv'] = self.get_inv_mass(self.__vars['q'], step_size = step_size)
        args.update(self.disc_dyn_core.call(args_core))
        self.disc_dyn = ca.Function('disc_dyn', args,
                                    [*inp_keys],
                                    ['q_next', 'dq_next'], self.__jit_options).expand()
        #self.build_lin_dyn(Mtilde_inv, B)
        return self.disc_dyn
    
    def build_disc_dyn_core(self, step_size):
        """ Build the dynamic update equation
            IN: step_size, length in seconds
            IN: visc_fric, the viscious friction in joint space
        """
        # Shorthand
        nq = self.nq
        nq2 = 2*self.nq
        q = self.__vars['q']
        dq = self.__vars['dq']
        args = self.__vars.get_attr('sym')
        args['tau_input'] = ca.SX.sym('tau_input', nq) # any additional torques which will be applied
        args['M_inv'] = ca.SX.sym('M_inv', nq, nq)

        inp_keys = list(args.keys())
        
        # Gravitational torques
        tau_g = cpin.computeGeneralizedGravity(self.__cmodel, self.__cdata, q)

        # Torque residual: joint torques not compensated by inner controller
        #tau_err = -self.vars['tau_g'] # gravity not compensated (UR)
        tau_err = ca.DM.zeros(self.nq) # gravity compensated by inner controller (Franka)

        F_ext = self.get_F(q)
        tau_ext = self.jac(q).T@self.get_F(q)

        # Joint acceleration, then integrate
        ddq = args['M_inv']@(-self.visc_fric@dq + tau_err + tau_ext + args['tau_input'])
        args['dq_next'] = dq + step_size*ddq
        args['q_next'] = q + step_size*args['dq_next']

        self.disc_dyn_core = ca.Function('disc_dyn',
                                         args,
                                         inp_keys,
                                         ['q_next', 'dq_next'], self.__jit_options).expand()
        

    def build_lin_dyn(self, Mtilde_inv, B):
        tau_ext = self.__vars['tau_ext']
        q = self.__vars['q']
        ddelta_dq = Mtilde_inv@ca.jacobian(tau_ext, q) #ignoring derivative of Mtilde_inv wrt q, ~5x speedup
        ddelta_ddq = -Mtilde_inv@B
        ddelta_dp = Mtilde_inv@ca.jacobian(tau_ext, self.vars['xi'][nq2:]) #ignoring derivative of Mtilde_inv wrt q, ~5x speedup

        #A = ca.jacobian(self.vars['xi_next'], self.vars['xi']) # Old method which is slightly less efficient
        A = ca.SX.eye(nx)
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

    def get_statedict_from_vec(self, xi):
        if type(xi) == ca.DM: xi = xi.full()
        d = self.__vars.dictize(xi)
        return self.get_statedict(d)
    
    def get_statedict(self, d):
        """ Produce all values which are derived from state.
            IN: complete state in dictionary d (q, dq)
        """
        d['p'], d['R'] = self.fwd_kin(d['q'])
        for sys in self.__subsys:
            d.update(sys.get_statedict(d))
        return d

    def get_inv_mass(self, q, step_size):
        M = cpin.crba(self.__cmodel, self.__cdata, q)
        M += 0.5*ca.DM.eye(self.nq)
        inv_mass = ca.inv(M+step_size*self.visc_fric)   # Semi-implicit inverse of mass matrix
        self.inv_mass_fn = ca.Function('inv_mass', [q], [inv_mass]).expand()
        return inv_mass
        
    # Returns the force on the TCP expressed in world coordinates
    def get_F(self, q):
        F_ext = 0
        arg_dict = {k:self.__vars[k] for k in self.__vars if k not in ['q', 'dq']}
        arg_dict['p'], arg_dict['R'] = self.fwd_kin(q)
        for sys in self.__subsys:
            F_ext += sys.get_force(arg_dict)
        return F_ext
        
    def get_tcp_motion(self, q, dq):
        x = self.fwd_kin(q)
        dx = self.d_fwd_kin(q, dq)
        return x, dx

    def get_linearized(self, xi):
        return self.A(xi), self.C(xi)
