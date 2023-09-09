import casadi as ca
import numpy as np

from robot import DynSys
from decision_vars import DecisionVarDict, NamedDict
"""
Single point contact stiffness model with parameters for
contact location in TCP, rest position, and stiffness.

This class is _stateless_, just building expressions.
Important variables:
  _pars: dict w/ numerical and symbolic parameters for building the expressions
  _state: DecisionVars for symbolic parameters and associated attributes
"""
class Contact(DynSys):
    """
    IN: name, a unique identifier which prefixes all vars/pars to ensure they're unique when passed up
    IN: pars, a dict with pos, stiff and rest, each a vector of length 3
    (IN): sym_vars, a list of strings for keys in pars which are to be made symbolic
    (IN): attrs, a nested dict of additional variable attributes {attr:{name:value}}
    """
    def __init__(self, name:str, pars:dict, sym_vars = [], attrs = {}):
        assert set(pars.keys()) == set(['pos', 'stiff', 'rest']), "Contact pars are [pos, stiff, rest]"
        self.name = name
        self._pars = NamedDict(name, {k:ca.DM(v) for k,v in pars.items()})     
        self.build_vars(sym_vars, name, attrs)

    
    def build_vars(self, sym_vars, name, attrs):
        self._state = DecisionVarDict(attr_names = list(attrs.keys()), name = name)
        if sym_vars is not []:
            init = {k: self._pars[k] for k in sym_vars}
            self._state.add_vars(init = init, **attrs)
            self._pars.update(self._state)

    """
    Return the derived state variables, evaluated at the numerical values 
    """
    def get_ext_state(self, num_dict):
        fn_input = {k:num_dict[k] for k in ['q', 'dq']+list(self._state.keys())}
        res = self.extended_state_fn(**fn_input)
        return {k:v for k, v in res.items()}

    """
    Build the contact forces and torques
    """
    def build_sys(self, q, dq, fwd_kin):
        p, R = fwd_kin(q)
        x = p + R@self._pars['pos']
        disp = x - self._pars['rest']
        n = self._pars['stiff']/ca.norm_2(self._pars['stiff'])
        j = ca.jacobian(n.T@x, q)
        dx = j@dq

        # Forces for visualization, no damping
        F = ca.times(self._pars['stiff'],(self._pars['rest']-x)) # Forces in world coord

        # Torques for dynamics w/ damping
        tau = j.T@(self._pars['stiff'].T@(self._pars['rest']-x))
        tau -= j.T@(ca.norm_2(self._pars['stiff'])*(0.02*dx))
        self.__F_fn = ca.Function('F', dict(q=q, dq=dq, F=F, **self._state),
                                ['q', 'dq', *self._state.keys()], ['F'])
        self.__tau_fn = ca.Function('tau', dict(q=q, dq=dq, tau=tau, **self._state),
                                    ['q', 'dq', *self._state.keys()], ['tau'])

        fn_dict = dict(q=q, dq=dq, **self._state)
        fn_output = NamedDict(self.name, {'x':x,'disp':disp,'n':n,'F':F, 'rest':self._pars['rest']})
        fn_dict.update(fn_output)
        self.extended_state_fn = ca.Function('statedict_fn', fn_dict,
                                             ['q', 'dq', *self._state.keys()],
                                             fn_output.keys())

    # Filter out unnecessary parameters and call the force fn
    def get_force(self, args):
        filtered_args = {k:v for k,v in args.items() if k in ['q', 'dq']+list(self._state.keys())}
        return self.__F_fn(**filtered_args)['F']

    def get_torque(self, args):
        filtered_args = {k:v for k,v in args.items() if k in ['q', 'dq']+list(self._state.keys())}
        return self.__tau_fn(**filtered_args)['tau']

