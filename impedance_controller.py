import casadi as ca
import numpy as np

from robot import DynSys
from decision_vars import DecisionVarSet, ParamSet
"""
Impedance model with rest position, stiffness, etc

This class is _stateless_, just building expressions.
Important variables:
  __pars: dict w/ numerical and symbolic parameters for building the expressions
  __vars: DecisionVars for symbolic parameters and associated attributes
"""

class ImpedanceController(DynSys):
    """
    IN: pars, a dict with initial stiffness and rest
    (IN): input_vars, a list of strings for keys in pars which are to be made symbolic
    (IN): attrs, a nested dict of additional variable attributes {attr:{name:value}}
    """
    def __init__(self, input_vars = [], attrs = {}):
        self.name = 'imp_ctrl'
        self.build_vars(input_vars, attrs)
        self.build_force()
 
    def build_vars(self, input_vars, attrs):
        self._input = DecisionVarSet(attr_names = ['lb', 'ub'])
        self._imp_pars = {} # used in impedance calc
        imp_init = {k:ca.DM.zeros(3) for k in ['imp_stiff', 'imp_rest']} 
        if input_vars:
            init = {var: imp_init.pop(var) for var in input_vars}
            self._input.add_vars(init = init, **attrs)
            self._imp_pars.update(self._input)
        if imp_init: # if there's anything left...
            self._param = ParamSet(imp_init)
            self._imp_pars.update(self._param)

    def build_force(self):
        p = ca.SX.sym('p', 3)
        R = ca.SX.sym('R', 3, 3)
        imp_stiff = self._imp_pars['imp_stiff']
        imp_damp = 3*ca.sqrt(imp_stiff)
        imp_rest = self._imp_pars['imp_rest']
        dx = ca.SX.sym('dx',3)

        # Builds force assuming impedance parameters are in cartesian space, as is the case with the Franka Emika
        F = ca.diag(imp_damp) @ dx + ca.diag(imp_stiff) @ (imp_rest)
        self.__F_fn = ca.Function('F', dict(p=p, R=R, dx=dx, F=F, **self._input, **self._param),
                                  ['p', 'R', 'dx', *self._input.keys(), *self._param.keys()], ['F'])
        
    def get_ext_state(self, d):
        return {'F_imp':self.get_force(d)}
    
    # Filter out unnecessary parameters and call the force fn
    def get_force(self, args):
        arg_names = ['p', 'R', 'dx', 'imp_stiff', 'imp_rest']
        assert set(arg_names) <= set(args.keys()), f"Need {arg_names} in args, have {args.keys()}" 
        filtered_args = {k:v for k,v in args.items() if k in arg_names}
        return self.__F_fn(**filtered_args)['F']
