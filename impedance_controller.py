import casadi as ca
import numpy as np

from robot import DynSys
from decision_vars import DecisionVarSet
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
    (IN): sym_vars, a list of strings for keys in pars which are to be made symbolic
    (IN): attrs, a nested dict of additional variable attributes {attr:{name:value}}
    """
    def __init__(self, pars:dict, sym_vars = [], attrs = {}):
        assert set(pars.keys()) == set(['imp_stiff', 'imp_rest']), "Impedance pars are [stiff, rest]"
        self.name = 'imp_ctrl'
        self._pars = {k:ca.DM(v) for k,v in pars.items()}      
        self.build_vars(sym_vars, attrs)
        self.build_force()
 
    def build_vars(self, sym_vars, attrs):
        self._state = DecisionVarSet(attr_names = ['lb', 'ub'])
        if sym_vars:
            init = {k: self._pars[k] for k in sym_vars}
            self._state.add_vars(init = init, **attrs)
            self._pars.update(self._state)

    def build_force(self):
        p = ca.SX.sym('p', 3)
        R = ca.SX.sym('R', 3, 3)
        imp_stiff = self._pars['imp_stiff']
        imp_damp = 3*ca.sqrt(imp_stiff)
        imp_rest = self._pars['imp_rest']
        dx = ca.SX.sym('dx',3)
        
        F = ca.diag(imp_damp) @ dx + ca.diag(imp_stiff) @ (imp_rest - p)
        self.__F_fn = ca.Function('F', dict(p=p, R=R, dx=dx, F=F, **self._state),
                                  ['p', 'R', 'dx', *self._state.keys()], ['F'])
            
    def get_ext_state(self, d):
        fn_input = {k:d[k] for k in ['p', 'R', 'dx']+list(self._state.keys())}
        return {'F_imp':self.__F_fn.call(fn_input)['F']}
    
    # Filter out unnecessary parameters and call the force fn
    def get_force(self, args):
        filtered_args = {k:v for k,v in args.items() if k in ['p', 'R', 'dx']+list(self._state.keys())}
        return self.__F_fn(**filtered_args)['F']


    
