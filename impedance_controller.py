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
        self.__pars = {k:ca.DM(v) for k,v in pars.items()}      
        self.build_vars(sym_vars, attrs)
        self.add_imp_ctrl()
 
    def build_vars(self, sym_vars, attrs):
        self.__vars = DecisionVarSet(attr_names = list(attrs.keys()))
        if sym_vars:
            inits = {k: self.__pars[k] for k in sym_vars}
            self.__vars.add_vars(inits = inits, **attrs)
            self.__pars.update(self.__vars)

    def add_imp_ctrl(self):
        p = ca.SX.sym('p', 3)
        R = ca.SX.sym('R', 3, 3)
        imp_stiff = self.__pars['imp_stiff']
        imp_damp = 3*ca.sqrt(imp_stiff)
        imp_rest = self.__pars['imp_rest']
        dx = ca.SX.sym('dx',3)
        
        F = ca.diag(imp_damp) @ dx + ca.diag(imp_stiff) @ (imp_rest - p)
        self.__F_fn = ca.Function('F', dict(p=p, R=R, dx=dx, F=F, **self.__vars),
                                  ['p', 'R', 'dx', *self.__vars.keys()], ['F'])

    def get_statedict(self, num_dict):
        fn_input = {k:num_dict[k] for k in ['p', 'R', 'dx']+list(self.__vars.keys())}
        return {'F_imp':self.__F_fn.call(fn_input)['F']}
    
    def get_dec_vars(self):
        return self.__vars
    
    # Filter out unnecessary parameters and call the force fn
    def get_force(self, args):
        filtered_args = {k:v for k,v in args.items() if k in ['p', 'R', 'dx']+list(self.__vars.keys())}
        return self.__F_fn(**filtered_args)['F']


    
