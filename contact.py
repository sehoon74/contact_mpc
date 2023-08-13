import casadi as ca
import numpy as np

from robot import DynSys
from decision_vars import DecisionVarSet
"""
Single point contact stiffness model with parameters for
contact location in TCP, rest position, and stiffness.

This class is _stateless_, just building expressions.
Important variables:
  __pars: dict w/ numerical and symbolic parameters for building the expressions
  __vars: DecisionVars for symbolic parameters and associated attributes
"""
class Contact(DynSys):
    """
    IN: pars, a dict with pos, stiff and rest, each a vector of length 3
    IN: sym_attrs, a nested dict {attr:{name:value}}
    """
    def __init__(self, pars, sym_vars = [], sym_attrs = {}):
        assert set(pars.keys()) == set(['pos', 'stiff', 'rest']), "Contact pars are [pos, stiff, rest]"
        self.__pars = {k:ca.DM(v) for k,v in pars.items()}        
        self.build_vars(sym_vars, sym_attrs)
        self.build_contact()

    def build_vars(self, sym_vars, sym_attrs):
        init = {k: self.__pars[k] for k in sym_vars}
        self.__vars = DecisionVarSet(init, **sym_attrs)
        self.__pars.update(self.__vars)

    def get_dec_vars(self):
        return self.__vars
    
    """
    Return the statedict variables evaluated at the numerical values 
    """
    def get_statedict(self, num_dict):
        fn_input = {k:num_dict[k] for k in self.__vars}
        return self.statedict_fn(fn_input)

    """
    Build the contact forces and torques
    """
    def build_contact(self):
        p = ca.SX.sym('p', 3)
        R = ca.SX.sym('R', 3, 3)
        x = p + R@self.__pars['pos']
        disp = x - self.__pars['pos']
        n = self.__pars['stiff']/ca.norm_2(self.__pars['stiff'])
        F = ca.times(self.__pars['stiff'],(self.__pars['rest']-x)) # Forces in world coord

        self.F_fn = ca.Function('F', dict(p=p, R=R, F=F, **self.__vars),
                                ['p', 'R', *self.__vars.keys()],
                                ['F'])
        
        fn_dict = dict(p=p, R=R, x=x, disp=disp, n=n, F=F, **self.__vars)
        self.statedict_fn = ca.Function('statedict_fn', fn_dict,
                                        ['p', 'R', *self.__vars.keys()],
                                        ['x', 'disp', 'n', 'F'])
        
        
