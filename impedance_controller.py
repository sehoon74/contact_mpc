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

    def add_imp_ctrl(self):
        imp_stiff = ca.SX.sym('imp_stiff', N_p)
        imp_damp = 3*ca.sqrt(imp_stiff)
        imp_rest = ca.SX.sym('imp_rest', N_p)
        
        x, dx = self.get_tcp_motion(self.__vars['q'], self.__vars['dq'])
        F_imp = ca.diag(imp_damp) @ dx + ca.diag(imp_stiff) @ (imp_rest - x0)
        tau_imp = self.jac(self.__vars['q']).T@F_imp
        

class ImpedanceController(DynSys):
    """
    IN: name, a unique identifier which prefixes all vars/pars to ensure they're unique when passed up
    IN: pars, a dict with pos, stiff and rest, each a vector of length 3
    (IN): sym_vars, a list of strings for keys in pars which are to be made symbolic
    (IN): attrs, a nested dict of additional variable attributes {attr:{name:value}}
    """
    def __init__(self, name:str, pars:dict, sym_vars = [], attrs = {}):
        assert set(pars.keys()) == set(['stiff', 'rest']), "Impedance pars are [stiff, rest]"
        self.name = name
        self.__pars = {self.ns(k):ca.DM(v) for k,v in pars.items()}        
        self.build_vars(sym_vars, attrs)
        self.build_contact()

    def ns(self, s):
        return self.name+'/'+s     
    
    def build_vars(self, sym_vars, attrs):
        ns_attrs = {name:{self.ns(k):v for k,v in attr.items()} for name, attr in attrs.items()}
        self.__vars = DecisionVarSet(attrs = list(ns_attrs.keys()))
        if sym_vars:
            inits = {self.ns(k): self.__pars[self.ns(k)] for k in sym_vars}
            self.__vars.add_vars(inits = inits, **ns_attrs)
            self.__pars.update(self.__vars)

    def get_dec_vars(self):
        return self.__vars
    
    """
    Return the statedict variables evaluated at the numerical values 
    """
    def get_statedict(self, num_dict):
        fn_input = {k:num_dict[k] for k in ['p', 'R']+list(self.__vars.keys())}
        res = self.statedict_fn(**fn_input)
        return {self.ns(k):v for k, v in res.items()}

    """
    Build the contact forces and torques
    """    
    def build_contact(self):
        p = ca.SX.sym('p', 3)
        R = ca.SX.sym('R', 3, 3)
        x = p + R@self.__pars[self.ns('pos')]
        disp = x - self.__pars[self.ns('pos')]
        n = self.__pars[self.ns('stiff')]/ca.norm_2(self.__pars[self.ns('stiff')])
        F = ca.times(self.__pars[self.ns('stiff')],(self.__pars[self.ns('rest')]-x)) # Forces in world coord

        self.__F_fn = ca.Function('F', dict(p=p, R=R, F=F, **self.__vars),
                                ['p', 'R', *self.__vars.keys()],
                                ['F'])
        
        fn_dict = dict(p=p, R=R, x=x, disp=disp, n=n, F=F, **self.__vars)
        self.statedict_fn = ca.Function('statedict_fn', fn_dict,
                                        ['p', 'R', *self.__vars.keys()],
                                        ['x', 'disp', 'n', 'F'])

    # Filter out unnecessary parameters and call the force fn
    def get_force(self, args):
        filtered_args = {k:v for k,v in args.items() if k in ['p', 'R']+list(self.__vars.keys())}
        return self.__F_fn(**filtered_args)['F']


    
