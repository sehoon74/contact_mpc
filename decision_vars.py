"""
Copyright (c) 2022, Kevin Haninger
Helper classes for decision variables in an optimization problem
"""

import casadi as ca
import numpy as np
from sys import version_info

"""
Helper class for sets of decision variables.

It's mostly a dict, and can reversably & repeatibly vectorize
  a variable which may have several attributes (e.g. lower bound).

The minimum is a dictionary of initial values, inits,
  which is used to construct symbolic vars of matching dimensions.

Additional attributes can be specified in **kwargs by giving dicts
  with either complete specification for each variable or a default
  in defaults.

"""
class DecisionVarSet(dict):
    """
    IN: inits, a dictionary with the name and initializing value of the symbolic variables
    IN: sym, the type of symbolic variable to create
    IN: attr_defaults, a dictionary of the default values to use for an attribute
    IN: **kwargs, additional attributes, where the value is a dict with keys for all keys in inits
    """
    def __init__(self,
                 inits,
                 sym = ca.SX.sym,
                 attr_defaults = dict(lb = -np.inf, ub = np.inf, cov = 0, noise = 0,),
                 **kwargs):
        super().__init__()
        assert version_info >= (3, 6), "Python 3.6 required to guarantee dicts are ordered"        
        
        self.__vars = {k:{} for k in ['sym', 'init']+list(kwargs.keys())}
            
        for name, init in inits.items():
            init = np.array(init)
            self.__vars['init'][name] = init
            self.__vars['sym'][name] = sym(name, *init.shape)
            for attr, vals in kwargs.items():
                assert set(vals) <= set(inits), f"All attribute {attr} keys must be in inits"
                assert attr in attr_defaults or name in vals, f"Attribute for {name} must have either defaults or specified in kwargs" 
                self.__vars[attr][name] = np.full(init.shape, vals.get(name, attr_defaults.get(attr)))

        for k in self.__vars['sym']:
            super().__setitem__(k, self.__vars['sym'][k])
                 
    def __setitem__(self, key, value):
        raise NotImplementedError

    def __delitem__(self, key):
        raise NotImplementedError

    def __add__(self, other):
        assert set(self.get_attr_list()) == set(other.get_attr_list()), "Dictionaries don't have matching attributes"
        for attr in self.get_attr_list():
            self.__vars[attr].update(other.get_attr(attr))
        for k in self.__vars['sym']:
            super().__setitem__(k, self.__vars['sym'][k])
        return self
    
    def __len__(self):
        return sum(val.size for val in self.__vars['init'].values())

    def __str__(self):
        s = "***** Decision Vars *****\n"
        s += f"Attributes: {self.get_attr_list()}\n"
        s += "Vars: \n"
        for key in self.__vars['sym']:
            s += f"  {key}: {self[key]}, {self[key].shape}\n"
        return s
    
    def vectorize(self, attr = 'sym'):
        return ca.vertcat(*[el.reshape((-1,1)) for el in self.__vars[attr].values()])

    def dictize(self, vec):
        """
        vec is the numerical optimization results, fills the dict x with reshaping as needed
        """
        assert len(vec) == len(self), "Length of optimization doesn't match initial x0"
        read_pos = 0
        d = {}
        for key, init in self.__vars['init'].items():
            v_size  = init.size
            v_shape = init.shape
            if len(v_shape) == 1: v_shape = (*v_shape,1)
            d[key] = np.squeeze(np.reshape(vec[read_pos:read_pos+v_size], v_shape))
            read_pos += v_size
        return d
    
    def get_vectors(self, *argv):
        return tuple([self.vectorize(arg) for arg in argv])

    def get_deviation(self, key):
        """
        Returns difference between initial value and symbolic (or numeric) value
        """
        return self.__vars['sym'][key]-self.__vars['init'][key]

    def get_attr_list(self):
        return list(self.__vars.keys())

    def get_attr(self, attr):
        return self.__vars[attr]
                                            
