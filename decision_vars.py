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
    IN: attr_names, a list of strings for the attributes
    IN: name, a prefix for variables to prename
    IN: attr_defaults, a dictionary of the default values to use for an attribute
    """
    def __init__(self, attr_names, name = '', sym = ca.SX.sym,
                 attr_defaults = dict(lb = -np.inf,
                                      ub =  np.inf,
                                      cov_init = 0,
                                      meas_noise = 0,
                                      proc_noise = 0)):
        super().__init__()
        assert version_info >= (3, 6), "Python 3.6 required to guarantee dicts are ordered"                
        self.name = name
        self.attr_names = attr_names
        self.attr_defaults = attr_defaults
        self.sym = sym
        self.__vars = {k:NamedDict(name) for k in ['sym', 'init']+attr_names}

    def add_vars(self, init = {}, **kwargs):
        assert init or ('init' in kwargs), f"Need inits for your variables! Have attrs {list(kwargs.keys())}"
        if not init:
            init = kwargs.pop('init')
        for name, ini in init.items():
            ini = np.array(ini)
            self.__vars['init'][name] = ini
            self.__vars['sym'][name] = self.sym(name, *ini.shape)

            # Add the additional attributes for each variable
            for attr in self.attr_names:
                assert attr in self.attr_defaults or name in kwargs.get(attr), f"Attribute for {name} must have either defaults or specified in kwargs"
                val = kwargs[attr][name] if name in kwargs.get(attr, []) else self.attr_defaults[attr]
                self.__vars[attr][name] = np.full(ini.shape, val)

        for k, v in self.__vars['sym'].items(): self[k] = v
    
    def __setitem__(self, key, value):
        super().__setitem__(key, value)

    def __delitem__(self, key):
        raise NotImplementedError

    """
    Adds other variables to self, the attributes are projected down to those in self
    """
    def __add__(self, other):
        assert set(self.attr_names) <= set(other.attr_names), "LHS set has attributes that RHS doesn't"
        for attr in list(self.attr_names)+['init', 'sym']:
            self.__vars[attr].update(other.get_vars(attr))
        for k in self.__vars['sym']:
            super().__setitem__(k, self.__vars['sym'].get(k))
        return self
    
    def __len__(self):
        return sum(val.size for val in self.__vars['init'].values())

    def __str__(self):
        s = "***** Decision Vars *****\n"
        s += f"Attributes: {self.attr_names}\n"
        s += "Vars: \n"
        for key in self.__vars['sym']:
            s += f"  {key}: {self[key]}, shape: {self[key].shape}\n"
        return s
    
    def vectorize(self, attr = 'sym', d = {}):
        if not d:
            d = self.__vars[attr]
        for k in d.keys():
            if type(d.get(k)) == float: d[k] = ca.DM(d[k])
        return ca.vertcat(*[d.get(k).reshape((-1,1)) for k in self.__vars['init'].keys()])

    def dictize(self, vec):
        """
        vec is the numerical optimization results, fills the dict x with reshaping as needed
        """
        #assert len(vec) == len(self), "Length of optimization doesn't match initial x0"
        read_pos = 0
        d = {}
        for key, init in self.__vars['init'].items():
            v_size  = init.size
            v_shape = init.shape
            if len(v_shape) == 0: v_shape = (1, 1)
            if len(v_shape) == 1: v_shape = (*v_shape,1)
            d[key] = ca.reshape(vec[read_pos:read_pos+v_size], v_shape)
            read_pos += v_size
        return d
    
    def get_vectors(self, *argv):
        return tuple([self.vectorize(arg) for arg in argv])
    
    def get_vars(self, attr = 'sym'):
        return {k:self.__vars[attr].get(k) for k in self.__vars[attr].keys()}
    
    def extend_vars(self, H):
        attrs = {attr:{k:ca.repmat(self.__vars[attr][k],1,H) for k in self.__vars[attr].keys()} for attr in self.attr_names+['init']}
        dec_vars = DecisionVarSet(attr_names=self.attr_names,
                                  name=self.name,
                                  attr_defaults=self.attr_defaults)
        dec_vars.add_vars(**attrs)
        return dec_vars
                 
class ParamSet(DecisionVarSet):
    def __init__(self, init, sym = ca.SX.sym):
        super().__init__(attr_names = [], sym = sym)
        super().add_vars(init = init)

"""
Class which prefixes the name on the keys for gets and sets via []
"""
class NamedDict(dict):
    def __init__(self, name, d = {}):
        self.name = name
        for k, v in d.items():
            self[k] = v
        
    def __setitem__(self, k, v):
        super().__setitem__(self.name+k, v)
        
    def __getitem__(self, k):
        return super().__getitem__(self.name+k)
