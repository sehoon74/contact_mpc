"""
Copyright (c) 2022, Kevin Haninger
Helper classes for decision variables in an optimization problem
"""

import casadi as ca
import numpy as np
from sys import version_info


class DecisionVarSet:
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

    def __init__(self,
                 inits,
                 symb = ca.SX.sym,
                 defaults = {lb = -np.inf, ub = np.inf, cov = 0, noise = 0,}
                 **kwargs):

        assert version_info >= (3, 6), "Python 3.6 required to guarantee dicts are ordered"
        self.__vars = {}          # Individual variables

        for name, init in inits.items():
            self.__vars['init'][var] = np.array(init)
            self.__vars['sym'][name] = symb(name, init.shape)
            for attr, vals in kwargs.items():
                try:
                    default = np.full(defaults[attr], init.shape)
                except:
                    print(f"No default for {attr}, make sure attr dicts are full")
                self.__vars[attr][name] = vals.get(name, default)

    def __getitem__(self, key):
        return self.__vars['sym'][key]

    def __len__(self):
        return sum(len(val.size) for val in self.__vars['init'].values())

    def __str__(self):
        s = "** Decision variables **\n"
        for key in self.__keys:
            s += f"{key}: {self[key]}\n"
        return s

    def vectorize(self, attr):
        return ca.vertcat(*[el.reshape((-1,1)) for el in self.__vars[attr].values()])

    def dictize(self, vec):
        """
        vec is the numerical optimization results, fills the dict x with reshaping as needed
        """
        assert len(x_opt) == len(self), "Length of optimization doesn't match initial x0"
        read_pos = 0
        d = {}
        for key, init in self.__['init'].items():
            v_size  = init.size
            v_shape = init.shape
            if len(v_shape) == 1: v_shape = (*v_shape,1)
            d[key] = np.squeeze(np.reshape(var[read_pos:read_pos+v_size], *v_shape))
            read_pos += v_size
        return d
    
    def get_vectors(self, *argv):
        return tuple([self.vectorize(arg) for arg in argv])

    def get_deviation(self, key):
        """
        Returns difference between initial value and symbolic (or numeric) value
        """
        return self.__vars['sym'][key]-self.__vars['init'][key]
