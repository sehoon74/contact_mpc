# Kevin Haninger, 2022
from sys import version_info

import casadi as ca
import numpy as np

class DecisionVarDict(dict):
    """
    Dictionary for sets of variables; storing and formatting their symbolic variables,
      bounds, or other associated attributes.
    
    Use: Specify your attributes than instantiate with add_vars.
      The minimum is a dictionary of initial values, `inits`, which
      is used to construct symbolic vars of matching dimensions.

      Additional attributes can be specified in **kwargs by giving dicts
      with either complete specification for each variable or a default
      in defaults.    
    """
    def __init__(self, attr_names, name = '', sym = ca.SX.sym,
                 attr_defaults = dict(lb = -np.inf,
                                      ub =  np.inf,
                                      cov_init = 0,
                                      meas_noise = 0,
                                      proc_noise = 0)):
        """
        IN: attr_names, a list of strings for the attributes*
              This is not automatically derived so you can dump a
              dictionary of attrs and pick out the relevant ones with attr_names
        IN: name, a prefix for all variables to avoid name collisions
        IN: attr_defaults, a dictionary of the default values for attributes
        """
        super().__init__()
        assert version_info >= (3, 6), "Python 3.6 required to guarantee dicts are ordered"                
        self.attr_names = attr_names
        self.name = name
        self.attr_defaults = attr_defaults
        self.sym = sym
        self.__vars = {k:{} for k in ['init']+attr_names}

    def add_vars(self, init, **kwargs):
        for name, ini in init.items():
            ini = np.array(ini)
            self.__vars['init'][self.name+name] = ini
            self[self.name+name] = self.sym(name, *ini.shape)

            # Add the additional attr for each variable
            for attr in self.attr_names:
                assert attr in self.attr_defaults or name in kwargs.get(attr), f"Attribute for {name} must have either defaults or specified in kwargs"
                val = kwargs[attr][name] if name in kwargs.get(attr, []) else self.attr_defaults[attr]
                self.__vars[attr][self.name+name] = np.full(ini.shape, val)
        
    def __delitem__(self, key):
        raise NotImplementedError

    def get_from_shortname(self, key):
        return super().__getitem__(self.name+key)

    def __add__(self, other):
        """ Adds other variables to self, the attributes are projected down to those in self """
        assert set(self.attr_names) <= set(other.attr_names), "LHS set has attributes that RHS doesn't"
        for attr in list(self.attr_names)+['init']:
            self.__vars[attr].update(other.get_vars(attr)) # Merge the dictionaries per attr in self
        self.update(other)
        return self
    
    def __len__(self):
        return sum(val.size for val in self.__vars['init'].values())

    def __str__(self):
        s = f"***** Decision Vars, name: {self.name} *****\n"
        s += f"Attributes: {self.attr_names}\nVars: \n"
        for key in self.keys():
            s += f"  {key}: {self.get(key)}, shape: {self.get(key).shape}\n"
        return s

    def vectorize_dict(self, d):
        for k in d.keys():
            if type(d[k]) == float or type(d[k]) == int:
                d[k] = ca.DM(d[k])
        return ca.vertcat(*[d[k].reshape((-1,1)) for k in self.keys()])
    
    def vectorize_attr(self, attr = None):
        """ Turns dictionary d or attribute attr into vector """
        d = self.get_vars(attr)
        return self.vectorize_dict(d)
        
    def dictize(self, vec):
        """ Returns a dict of vec, with reshaping as needed """
        d = {}
        read_pos = 0
        for key, init in self.__vars['init'].items():
            v_size  = init.size
            v_shape = init.shape
            if len(v_shape) == 0: v_shape = (1, 1)
            if len(v_shape) == 1: v_shape = (*v_shape,1)
            d[self.name+key] = ca.reshape(vec[read_pos:read_pos+v_size], v_shape)
            read_pos += v_size
        return d

    def get_vectors(self, *attr_names):
        """ Returns a tuple of vectorized attributes """
        return tuple([self.vectorize_attr(attr) for attr in attr_names])
    
    def get_vars(self, attr = None):
        """ Returns a dict of attr, none returns symbolic """
        if not attr or attr == 'sym':
            return {k:self[k] for k in self}
        return {k:self.__vars[attr][k] for k in self.__vars[attr]}

    def clone_from_attrs(self, **attrs):
        dec_vars = DecisionVarDict(attr_names=self.attr_names,
                                   name=self.name,
                                   attr_defaults=self.attr_defaults)
        dec_vars.add_vars(**attrs)
        return dec_vars
    
    def clone_and_extend(self, H):
        attrs = {attr:{k[len(self.name):]:ca.repmat(self.__vars[attr][k],1,H) for k in self.__vars[attr].keys()} for attr in self.attr_names+['init']}
        return self.clone_from_attrs(**attrs)
    
    def clone_and_vectorize(self, new_vector_name):
        """ Creates a new dec var set with a single vector for the set """
        attrs = {attr:{new_vector_name:self.vectorize_attr(attr)} for attr in self.attr_names+['init']}
        dec_vars = self.clone_from_attrs(**attrs)
        dec_vars[new_vector_name] = self.vectorize_attr() # want exactly the same sym vars
        return dec_vars

class ParamDict(DecisionVarDict):
    def __init__(self, init, sym = ca.SX.sym):
        super().__init__(attr_names = [], sym = sym)
        super().add_vars(init = init)

class NamedDict(dict):
    def __init__(self, name, d = {}):
        self.name = name
        for k,v in d.items():
            super().__setitem__(self.name+k, v)

    def __getitem__(self, k):
        return super().__getitem__(self.name+k)

    def __setitem__(self, k, v):
        super().__setitem__(self.name+k, v)
        
