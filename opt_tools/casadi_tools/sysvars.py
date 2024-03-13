
"""Helper classes for decision variables in an optimization problem.

@Christian Hegeler 2022, first prototype as dict
@Kevin Haninger    2023, port to dataclass
 
Design goals:
- Easily define dataclasses which
  - allow variables to be addressed by name
  - allow typehints which better document the code
  - are single point of truth for dimensions
- (CasADi) provide automatic construction of symbolic vars
- Vectorization, also of associated attributes (e.g. initialization x0, bounds ub/lb)
- Tools to extend the whole class to trajectories and batches
"""

from __future__ import annotations  # automatic forward declaration of types
from dataclasses import dataclass, asdict, fields, field, Field, make_dataclass
from typing import Type, Tuple, Dict, List, Union, TypeVar, Any, Callable
from types import new_class

import casadi as ca
import numpy as np

Array = np.ndarray  # Type used for type hints
_SVS = TypeVar('_SVS')  # Generic type for SysVarSet
Sym = ca.SX  # Symbolic variable type


@dataclass
class SysVar:
    """Associated attributes of a decision variable"""
    x0: Array
    lb: Union[float, Array] = -np.inf
    ub: Union[float, Array] = np.inf
    
########### Helper functions ############
def enforce_sysvar(var: Union[float, list, Array, ca.DM, SysVar]) -> SysVar:
    """Ensure var is a SysVar, where x0 is a DM array."""
    if isinstance(var, (int, float, list, Array, ca.DM)):  # turn into SysVar
        var = SysVar(x0=ca.DM(var))
    elif isinstance(var, Sym):
        var = SysVar(x0=var)
    elif isinstance(var, SysVar): # ensure init val is DM
        var.x0 = ca.DM(var.x0)
    elif isinstance(var, Field): # fields can go through as they are
        var = var.default
    else:
        raise TypeError(f"Need float, list, np.ndarray, ca.DM, or SysVar, got {var}")
    return var

def decvarset_factory(name: str, dic: Tuple[str, Union[Array, SysVar, Field]]) -> SysVarSet:
    """Generate new DecVarSet with name and contents in dic."""
    decvar_dict = {k: enforce_sysvar(v) for k,v in dic.items()}
    decvar_dict['__annotations__'] = {k:SysVar for k in dic.keys()}
    return new_class(name,
                     bases=(SysVarSet,),
                     exec_body=lambda ns: ns.update(decvar_dict))

def vectorize(dic: Tuple[str, Array]) -> Array:
    """Turn a dictionary of (possibly higher dim) arrrays into a vector of dim (N, 1)."""
    return ca.vertcat(*[ca.reshape(el, np.prod(el.shape), 1) for el in dic.values()])

@dataclass
class SysVarSet:
    """Helper class for sets of decision variables.

    This is uses dataclass to automatically generate an __init__ for each
    inherited class. This allows one to quickly and easily define your variables,
    then access them. E.g. if your class is

    class CustomClass(SysVarSet):
        pos = np.zeros((3,1))  # sets x0 for this variable
        vel = SysVar(x0=np.zeros(3,),
                     lb=0.2) # lb/ub default to -/+np.inf

    The symbolic var of matching dims is available at classInstance.pos.
    The variables are vectorized in shape (*,1) and available at
      symbolic: classInstance.symvec
      numeric:  classInstance.[x0, ub, lb]

    The class is fully static - after initialization, there is no change
    to the variables.

    Examples in test_sysvars.py
    """

    def __init_subclass__(cls: Type[_SVS]):
        """Called before __init__ on any class that inherits this class."""
        for name, typ in cls.__dict__.get('__annotations__', {}).items():
            assert hasattr(cls, name), f"{cls} needs defaults for all annotations, check {name}"
            if not typ == SysVar: continue # ignore non-sysvars
            default = enforce_sysvar(getattr(cls, name))
            new_field = field(default = default,
                              metadata={'var_dim':default.x0.shape},)
            setattr(cls, name, new_field)
        
        cls = dataclass(cls)

    def __post_init__(self):
        """Ensure all vars are sysvars, except if initialized with symbolics."""
        x0s = {}
        for fi in fields(self):
            var =  enforce_sysvar(getattr(self, fi.name)) # enforce the initialized values
            x0s[fi.name] = var.x0
            if not isinstance(getattr(self, fi.name), Sym):
                sym = Sym.sym(fi.name, *fi.metadata['var_dim'])
                setattr(self, fi.name, sym)
        self.x0 = vectorize(x0s)
                            
    def __add__(self, other):
        new_dict = {f.name:f for f in fields(self) + fields(other)}
        new_name = self.__class__.__name__+other.__class__.__name__
        new = decvarset_factory(new_name, new_dict)()
        
        # Carry over _exactly_ the symbolic var
        for attr, val in {**asdict(self), **asdict(other)}.items():
            setattr(new, attr, val)
        return new

    def __getattribute__(self, attr):
        """We define these here instead of setting as attributes because
        they're derived from the fields, doing it like this makes that explicit."""
        if attr == "symvec":
            return vectorize(asdict(self))
        elif attr == 'lb':
            return vectorize({f.name: np.full_like(f.default.x0, f.default.lb) for f in fields(self)})
        elif attr == 'ub':
            return vectorize({f.name: np.full_like(f.default.x0, f.default.ub) for f in fields(self)})
        else:
            return object.__getattribute__(self, attr)

    @classmethod
    def skip_sym(cls, **kwargs):
        """Make a SysVarSet without contructing symbolic attributes.
        Any attribute not specified in kwargs will be taken from default."""
        inst = cls()
        for fi in fields(inst):
            var = enforce_sysvar(kwargs.get(fi.name, fi.default))
            setattr(inst, fi.name, var.x0)
        return inst
        
    @classmethod
    def from_vec(cls: Type[_SVS], vec: Array) -> _SVS:
        """Turn a vector into a class instance."""
        inst = cls()
        read_pos = 0
        for field_info in fields(inst):
            shape = field_info.metadata['var_dim']
            size = np.prod(shape)
            setattr(inst, field_info.name,
                    ca.reshape(vec[read_pos:read_pos+size], shape).full())
            read_pos += size
        return inst

    @classmethod
    def generate_traj(cls: Type[_SVS], horizon: int) -> SysVarSet:
        """Turn SysVarSet cls into a trajectory by extending the dimensions."""
        def extend(var: SysVar):
            sh = (*var.x0.shape[:-1], horizon)
            ret_var = SysVar(x0=np.full(sh, var.x0), lb=var.lb, ub=var.ub)
            return ret_var
        defaults = {fi.name: fi.default for fi in fields(cls)}
        traj_dict = {k: extend(v) for k, v in defaults.items()}
        traj = decvarset_factory(cls.__name__+"_traj", traj_dict)()
        traj.parent = cls
        return traj

    def continuity_constraint(self, x0:Sym=None, x_next:Sym=None) -> Sym:
        """Generate continuity constraints to a trajectory."""
        assert getattr(self, 'parent'), 'Continuity can only be applied to trajectories'
        selfdict = asdict(self)
        g = []
        if x0:
            x0dict = asdict(x0)
            g = vectorize({k: selfdict[k][:, 0] - x0dict[k] for k in selfdict})
        if x_next:
            x_nextdict = asdict(x_next)
            gnext = vectorize(
                {k: selfdict[k][:, 1:] - x_nextdict[k][:, :-1] for k in selfdict.keys()})
            g = ca.vertcat(g, gnext)
        return g
