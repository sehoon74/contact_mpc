"""Helper classes for decision variables in a JAX optimization problem.

@Christian Hegeler 4.2022, initial impementation as decision_vars.py
@Kevin Haninger 11.2023, port to JAX"""

from __future__ import annotations  # automatic forward declaration of types
from functools import partial
from typing import Dict, Union, Callable, Iterable, Tuple
from typing_extensions import dataclass_transform
from dataclasses import field, fields, Field, dataclass, asdict

import jax.numpy as jnp
from jax import Array
from jax.tree_util import tree_map, tree_flatten, tree_unflatten, tree_leaves
from jax.flatten_util import ravel_pytree
import matplotlib.pyplot as plt

from .mystruct import field as str_field
from .mystruct import dataclass as str_dataclass


def tree_map_relaxed(fn: Callable[[Array, *Array], Array],
                     tree: SysVarSet,
                     *rest: SysVarSet,
                     typ: Union[SysVarSet, Traj, Batch, TrajBatch]=None
                     ) -> Union[SysVarSet, Traj, Batch, TrajBatch]:
    """tree_map but ignoring class/metadata incompatibility.
    IN:
      fn: function to be called on all leaves of the tree
      tree: SysVarSet to get mapped
      *rest: Leaves of rest are given as additional arguments to fn
      typ: Type to give result back as. If none, type of tree is used
    OUT:
      tree-mapped result, of type determined by typ"""
    treedef = tree._treedefs[typ] if typ else tree_flatten(tree)[1]
    all_leaves = [tree_flatten(tree)[0]]+[tree_flatten(r)[0] for r in rest]
    return treedef.unflatten(fn(*l) for l in zip(*all_leaves))

@dataclass
class SysVar:
    x0: Array
    lb: Union[float, Array] = -jnp.inf
    ub: Union[float, Array] = jnp.inf


def enforce_sysvar(var: Union[float, list, Array, SysVar]) -> SysVar:
    """Ensure var is a sysvar, where x0 is an Array."""
    if not isinstance(var, SysVar):
        var = SysVar(x0=var)
    if not isinstance(var.x0, Array) and not isinstance(var.x0, Field):
        var.x0 = jnp.atleast_1d(var.x0)
    return var


class Traj:
    """A trajectory, i.e. a temporally-related series of SysVars."""
    def reduce_traj(self, fn: Callable) -> SysVarSet:
        """Apply the function fn to all nodes in SysVar.
        In:
          fn: function to apply - *must take `axis` as kwarg*
        Out:
          Reduced SysVar where the traj_dim has been reduced by fn"""
        red_fn = partial(fn, axis=self.traj_dim)
        red_dict = asdict(tree_map(red_fn, self))
        red_dict['traj_dim'] = None
        return type(self).__bases__[0](**red_dict)

    def __getitem__(self, index: int) -> SysVarSet:
        """Access the index of trajectory."""
        return tree_map_relaxed(lambda l: jnp.take(l, index, self.traj_dim),
                                self,
                                typ=SysVarSet)

    def __iter__(self):
        """Return list over traj elements."""
        assert not isinstance(self, Batch), f'Cant iter a TrajBatch, got {self}'
        leaves, treedef = tree_flatten(self)
        # we take advantage that zip(*array) goes over the 0th dimension
        return iter([treedef.unflatten(leaf) for leaf in zip(*leaves)])

class Batch:
    """A batch, i.e. a parallel collection of SysVars."""
    def __add__(self, other: Union[Batch, SysVarSet]):
        """Concatenate all leaves along dim 0, assuming batch_dim=0."""
        if not isinstance(other, Batch):
            other = tree_map(lambda l: jnp.expand_dims(l, axis=0), other)
        return tree_map_relaxed(lambda s, o: jnp.vstack((s, o)), self, other)

    def __iter__(self):
        """Return (low performance!) iterator over batch elements."""
        leaves, treedef = tree_flatten(self)
        # we take advantage that zip(*array) goes over the 0th dimension
        trees = [treedef.unflatten(leaf) for leaf in zip(*leaves)]
        return iter(trees)

    def __len__(self):
        return list(asdict(self).values())[-1].shape[self.batch_dim]
    
    def reduce_batch(self, fn: Callable) -> SysVarSet:
        """Apply the function `fn` to reduce all nodes in SysVar.
        In:
          fn: function to appy - *must take `axis` as kwarg*
        Out
          Reduced SysVar where the batch_dim has been reduced by fn"""
        red_fn = partial(fn, axis=self.batch_dim)
        red_dict = asdict(tree_map(red_fn, self))
        red_dict['batch_dim'] = None
        red_dict['traj_dim'] = 0 if self.traj_dim is not None else None
        return type(self).__bases__[0](**red_dict)


def to_traj(list_of_svs: Iterable[SysVarSet]) -> Traj:
    return tree_map_relaxed(lambda *l: jnp.stack(l, axis=0), *list_of_svs, typ=Traj)

def to_batch(list_of_svs: Iterable[SysVarSet]) -> Batch:
    return tree_map_relaxed(lambda *l: jnp.stack(l, axis=0), *list_of_svs, typ=Batch)

class TrajBatchMeta(type):
    def __instancecheck__(cls, instance):
        return isinstance(instance, Traj) and isinstance(instance, Batch)

    def __subclasscheck__(cls, subcls):
        return issubclass(subcls, Traj) and issubclass(subcls, Batch)


class TrajBatch(Traj, Batch, metaclass=TrajBatchMeta): ...

@dataclass_transform(field_specifiers=(str_field,))
@dataclass
class SysVarSet:
    """Abstract dataclass for system variables (Input, State, Output).

    It extends flax.struct.dataclass, which registers with
    JAX as a pytree to provide flattening, ravelling, etc.

    This base class gets extened to Traj[SysVarChild], Batch[SysVarChild]
    to represent trajectories and batches of them.

    If the dimensions of an attribute are, e.g. (N,M,...), a
      Traj has leaves of size (horizon, N, M, ...)
      Batch (batch_size, N, M, ...)
      TrajBatch (batch_size, horizon, N, M, ...)

    Initializing traj/batches:
      With the constructor DerivedClass(**kwargs, traj_dim=dt, batch_dim=db),
       automatically inherits on construction from Traj / Batch when the
       dims are not None

      With .generate_traj, .generate_batch on existing SysVarSet

      With tree_map_relaxed, giving the optional typ kwarg to specify
    """
    
    traj_dim: int = str_field(default=None, pytree_node=False)
    batch_dim: int = str_field(default=None, pytree_node=False)

    def __init_subclass__(cls, **kwargs):
        """This function is called when a subclass is defined, and before that
        subclass is instantiated.  We use this to enforce some desired properties
        of the attributes, as well as set up the lowerbound/upperbound dicts."""
        
        # lb_d and ub_d are lower/upper bound dictionaries,
        # they are initialized like this so that they can handle inheritance
        lb_d = getattr(cls, '_lb_dict', {})
        ub_d = getattr(cls, '_ub_dict', {})
        for name, typ in cls.__dict__.get('__annotations__', {}).items():
            assert hasattr(
                cls, name), f"{cls} needs defaults for all annotations, check {name}"
            var = enforce_sysvar(getattr(cls, name))
            lb_d[name], ub_d[name] = var.lb, var.ub
            setattr(cls, name, field(default_factory=lambda x0=var.x0: x0))
        
        cls = str_dataclass(cls)
        
        cls._lb_dict = lb_d
        cls._ub_dict = ub_d

        if not issubclass(cls, Traj) and not issubclass(cls, Batch):
            # if in base class, build those treedefs!
            cls._treedefs = {SysVarSet: tree_flatten(cls())[1],
                             Traj: tree_flatten(cls(traj_dim=0))[1],
                             Batch: tree_flatten(cls(batch_dim=0))[1],
                             TrajBatch: tree_flatten(cls(batch_dim=0, traj_dim=1))[1]}

    def __new__(cls, *args, **kwargs):
        """We override the object constructor, which is called before __init__
        such that we can change the object to inherit from Traj or Batch if
        it is being initialized with traj_dim or batch_dim"""
        new_cls = cls
        if kwargs.get('batch_dim') is not None and not issubclass(cls, Batch):
            new_cls = type("Batch"+new_cls.__name__, (new_cls, Batch), {})
        if kwargs.get('traj_dim') is not None and not issubclass(cls, Traj):
            new_cls = type("Traj"+new_cls.__name__, (new_cls, Traj), {})
        return object.__new__(new_cls)

    def __getattribute__(self, attr):
        """We overwrite symvec, x0, lb, and ub here because these are derived
        properties, I don't want them living as attributes where they could
        add state."""
        if attr == "symvec" or attr == "x0":  # symvec for compatibility w CasADi
            return ravel_pytree(self)[0]
        elif attr == 'lb':
            _, meta = tree_flatten(self)
            data, _ = tree_flatten(type(self)(**self._lb_dict))
            lb_tree = tree_map(lambda l, b: jnp.full(
                l.shape, b), self, tree_unflatten(meta, data))
            return ravel_pytree(lb_tree)[0]
        elif attr == 'ub':
            _, meta = tree_flatten(self)
            data, _ = tree_flatten(type(self)(**self._ub_dict))
            ub_tree = tree_map(lambda l, b: jnp.full(
                l.shape, b), self, tree_unflatten(meta, data))
            return ravel_pytree(ub_tree)[0]
        else:
            return object.__getattribute__(self, attr)

    @classmethod
    def pytree_fields(cls):
        """Return all fields which are pytree_nodes."""
        return [f for f in fields(cls) if f.metadata.get('pytree_node', True)]
 
    def generate(self,
                 horizon: int = None,
                 batch_size: int = None,
                 factory: Callable[[Tuple[int], Array], Array] = jnp.full
                 ) -> Union[SysVarSet, Traj, Batch, TrajBatch]:
        """Generate a Traj, Batch, or TrajBatch from a SysVarSet.
        IN:
          horizon: length of trajectory
          batch_size: number of elements in batch
          factory: a factory to be called on leaves of the class
                   the factory is called with ((*dims), leaf), where
                   dims = (batch_size, horizon, *leaf.shape)
        """
        isTraj = (horizon is not None) or isinstance(self, Traj)
        isBatch = (batch_size is not None) or isinstance(self, Batch)
        if isBatch:
            typ = TrajBatch if isTraj else Batch
        else: # not a batch
            typ = Traj if isTraj else SysVarSet
        def fn(l):
            dims = tuple(el for el in (batch_size, horizon, *l.shape) if el is not None)
            return factory(dims, l)
        return tree_map_relaxed(fn, self, typ=typ)
    
    def generate_traj(self,
                      horizon: int,
                      factory: Callable[[Tuple[int], Array], Array] = jnp.full
                      ) -> Traj:
        """Extend the sysvar into a trajectory by applying factory function. Factory
        function should take a tuple of dimensions as first argument."""
        return self.generate(horizon=horizon, factory=factory)

    def generate_batch(self,
                       batch_size: int,
                       factory: Callable[[Tuple[int], Array], Array] = jnp.full
                       ) -> Batch:
        """Extend the sysvar into a batch by applying factory function. Factory
        function should take a tuple of dimensions as first argument."""
        return self.generate(batch_size=batch_size, factory=factory)

    @classmethod
    def update_defaults(cls, **kwargs):
        """Update the defaults of attributes in kwargs."""
        for f in cls.pytree_fields():
            if (val := kwargs.get(f.name, None)) is not None:
                # update the default factory in __dataclass_fields__
                var = enforce_sysvar(val)
                f.default_factory = lambda x0=var.x0: x0
        new_cls = type(cls.__name__, (cls,), {}) # rebuild dclass so __init__ gets rewritten
        return new_cls

    def plot(self):
        attrs = self.pytree_fields()
        fig, axs = plt.subplots(len(attrs), 1)
        if len(attrs) == 1: axs = [axs]
        names = [f.name for f in attrs]
        data = [getattr(self, n) for n in names]
        for name, data, ax in zip(names, data, axs):
            ax.plot(data)
            ax.set_ylabel(name)
        plt.show()
        #return fig, axs
