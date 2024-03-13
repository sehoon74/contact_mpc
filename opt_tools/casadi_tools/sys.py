from typing import Union, Dict, Callable, Any
from dataclasses import asdict

import casadi as ca

from .sysvars import SysVarSet, Sym, Array

Input = SysVarSet
State = SysVarSet
MPCParams = SysVarSet

# Helper functions
def flatten_sym(d: Union[Dict[str, Union[SysVarSet, Sym]], SysVarSet]
                ) -> Dict[str, Sym]:
    """Make d, which may contain SysVarSets, into pure symbolic dict."""
    if isinstance(d, SysVarSet): # single SysVarSet
        return asdict(d)
    d_ret = {}
    for k, v in d.items():
        if isinstance(v, SysVarSet):
            d_ret.update(asdict(v))
        elif isinstance(v, Sym):  # symbolic var or expression
            d_ret[k] = v
    return d_ret


def vectorize_fn_eval(fn: Callable[[SysVarSet], Union[Dict, SysVarSet]],
                      traj_args: Dict[str, SysVarSet], **kwargs
                      ) -> Union[Dict[str, Sym], SysVarSet]:
    """Evaluate the *python* fn which takes SysVarSet args and have it evaluate a
       time-series/trajectory of SysVarSet args. This is roughly the vectorization functionality
       we get from CasADi fns, but re-built so we can use a Python fn with SysVarSet args.

       IN:
         fn: Python function which takes SysVarSets as arguments, and returns a dict or SysVarSet
         traj_args: Arguments over which to vectorize
         kwargs: other arguments to fn, not to be vectorized (i.e. they stay as singletons)
       OUT: The output in the format of the fn output, but with time-series outputs"""

    # Get single elements from the default initialization of the parents to the trajectory inputs
    single_kwargs = {k: getattr(v, 'parent', type(v))() if isinstance(
        v, SysVarSet) else v for k, v in traj_args.items()}
    single_res = fn(**single_kwargs, **kwargs) # get result for a single eval

    # Flatten the arguments/results into single dicts
    flat_args = flatten_sym(kwargs) # get any symbolic args from kwargs
    flat_args.update(flatten_sym(
        {k: v for k, v in single_kwargs.items() if isinstance(v, SysVarSet)}))
    flat_res = flatten_sym(single_res)

    # Make a CasADi function with arguments/returns from the dicts
    ca_fn = ca.Function(fn.__name__.lstrip('_'),
                        list(flat_args.values()), list(flat_res.values()),
                        list(flat_args.keys()), list(flat_res.keys()))

    # Now call the function with the _real_ (i.e. vectorized) arguments
    # letting CasADi handle vectorization over columns
    args = flatten_sym(kwargs)
    args.update(flatten_sym(traj_args))
    res = ca_fn(**args)
    
    return type(single_res)(**res)


class AbstractSys:
    def _step(self, u: Input, x: State, **kwargs) -> State:
        """Discrete-time dynamic update equation for singleton input and state."""
        return NotImplementedError(f"Write your _step for derived class {type(self)}")

    def step(self, u: Input, x: State, **kwargs) -> State:
        """Returns singleton or trajectory result of _step for u/x."""
        return vectorize_fn_eval(self._step, {'u':u, 'x':x}, **kwargs)
    
    def _cost(self, u: Input, x: State, **kwargs) -> Dict:
        """Cost for a singleton input and state."""
        return NotImplementedError(f"Write your _cost for derived class {type(self)}")

    def cost(self, u: Input, x: State, **kwargs) -> float:
        """Returns total cost for the singleton or trajectories u/x."""
        traj_cost = vectorize_fn_eval(self._cost, {'u':u, 'x':x}, **kwargs)
        return {k:ca.sum2(v) for k, v in traj_cost.items()}
    
