from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import tree_leaves

def is_float(arr: jnp.ndarray) -> bool:
    if not isinstance(arr, jnp.ndarray): return False 
    return jnp.issubdtype(arr, jnp.float16) or \
           jnp.issubdtype(arr, jnp.float32) or \
           jnp.issubdtype(arr, jnp.float64)

def get_dtype(tree):
    """Returns a float dtype from the pytree."""
    for leaf in tree_leaves(tree):
        if is_float(leaf):
            return leaf.dtype

def cast_leaves(tree: Any, float_dtype) -> Any:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    cast_leaves = [leaf.astype(float_dtype) if is_float(leaf) else leaf for leaf in leaves]
    return jax.tree_util.tree_unflatten(treedef, cast_leaves)
