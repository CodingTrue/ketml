import pyopencl as cl
import numpy as np

from typing import Any
from importlib import resources
from ketml import builtin_operations

def get_cl_info(target: object, key: int) -> Any:
    try:
        return target.get_info(key)
    except (cl.LogicError, Exception):
        return None

def as_2d(value: np.ndarray) -> np.ndarray:
    return np.atleast_2d(value.squeeze())

def read_builtin_op(filepath: str) -> str:
    return (resources.files(builtin_operations) / filepath).read_text()