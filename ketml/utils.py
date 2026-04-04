import pyopencl as cl
import numpy as np

from typing import Any

def get_cl_info(target: object, key: int) -> Any:
    try:
        return target.get_info(key)
    except (cl.LogicError, Exception):
        return None

def as_2d(value: np.ndarray) -> np.ndarray:
    return np.atleast_2d(value.squeeze())