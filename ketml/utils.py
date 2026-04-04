import pyopencl as cl

from typing import Any

def get_cl_info(target: object, key: int) -> Any:
    try:
        return target.get_info(key)
    except (cl.LogicError, Exception):
        return None