import pyopencl as cl
import numpy as np

from typing import Sequence
from enum import Enum

from ketml.device import Device
from ketml.tensor import Tensor

class AddressSpaceQualifier(Enum):
    STATIC = '__constant'
    DYNAMIC = '__global'

_FLAGS = {
    AddressSpaceQualifier.STATIC: cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
    AddressSpaceQualifier.DYNAMIC: cl.mem_flags.ALLOC_HOST_PTR
}

class Buffer:
    def __init__(self, identifier: str, asq: AddressSpaceQualififer = AddressSpaceQualifier.STATIC):
        self.identifier = identifier
        self.asq = asq

        self.cl_buffer: cl.Buffer = None
        self.size = 0

    def init_buffer(self, init_tensors: Sequence[Tensor]|None = None):
        ctx = Device.get_default().get_context()

        hostbuf = np.concatenate([
            tensor.data for tensor in init_tensors
        ]) if init_tensors else None


        flags = _FLAGS[self.asq]
        hostbuf = hostbuf
        size = 0 if hostbuf is not None else self.size * np.dtype(np.float32).itemsize

        self.cl_buffer = cl.Buffer(
            context=ctx,
            flags=flags,
            size=size,
            hostbuf=hostbuf
        )

    def get_buffer(self) -> cl.Buffer:
        if self.cl_buffer is None:
            raise RuntimeError("Buffer is not initialized.")

        return self.cl_buffer

    def expand(self, amount: int):
        self.size += amount

    def get_decleration(self) -> str:
        return f"{self.asq.value} float* {self.identifier}"