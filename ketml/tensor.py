import numpy as np

from typing import Sequence
from enum import Enum

from ketml.utils import as_2d

class TensorOperation(Enum):
    NONE = "none"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"

class TensorContext:
    def __init__(self, operation: TensorOperation = TensorOperation.NONE, children: Sequence[Tensor] | None = None):
        self.operation = operation
        self.children = children or []

        if not self.is_static() and not children:
            raise Exception("A tensor must have children if the tensor operation is not set to none.")

    def is_static(self) -> bool:
        return self.operation is TensorOperation.NONE

class Tensor:
    def __init__(self, *data: Sequence[int|float]|np.ndarray, context: TensorContext = None):
        self.data = as_2d(np.asarray(data, dtype=np.float32))
        self.context = context or TensorContext()

        if self.data.ndim > 2:
            raise Exception("Tensor shapes may not exceed 2d.")

    def __add__(self, other):
        return Tensor(context=TensorContext(operation=TensorOperation.ADD, children=(self, other)))

    def __sub__(self, other):
        return Tensor(context=TensorContext(operation=TensorOperation.SUBTRACT, children=(self, other)))

    def __mul__(self, other):
        return Tensor(context=TensorContext(operation=TensorOperation.MULTIPLY, children=(self, other)))

    def __truediv__(self, other):
        return Tensor(context=TensorContext(operation=TensorOperation.DIVIDE, children=(self, other)))

    def get_context(self) -> TensorContext:
        return self.context

    @staticmethod
    def random(*size: Sequence[int]|np.ndarray) -> Tensor:
        return Tensor(np.random.random(np.asarray(size, dtype=int).squeeze()))