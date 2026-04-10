import pyopencl as cl

from typing import Sequence

from ketml.buffer import Buffer
from ketml.device import Device
from ketml.operation import Operation, OperationContext

class Kernel:
    def __init__(self, operation_context: OperationContext, operations: Sequence[Operation]|None = None):
        self.operations = operations or []
        self.dispatch_shape: tuple[int, int] = (1, 0)
        self.operation_context = operation_context

        self.compute_kernel: cl.Kernel = None

    def add_operation(self, operation: Operation):
        self.dispatch_shape = operation.get_output_shape()

        self.operations.append(operation)

    def compile(self):
        sources = []
        calls = []

        for operation in self.operations:
            sources.append(operation.get_source())
            calls.append(operation.get_call())

        operation_definitions = '\n'.join(sources)
        body = '\n\t'.join(calls)

        buffer_definitions = ',\n\t'.join(buffer.get_decleration() for buffer in [
            self.operation_context.static_buffer,
            self.operation_context.dynamic_buffer
        ])

        source = '\n'.join([
            operation_definitions,
            '__kernel void compute_kernel(',
            f'\t{buffer_definitions}',
            ') {',
            f'\t{body}',
            '}'
        ])

        program = cl.Program(
            Device.get_default().get_context(),
            source
        ).build()

        self.compute_kernel = program.compute_kernel

    def is_compiled(self) -> bool:
        return self.compute_kernel is not None