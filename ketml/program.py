import pyopencl as cl

from typing import Any

from ketml.buffer import Buffer, AddressSpaceQualifier
from ketml.computegraph import ComputeGraph
from ketml.device import Device
from ketml.kernel import Kernel
from ketml.operation import Operation, OperationContext, ADD_OP, SUBTRACT_OP, MULTIPLY_OP, DIVIDE_OP
from ketml.tensor import Tensor, TensorOperation

TO2OP = {
    TensorOperation.ADD: ADD_OP,
    TensorOperation.SUBTRACT: SUBTRACT_OP,
    TensorOperation.MULTIPLY: MULTIPLY_OP,
    TensorOperation.DIVIDE: DIVIDE_OP,
}

class Program:
    def __init__(self, computegraph: ComputeGraph):
        self.computegraph = computegraph

        self.static_buffer = Buffer(identifier='static_tensors', asq=AddressSpaceQualifier.STATIC)
        self.dynamic_buffer = Buffer(identifier='dynamic_tensors', asq=AddressSpaceQualifier.DYNAMIC)

        self.operation_context = OperationContext(
            static_buffer=self.static_buffer,
            dynamic_buffer=self.dynamic_buffer
        )
        self.kernels: list[Kernel] = []

        self.prepare_kernels()
        self.compile_kernels()

    def prepare_kernels(self):
        for tensor in self.computegraph.static_tensors:
            self.operation_context.record_tensor(
                tensor=tensor,
                output_size=tensor.get_size(),
                output_offset=0,
                output_shape=tensor.get_shape()
            )

        last_shape = None
        for tensor in self.computegraph.dynamic_tensors:
            base_op = TO2OP[tensor.get_context().operation]

            operation = base_op.specialize(tensor=tensor, context=self.operation_context)

            output_shape = operation.get_output_shape()

            if not (last_shape == output_shape):
                self.kernels.append(Kernel(operation_context=self.operation_context))

            self.kernels[-1].add_operation(operation=operation)
            last_shape = output_shape

    def compile_kernels(self):
        for kernel in self.kernels:
            kernel.compile()

    def run(self, fill_all_tensors: bool = False):
        self.static_buffer.init_buffer(init_tensors=self.computegraph.static_tensors)
        self.dynamic_buffer.init_buffer()

        queue = Device.get_default().get_queue()

        for kernel in self.kernels:
            kernel.compute_kernel.set_args(
                self.static_buffer.get_buffer(),
                self.dynamic_buffer.get_buffer()
            )

        for kernel in self.kernels:
            event = cl.enqueue_nd_range_kernel(
                queue=queue,
                kernel=kernel.compute_kernel,
                global_work_size=kernel.dispatch_shape[::-1],
                local_work_size=None
            )
            event.wait()

        size, offset = 0, 0
        if fill_all_tensors:
            size = self.dynamic_buffer.size
        else:
            tensor_record = self.operation_context.get_tensor_record(tensor=self.computegraph.dynamic_tensors[-1])
            size = tensor_record.size
            offset = tensor_record.offset

        dest = Tensor.empty(size)
        dest.data = dest.data.ravel()

        cl.enqueue_copy(
            queue=queue,
            dest=dest.data,
            src=self.dynamic_buffer.get_buffer(),
            src_offset=offset * 4
        )

        if fill_all_tensors:
            pointer = 0

            for tensor in self.computegraph.dynamic_tensors:
                size = self.operation_context.get_tensor_record(tensor=tensor).size

                tensor.data = dest.data[pointer:pointer+size]
                pointer += size
        else:
            self.computegraph.dynamic_tensors[-1].data = dest.data

        queue.finish()
