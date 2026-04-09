import re

from dataclasses import dataclass
from typing import Sequence

from ketml.buffer import Buffer
from ketml.tensor import Tensor
from ketml.utils import read_builtin_op

@dataclass
class TensorRecord:
    buffer: Buffer
    offset: int
    size: int
    output_size: int
    output_offset: int
    output_shape: tuple[int, int]
    is_static: bool

class OperationContext:
    def __init__(self, static_buffer: Buffer, dynamic_buffer: Buffer):
        self.tensor_map: dict[Tensor, dict[str, Any]] = {}

        self.static_buffer = static_buffer
        self.dynamic_buffer = dynamic_buffer

    def record_tensor(
        self,
        tensor: Tensor,
        output_size: int,
        output_offset: int,
        output_shape: tuple[int, int]
    ):
        is_static = tensor.get_context().is_static()
        buffer = (self.static_buffer if is_static else self.dynamic_buffer)
        size = output_size + output_offset

        self.tensor_map[tensor] = TensorRecord(
            buffer=buffer,
            offset=buffer.size,
            size=size,
            output_size=output_size,
            output_offset=output_offset,
            output_shape=output_shape,
            is_static=is_static
        )

        buffer.expand(amount=size)

    def get_tensor_record(self, tensor: Tensor) -> TensorRecord:
        return self.tensor_map[tensor]

def reference_tensor(tensor_record: TensorRecord, name: str) -> list[tuple[str, str]]:
    return [f"{cl_type} {name}_{field_name}" for cl_type, field_name in [
        (f'{tensor_record.buffer.asq.value} float*', 'data'),
        ('int', 'offset'),
        ('int', 'size'),
        ('int', 'output_offset'),
        ('bool', 'is_static'),
    ]]

def pack_tensor(tensor_record: TensorRecord) -> list[str]:
    return [str(arg).lower() for arg in [
        tensor_record.buffer.identifier,
        tensor_record.offset,
        tensor_record.size,
        tensor_record.output_offset,
        tensor_record.is_static
    ]]

class Operation:
    def __init__(
        self,
        identifier: str,
        source: str,
        arguments: Sequence[Tensor]|None = None,
        output_tensor: Tensor|None = None,
        context: OperationContext|None = None
    ):
        self.identifier = identifier
        self.source = source

        self.arguments = arguments or []
        self.output_tensor = output_tensor
        self.context = context

    def specialize(self, tensor: Tensor, context: OperationContext) -> Operation:
        op = Operation(
            identifier=self.identifier,
            source=self.source,
            arguments=tensor.get_context().children,
            output_tensor=tensor,
            context=context
        )

        op.register_output()
        return op

    def register_output(self):
        self.context.record_tensor(
            tensor=self.output_tensor,
            output_size=self.get_output_size(),
            output_offset=self.get_output_offset(),
            output_shape=self.get_output_shape()
        )

    def get_source(self) -> str:
        source = self.source
        tensors = iter([*self.arguments, self.output_tensor])

        for match in re.finditer(r"Tensor\s*([^0-9][a-zA-Z0-9_]*)?", source):
            match_str, name = match.group(0), match.group(1)

            tensor_reference = reference_tensor(
                tensor_record=self.context.get_tensor_record(tensor=next(tensors)),
                name=name
            )
            reference_str = ',\n\t'.join(field for field in tensor_reference)

            source = source.replace(match_str, reference_str, count=1)
            source = source.replace(f"{name}.", f'{name}_')

        source = source.replace(self.identifier, f"{self.identifier}_{self.get_variant_name()}", count=1)
        return source

    def get_variant_name(self) -> str:
        variant = ""
        for tensor in self.arguments:
            is_static = tensor.get_context().is_static()

            buffer = self.context.static_buffer if is_static else self.context.dynamic_buffer
            variant += buffer.identifier[0]
        return variant

    def get_call(self) -> str:
        arguments = []

        for tensor in self.arguments:
            arguments.extend(pack_tensor(tensor_record=self.context.get_tensor_record(tensor=tensor)))
        arguments.extend(pack_tensor(tensor_record=self.context.get_tensor_record(tensor=self.output_tensor)))

        return f"{self.identifier}_{self.get_variant_name()}({', '.join(arguments)});"

    def get_output_size(self) -> int:
        return min([self.context.get_tensor_record(tensor=tensor).size for tensor in self.arguments])

    def get_output_shape(self) -> tuple[int, int]:
        return (1, self.get_output_size())

    def get_output_offset(self) -> int:
        return 0

ADD_OP = Operation('add', read_builtin_op(filepath='add.ketcl'))
SUBTRACT_OP = Operation('subtract', read_builtin_op(filepath='subtract.ketcl'))
MULTIPLY_OP = Operation('multiply', read_builtin_op(filepath='multiply.ketcl'))
DIVIDE_OP = Operation('divide', read_builtin_op(filepath='divide.ketcl'))