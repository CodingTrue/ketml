from ketml.tensor import Tensor

class ComputeGraph:
    def __init__(self, root: Tensor):
        self.root = root
        self.static_tensors, self.dynamic_tensors = self.build_graph(tensor=root)

    @staticmethod
    def build_graph(
        tensor: Tensor,
        static_tensors: list[Tensor] | None = None,
        dynamic_tensors: list[Tensor] | None = None
    ) -> tuple[list[Tensor], list[Tensor]]:
        static_tensors = static_tensors or []
        dynamic_tensors = dynamic_tensors or []

        ctx = tensor.get_context()
        if ctx.is_static():
            if tensor not in static_tensors:
                static_tensors.append(tensor)
        else:
            for child in ctx.children:
                static_tensors, dynamic_tensors = ComputeGraph.build_graph(
                    tensor=child,
                    static_tensors=static_tensors,
                    dynamic_tensors=dynamic_tensors
                )

            if tensor not in dynamic_tensors:
                dynamic_tensors.append(tensor)
        return static_tensors, dynamic_tensors