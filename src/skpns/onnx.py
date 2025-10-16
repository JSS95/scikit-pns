"""Custom ONNX converter for PNS."""

__all__ = [
    "pns_shape_calculator",
    "pns_converter",
]


def pns_shape_calculator(operator):
    op = operator.raw_operator
    input_type = operator.inputs[0].type.__class__
    input_dim = operator.inputs[0].get_first_dimension()
    output_type = input_type([input_dim, op.n_components])
    operator.outputs[0].type = output_type


def pns_converter(scope, operator, container):
    raise NotImplementedError
