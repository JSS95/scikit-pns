"""Custom ONNX converter for PNS."""

__all__ = [
    "pns_shape_calculator",
    "pns_converter",
]


def pns_shape_calculator(operator):
    raise NotImplementedError


def pns_converter(scope, operator, container):
    raise NotImplementedError
