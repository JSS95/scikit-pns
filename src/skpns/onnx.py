"""Custom ONNX converter for PNS."""

import numpy as np
from skl2onnx.algebra.onnx_ops import (
    OnnxAcos,
    OnnxAdd,
    OnnxDiv,
    OnnxMatMul,
    OnnxMul,
    OnnxSin,
    OnnxSub,
)

from .pns import _R

__all__ = [
    "shape_calculator",
    "intrinsicpns_converter",
    "extrinsicpns_converter",
]


def shape_calculator(operator):
    op = operator.raw_operator
    input_type = operator.inputs[0].type.__class__
    input_dim = operator.inputs[0].get_first_dimension()
    output_type = input_type([input_dim, op.n_components])
    operator.outputs[0].type = output_type


def intrinsicpns_converter(scope, operator, container):
    raise NotImplementedError


def extrinsicpns_converter(scope, operator, container):
    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs

    X = operator.inputs[0]

    for v, r in zip(op.v_[:-1], op.r_[:-1]):
        v, r = v, r.reshape(1)
        A = onnx_proj(X, v, r, opv)
        X = onnx_to_unit_sphere(A, v, r, opv)
    v, r = op.v_[-1], op.r_[-1].reshape(1)
    A = onnx_proj(X, v, r, opv)
    X = onnx_to_unit_sphere(A, v, r, opv, out[:1])
    X.add_to(scope, container)


def onnx_proj(X, v, r, opv, outnames=None):
    geod = OnnxAcos(
        OnnxMatMul(X, v.reshape(-1, 1), op_version=opv),
        op_version=opv,
    )  # (N, 1)
    ret = OnnxDiv(
        OnnxAdd(
            OnnxMul(np.sin(r), X, op_version=opv),  # (N, d+1)
            OnnxMul(
                OnnxSin(OnnxSub(geod, r, op_version=opv), op_version=opv),  # (N, 1)
                v,  # (d+1,)
                op_version=opv,
            ),  # (N, d+1)
        ),  # (N, d+1)
        OnnxSin(geod, op_version=opv),  # (N, 1)
        op_version=opv,
        output_names=outnames,
    )  # (N, d+1)
    return ret


def onnx_to_unit_sphere(x, v, r, opv, outnames=None):
    R = _R(v)
    ret = OnnxMatMul(
        x,
        (1 / np.sin(r) * R[:-1:, :]).T,
        op_version=opv,
        output_names=outnames,
    )
    return ret
