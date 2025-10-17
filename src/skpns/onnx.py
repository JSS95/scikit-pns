"""Custom ONNX converter for PNS."""

import numpy as np
from skl2onnx.algebra.onnx_ops import (
    OnnxAdd,
    OnnxDiv,
    OnnxMatMul,
    OnnxMul,
    OnnxSin,
    OnnxSub,
)
from skl2onnx.common.data_types import guess_numpy_type

from .pns import _R

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
    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs

    X = operator.inputs[0]
    dtype = guess_numpy_type(X.type)

    for v, r in zip(op.v_[:-1], op.r_[:-1]):
        v = v.astype(dtype)
        r = r.reshape(1).astype(dtype)
        A = onnx_proj(X, v, r, opv)
        X = onnx_to_unit_sphere(A, v, r, opv)
    v, r = op.v_[-1].astype(dtype), op.r_[-1].reshape(1).astype(dtype)
    A = onnx_proj(X, v, r, opv)
    X = onnx_to_unit_sphere(A, v, r, opv, out[:1])
    X.add_to(scope, container)


def onnx_proj(X, v, r, opv, outnames=None):
    r = r.reshape(1)
    geod = OnnxMatMul(X, v.reshape(-1, 1), op_version=opv)  # (N, 1)
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
