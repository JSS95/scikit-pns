"""Custom ONNX converter for PNS."""

import numpy as np
import pns as pnspy
from skl2onnx.algebra.onnx_ops import (
    OnnxAcos,
    OnnxAdd,
    OnnxAtan,
    OnnxConcat,
    OnnxConstantOfShape,
    OnnxDiv,
    OnnxGather,
    OnnxMatMul,
    OnnxMul,
    OnnxShape,
    OnnxSin,
    OnnxSqrt,
    OnnxSub,
)

__all__ = [
    "shape_calculator",
    "intrinsicpns_converter",
    "extrinsicpns_converter",
    "inverse_extrinsicpns_converter",
    "inverse_intrinsicpns_converter",
]


def shape_calculator(operator):
    op = operator.raw_operator
    input_type = operator.inputs[0].type.__class__
    input_dim = operator.inputs[0].get_first_dimension()
    output_type = input_type([input_dim, op.n_components])
    operator.outputs[0].type = output_type


def intrinsicpns_converter(scope, operator, container):
    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs

    X = operator.inputs[0]

    d = X.type.shape[1] - 1
    residuals = []

    # Get the dtype from the operator's stored arrays
    sin_r = np.array(1.0, dtype=np.float32)
    for k in range(1, d):
        v, r = op.v_[k - 1], op.r_[k - 1].reshape(1)
        P, xi = onnx_project(X, v, r, opv)
        X = onnx_embed(P, v, r, opv)
        Xi = OnnxMul(sin_r, xi, op_version=opv)
        residuals.append(Xi)
        sin_r = (sin_r * np.sin(r)).astype(np.float32)

    v, r = op.v_[d - 1], op.r_[d - 1].reshape(1)
    _, xi = onnx_project(X, v, r, opv)
    Xi = OnnxMul(sin_r, xi, op_version=opv)
    residuals.append(Xi)

    ret = list(reversed(residuals))[: op.n_components]
    ret = OnnxConcat(*ret, axis=-1, op_version=opv, output_names=out[:1])
    ret.add_to(scope, container)


def onnx_project(X, v, r, opv, outnames=None):
    if v.shape[0] > 2:
        rho = OnnxAcos(
            OnnxMatMul(X, v.reshape(-1, 1), op_version=opv),
            op_version=opv,
        )  # (N, 1)
    else:
        # For 2D case (circle), use arctan2 to preserve sign
        rotation_matrix = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        rotated_v = v @ rotation_matrix
        y = OnnxMatMul(X, rotated_v.reshape(-1, 1), op_version=opv)  # (N, 1)
        x = OnnxMatMul(X, v.reshape(-1, 1), op_version=opv)  # (N, 1)
        rho = OnnxAtan2(y, x, op_version=opv)  # (N, 1)

    P = OnnxDiv(
        OnnxAdd(
            OnnxMul(np.sin(r).astype(np.float32), X, op_version=opv),  # (N, d+1)
            OnnxMul(
                OnnxSin(OnnxSub(rho, r, op_version=opv), op_version=opv),  # (N, 1)
                v,  # (d+1,)
                op_version=opv,
            ),  # (N, d+1)
        ),  # (N, d+1)
        OnnxSin(rho, op_version=opv),  # (N, 1)
        op_version=opv,
        output_names=outnames,
    )  # (N, d+1)
    return P, OnnxSub(rho, r, op_version=opv, output_names=outnames)


def onnx_inverse_project(xP, res, v, r, opv, outnames=None):
    rho = OnnxAdd(res, r, op_version=opv)
    numerator = OnnxSub(
        OnnxMul(xP, OnnxSin(rho, op_version=opv), op_version=opv),
        OnnxMul(OnnxSin(res, op_version=opv), v, op_version=opv),
        op_version=opv,
    )
    return OnnxDiv(
        numerator,
        np.sin(r).astype(np.float32),
        op_version=opv,
        output_names=outnames,
    )


def onnx_embed(x, v, r, opv, outnames=None):
    R = pnspy.base.rotation_matrix(v)
    coeff = (1 / np.sin(r) * R[:-1:, :]).T.astype(np.float32)
    ret = OnnxMatMul(
        x,
        coeff,
        op_version=opv,
        output_names=outnames,
    )
    return ret


def OnnxAtan2(y, x, op_version):
    """Implement atan2(y, x) using ONNX operations.

    Uses the formula: atan2(y, x) = 2 * atan(y / (sqrt(x^2 + y^2) + x))

    Parameters
    ----------
    y : OnnxOperator
        Y coordinate
    x : OnnxOperator
        X coordinate
    op_version : int
        ONNX opset version

    Returns
    -------
    OnnxOperator
        atan2(y, x) result
    """
    x_sq = OnnxMul(x, x, op_version=op_version)
    y_sq = OnnxMul(y, y, op_version=op_version)
    r_val = OnnxSqrt(OnnxAdd(x_sq, y_sq, op_version=op_version), op_version=op_version)
    numerator = y
    denominator = OnnxAdd(r_val, x, op_version=op_version)
    # Get dtype from the inputs if available, otherwise use default
    return OnnxMul(
        np.array(2.0, dtype=np.float32),
        OnnxAtan(
            OnnxDiv(numerator, denominator, op_version=op_version),
            op_version=op_version,
        ),
        op_version=op_version,
    )


def extrinsicpns_converter(scope, operator, container):
    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs

    X = operator.inputs[0]

    for v, r in zip(op.v_[:-1], op.r_[:-1]):
        v, r = v, r.reshape(1).astype(np.float32)
        P, _ = onnx_project(X, v, r, opv)
        X = onnx_embed(P, v, r, opv)
    v, r = op.v_[-1], op.r_[-1].reshape(1).astype(np.float32)
    P, _ = onnx_project(X, v, r, opv)
    X = onnx_embed(P, v, r, opv, out[:1])
    X.add_to(scope, container)


def onnx_full(arr, val, opv):
    N = OnnxGather(
        OnnxShape(arr, op_version=opv),
        np.array([0], dtype=np.int64),
        op_version=opv,
    )
    shape = OnnxConcat(N, np.array([1], dtype=np.int64), axis=0, op_version=opv)
    ones = OnnxConstantOfShape(
        shape, value=np.array([1.0], dtype=np.float32), op_version=opv
    )
    return OnnxMul(ones, val, op_version=opv)


def onnx_reconstruct(x, v, r, opv, outnames=None):
    R = pnspy.base.rotation_matrix(v).astype(np.float32)
    vec = OnnxConcat(
        OnnxMul(np.array(np.sin(r)), x, op_version=opv),
        onnx_full(x, np.array(np.cos(r)), opv),
        axis=1,
        op_version=opv,
    )
    return OnnxMatMul(vec, R, op_version=opv, output_names=outnames)


def inverse_extrinsicpns_converter(scope, operator, container):
    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs

    x = operator.inputs[0]
    for i, (v, r) in enumerate(zip(reversed(op.v_), reversed(op.r_))):
        r = r.astype(np.float32)
        if i < len(op.v_) - 1:
            x = onnx_reconstruct(x, v, r, opv)
        else:
            x = onnx_reconstruct(x, v, r, opv, out[:1])
    x.add_to(scope, container)


def inverse_intrinsicpns_converter(scope, operator, container):
    raise NotImplementedError
