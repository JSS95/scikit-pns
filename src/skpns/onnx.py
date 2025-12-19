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

from .pns import _R

__all__ = [
    "shape_calculator",
    "intrinsicpns_converter",
    "extrinsicpns_converter",
    "inverse_extrinsicpns_converter",
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
        P, xi = onnx_proj(X, v, r, opv)
        X = onnx_embed(P, v, r, opv)
        Xi = OnnxMul(sin_r, xi, op_version=opv)
        residuals.append(Xi)
        sin_r = (sin_r * np.sin(r)).astype(np.float32)

    v, r = op.v_[d - 1], op.r_[d - 1].reshape(1)
    _, xi = onnx_proj(X, v, r, opv)
    Xi = OnnxMul(sin_r, xi, op_version=opv)
    residuals.append(Xi)

    ret = list(reversed(residuals))[: op.n_components]
    ret = OnnxConcat(*ret, axis=-1, op_version=opv, output_names=out[:1])
    ret.add_to(scope, container)


def onnx_proj(X, v, r, opv, outnames=None):
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


def onnx_embed(x, v, r, opv, outnames=None):
    R = _R(v)
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

    proj = pnspy.transform.Project(
        lambda a, b: OnnxAdd(a, b, op_version=opv),
        lambda a, b: OnnxSub(a, b, op_version=opv),
        lambda a, b: OnnxMul(a, b, op_version=opv),
        lambda a, b: OnnxDiv(a, b, op_version=opv),
        lambda a: OnnxSin(a, op_version=opv),
        lambda a: OnnxAcos(a, op_version=opv),
        lambda y, x: OnnxAtan2(y, x, op_version=opv),
        lambda a, b: OnnxMatMul(a, b, op_version=opv),
        dtype=np.float32,
    )
    embed = pnspy.transform.Embed(
        lambda a, b, **kwargs: OnnxMatMul(a, b, op_version=opv, **kwargs),
        dtype=np.float32,
    )
    # reconstruction function will not be used in this converter so pass None.
    extrinsic_pns = pnspy.ExtrinsicPNS(proj, embed, None, dtype=np.float32)

    X = operator.inputs[0]
    x = extrinsic_pns(X, op.v_, op.r_, dict(output_names=out[:1]))
    x.add_to(scope, container)


def inverse_extrinsicpns_converter(scope, operator, container):
    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs

    def onn_full(arr, val):
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

    reconstruct = pnspy.transform.Reconstruct(
        lambda a, b: OnnxMul(a, b, op_version=opv),
        lambda a: np.sin([a], dtype=np.float32),
        lambda a: np.cos([a], dtype=np.float32),
        onn_full,
        lambda args: OnnxConcat(*args, axis=1, op_version=opv),
        lambda a, b, **kwargs: OnnxMatMul(a, b, op_version=opv, **kwargs),
        np.float32,
    )
    # will only use reconstruction, so pass None to other arguments
    extrinsic_pns = pnspy.ExtrinsicPNS(None, None, reconstruct, dtype=np.float32)

    X = operator.inputs[0]
    x = extrinsic_pns.inverse(X, op.v_, op.r_, dict(output_names=out[:1]))
    x.add_to(scope, container)
