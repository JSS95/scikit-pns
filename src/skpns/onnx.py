"""Custom ONNX converter for PNS."""

import numpy as np
from skl2onnx.algebra.onnx_ops import (
    OnnxAcos,
    OnnxAdd,
    OnnxCos,
    OnnxDiv,
    OnnxEyeLike,
    OnnxMatMul,
    OnnxMul,
    OnnxReduceL2,
    OnnxReduceSum,
    OnnxReshape,
    OnnxSin,
    OnnxSlice,
    OnnxSub,
    OnnxTranspose,
    OnnxUnsqueeze,
)

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


def onnx_proj(x, v, r, op_version=None):
    v_reshaped = OnnxReshape(
        v, np.array([-1, 1], dtype=np.int64), op_version=op_version
    )  # (d+1, 1)
    dot = OnnxMatMul(x, v_reshaped, op_version=op_version)  # (N, 1)
    geod = OnnxAcos(dot, op_version=op_version)  # (N, 1)
    sin_geod = OnnxSin(geod, op_version=op_version)  # (N, 1)
    sin_r = OnnxSin(r, op_version=op_version)  # scalar
    term1 = OnnxMul(sin_r, x, op_version=op_version)  # (N, d+1)
    geod_minus_r = OnnxSub(geod, r, op_version=op_version)  # (N, 1) - scalar
    sin_geod_minus_r = OnnxSin(geod_minus_r, op_version=op_version)  # (N, 1)
    term2 = OnnxMul(sin_geod_minus_r, v, op_version=op_version)  # (N, d+1)
    numerator = OnnxAdd(term1, term2, op_version=op_version)  # (N, d+1)
    result = OnnxDiv(numerator, sin_geod, op_version=op_version)  # (N, d+1)
    return result


def onnx_to_unit_sphere(x, v, r, op_version=None):
    d = x.type.shape[1] - 1

    # a = [0, 0, ..., 1]
    a_np = np.zeros((d + 1,), dtype=np.float32)
    a_np[-1] = 1.0
    a = OnnxReshape(a_np, np.array([-1], dtype=np.int64), op_version=op_version)

    # a @ b (v)
    a_dot_b = OnnxReduceSum(
        OnnxMul(a, v, op_version=op_version),
        axes=[0],
        keepdims=0,
        op_version=op_version,
    )

    # a * (a @ b)
    a_mul_dot = OnnxMul(a, a_dot_b, op_version=op_version)

    # c = b - a * (a @ b)
    c = OnnxSub(v, a_mul_dot, op_version=op_version)

    # ||c||
    c_norm = OnnxReduceL2(c, axes=[0], keepdims=1, op_version=op_version)

    # c /= norm(c)
    c_unit = OnnxDiv(c, c_norm, op_version=op_version)

    # outer(a, c) - outer(c, a)
    a_unsq = OnnxUnsqueeze(
        a, np.array([1], dtype=np.int64), op_version=op_version
    )  # (d+1, 1)
    c_unsq = OnnxUnsqueeze(
        c_unit, np.array([0], dtype=np.int64), op_version=op_version
    )  # (1, d+1)

    outer_ac = OnnxMatMul(a_unsq, c_unsq, op_version=op_version)  # (d+1, d+1)
    outer_ca = OnnxMatMul(c_unsq, a_unsq, op_version=op_version)
    A = OnnxSub(outer_ac, outer_ca, op_version=op_version)

    # theta = arccos(v[-1])
    v_last = OnnxSlice(
        v, starts=[d], ends=[d + 1], axes=[0], op_version=op_version
    )  # (1,)
    theta = OnnxAcos(v_last, op_version=op_version)  # (1,)

    # Identity matrix
    eye = OnnxEyeLike(A, dtype=np.float32, op_version=op_version)

    # sin(theta) * A
    sin_theta = OnnxSin(theta, op_version=op_version)
    sin_theta_A = OnnxMul(sin_theta, A, op_version=op_version)

    # cos(theta) - 1
    cos_theta = OnnxCos(theta, op_version=op_version)
    cos_theta_minus_1 = OnnxSub(
        cos_theta, np.array([1.0], dtype=np.float32), op_version=op_version
    )

    # outer(a,a) + outer(c,c)
    outer_aa = OnnxMatMul(
        a_unsq, OnnxTranspose(a_unsq, op_version=op_version), op_version=op_version
    )
    outer_cc = OnnxMatMul(
        c_unsq, OnnxTranspose(c_unsq, op_version=op_version), op_version=op_version
    )
    sum_outer = OnnxAdd(outer_aa, outer_cc, op_version=op_version)

    # (cos(theta) - 1) * (outer(a,a) + outer(c,c))
    rot_part = OnnxMul(cos_theta_minus_1, sum_outer, op_version=op_version)

    # R = I + sin(theta)*A + ...
    R = OnnxAdd(
        OnnxAdd(eye, sin_theta_A, op_version=op_version),
        rot_part,
        op_version=op_version,
    )  # (d+1, d+1)

    # R[:-1, :]
    R_part = OnnxSlice(
        R, starts=[0], ends=[d], axes=[0], op_version=op_version
    )  # (d, d+1)

    # sin(r)
    sin_r = OnnxSin(r, op_version=op_version)  # scalar

    # 1 / sin(r)
    inv_sin_r = OnnxDiv(np.array([1.0], dtype=np.float32), sin_r, op_version=op_version)

    # R_part @ x.T
    x_T = OnnxTranspose(x, op_version=op_version)  # (d+1, N)
    Rx = OnnxMatMul(R_part, x_T, op_version=op_version)  # (d, N)

    # scale * Rx
    scaled = OnnxMul(inv_sin_r, Rx, op_version=op_version)  # (d, N)

    # Transpose back → (N, d)
    result = OnnxTranspose(scaled, op_version=op_version)  # (N, d)

    return result
