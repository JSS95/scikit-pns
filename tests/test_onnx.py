import numpy as np
import onnxruntime as rt
from pns.util import circular_data
from skl2onnx import to_onnx

from skpns import ExtrinsicPNS, IntrinsicPNS, InverseExtrinsicPNS, InverseIntrinsicPNS


def test_IntrinsicPNS_onnx(tmp_path):
    path = tmp_path / "pns.onnx"

    pns = IntrinsicPNS()
    X = circular_data([0, -1, 0]).astype(np.float32)
    Xpred = pns.fit_transform(X)

    onx = to_onnx(pns, X[:1])
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())

    sess = rt.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    Xpred_onnx = sess.run([label_name], {input_name: X})[0]

    assert np.all(np.isclose(Xpred, Xpred_onnx, atol=1e-3))


def test_ExtrinsicPNS_onnx(tmp_path):
    path = tmp_path / "pns.onnx"

    pns = ExtrinsicPNS()
    X = circular_data([0, -1, 0]).astype(np.float32)
    Xpred = pns.fit_transform(X)

    onx = to_onnx(pns, X[:1])
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())

    sess = rt.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    Xpred_onnx = sess.run([label_name], {input_name: X})[0]

    assert np.all(np.isclose(Xpred, Xpred_onnx, atol=1e-3))


def test_InverseExtrinsicPNS_onnx(tmp_path):
    path = tmp_path / "pns.onnx"

    pns = ExtrinsicPNS()
    X = pns.fit_transform(circular_data([0, -1, 0])).astype(np.float32)
    Xpred = pns.inverse_transform(X)

    onx = to_onnx(InverseExtrinsicPNS(pns), X[:1])
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())

    sess = rt.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    Xpred_onnx = sess.run([label_name], {input_name: X})[0]

    assert np.all(np.isclose(Xpred, Xpred_onnx, atol=1e-3))


def test_InverseIntrinsicPNS_onnx(tmp_path):
    path = tmp_path / "pns.onnx"

    pns = IntrinsicPNS()
    X = pns.fit_transform(circular_data([0, -1, 0])).astype(np.float32)
    Xpred = pns.inverse_transform(X)

    onx = to_onnx(InverseIntrinsicPNS(pns), X[:1])
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())

    sess = rt.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    Xpred_onnx = sess.run([label_name], {input_name: X})[0]

    assert np.all(np.isclose(Xpred, Xpred_onnx, atol=1e-3))
