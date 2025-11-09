import numpy as np
import onnxruntime as rt
from skl2onnx import to_onnx

from skpns import ExtrinsicPNS, IntrinsicPNS
from skpns.util import circular_data


def test_IntrinsicPNS_onnx(tmp_path):
    path = tmp_path / "pns.onnx"

    X = circular_data().astype(np.float32)
    pns = IntrinsicPNS().fit(X)
    Xpred = pns.transform(X)

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

    X = circular_data().astype(np.float32)
    pns = ExtrinsicPNS().fit(X)
    Xpred = pns.transform(X)

    onx = to_onnx(pns, X[:1])
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())

    sess = rt.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    Xpred_onnx = sess.run([label_name], {input_name: X})[0]

    assert np.all(np.isclose(Xpred, Xpred_onnx, atol=1e-3))
