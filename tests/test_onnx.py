import onnxruntime as rt
from skl2onnx import to_onnx

from skpns import PNS
from skpns.util import circular_data


def test_onnx(tmp_path):
    path = tmp_path / "pns.onnx"

    X = circular_data()
    pns = PNS(n_components=2).fit(X)
    onx = to_onnx(pns, X[:1])
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())

    sess = rt.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    sess.run([label_name], {input_name: X})[0]
