from skpns import PNS
from skpns.util import circular_data
from skl2onnx import to_onnx


def test_to_onnx():
    X = circular_data()
    pns = PNS(n_components=2).fit(X)
    to_onnx(pns, X[:1])
