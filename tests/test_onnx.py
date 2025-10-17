from skl2onnx import to_onnx

from skpns import PNS
from skpns.util import circular_data


def test_to_onnx():
    X = circular_data()
    pns = PNS(n_components=2).fit(X)
    to_onnx(pns, X[:1])
