import numpy as np
from pns.util import circular_data

from skpns import ExtrinsicPNS, IntrinsicPNS

np.random.seed(0)


def test_IntrinsicPNS_transform_noreduction():
    X = circular_data([0, -1, 0])
    pns = IntrinsicPNS()
    Xnew = pns.fit_transform(X)
    assert Xnew.shape[1] == X.shape[1] - 1


def test_ExtrinsicPNS_transform_noreduction():
    X = circular_data([0, -1, 0])
    pns = ExtrinsicPNS(n_components=X.shape[1])
    Xnew = pns.fit_transform(X)
    assert np.all(X == Xnew)
    assert np.all(X == pns.transform(X))
