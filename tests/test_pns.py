import numpy as np

from skpns import ExtrinsicPNS, IntrinsicPNS
from skpns.util import circular_data

np.random.seed(0)


def test_IntrinsicPNS_transform():
    X = circular_data()
    pns = IntrinsicPNS(n_components=2)
    Xnew_1 = pns.fit_transform(X)
    Xnew_2 = pns.transform(X)
    assert np.all(Xnew_1 == Xnew_2)


def test_IntrinsicPNS_transform_noreduction():
    X = circular_data()
    pns = IntrinsicPNS()
    Xnew = pns.fit_transform(X)
    assert Xnew.shape[1] == X.shape[1] - 1


def test_IntrinsicPNS_inverse_transform():
    X = circular_data()
    pns = IntrinsicPNS()
    Xinv = pns.inverse_transform(pns.fit_transform(X))
    assert np.all(np.isclose(X, Xinv, atol=1e-3))


def test_ExtrinsicPNS_transform():
    X = circular_data()
    pns = ExtrinsicPNS(n_components=2)
    Xnew_1 = pns.fit_transform(X)
    Xnew_2 = pns.transform(X)
    assert np.all(Xnew_1 == Xnew_2)


def test_ExtrinsicPNS_transform_noreduction():
    X = circular_data()
    pns = ExtrinsicPNS(n_components=X.shape[1])
    Xnew = pns.fit_transform(X)
    assert np.all(X == Xnew)
    assert np.all(X == pns.transform(X))
