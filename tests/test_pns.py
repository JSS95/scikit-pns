import numpy as np

from skpns import PNS
from skpns.util import circular_data

np.random.seed(0)


def test_PNS_transform():
    X = circular_data()
    pns = PNS(n_components=2)
    Xnew_1 = pns.fit_transform(X)
    Xnew_2 = pns.transform(X)
    assert np.all(Xnew_1 == Xnew_2)


def test_PNS_transform_noreduction():
    X = circular_data()
    pns = PNS(n_components=X.shape[1])
    Xnew = pns.fit_transform(X)
    assert np.all(X == Xnew)
    assert np.all(X == pns.transform(X))
