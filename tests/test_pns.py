import numpy as np
import warnings

from skpns import ExtrinsicPNS, IntrinsicPNS
from skpns.pns import pss, pns
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


def test_pss_maxiter_none():
    """Test that pss works with maxiter=None (default behavior)."""
    X = circular_data()
    v1, r1 = pss(X)
    v2, r2 = pss(X, maxiter=None)
    assert np.allclose(v1, v2)
    assert np.isclose(r1, r2)


def test_pss_maxiter_warning():
    """Test that pss raises warning when maxiter is reached."""
    X = circular_data()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v, r = pss(X, maxiter=1)
        # Check that a warning was raised
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "Maximum number of iterations" in str(w[-1].message)


def test_pss_maxiter_no_warning():
    """Test that pss doesn't raise warning when converged before maxiter."""
    X = circular_data()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v, r = pss(X, maxiter=1000)
        # Check that no warning was raised
        assert len(w) == 0


def test_ExtrinsicPNS_with_maxiter():
    """Test that ExtrinsicPNS works with maxiter parameter."""
    X = circular_data()
    pns_model = ExtrinsicPNS(n_components=2, maxiter=1000)
    Xnew = pns_model.fit_transform(X)
    assert Xnew.shape[1] == 2


def test_IntrinsicPNS_with_maxiter():
    """Test that IntrinsicPNS works with maxiter parameter."""
    X = circular_data()
    pns_model = IntrinsicPNS(n_components=2, maxiter=1000)
    Xnew = pns_model.fit_transform(X)
    assert Xnew.shape[1] == 2


def test_pns_generator_with_maxiter():
    """Test that pns generator works with maxiter parameter."""
    X = circular_data()
    pns_gen = pns(X, maxiter=1000)
    v1, r1, xd1 = next(pns_gen)
    assert v1 is not None
    assert r1 is not None
    assert xd1 is not None
