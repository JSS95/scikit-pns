"""
PNS
---

Functions to find principal nested spheres.
"""

import numpy as np
from scipy.optimize import least_squares

__all__ = [
    "pns",
    "pss",
    "proj",
    "residual",
    "to_unit_sphere",
    "from_unit_sphere",
    "Exp",
    "Log",
]


def pns(x, tol=1e-3):
    """Principal nested spheres analysis.

    Parameters
    ----------
    x : (N, d+1) real array
        Data on d-sphere.
    tol : float, default=1e-3
        Convergence tolerance in radian.

    Yields
    ------
    v : 1-D ndarray
        Principal axis.
    r : scalar
        Principal geodesic distance.
    x : (N, d-i) real array
        Data transformed onto low-dimensional unit hypersphere.

    Examples
    --------
    >>> from skpns.pns import pns, from_unit_sphere
    >>> from skpns.util import circular_data, unit_sphere, circle
    >>> x = circular_data()
    >>> v, r, A = next(pns(x))
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... ax = plt.figure().add_subplot(projection='3d', computed_zorder=False)
    ... ax.plot_surface(*unit_sphere(), color='skyblue', alpha=0.6, edgecolor='gray')
    ... ax.scatter(*x.T, marker=".")
    ... ax.scatter(*from_unit_sphere(A, v, r).T, marker="x")
    ... ax.plot(*circle(v, r), color="tab:red")
    """
    d = x.shape[1] - 1

    for _ in range(1, d):
        v, r = pss(x, tol)
        A = proj(x, v, r)
        x = to_unit_sphere(A, v, r)
        yield v, r, x

    v, r = pss(x, tol)
    x = np.full((len(x), 1), 0)
    yield v, r, x


def pss(x, tol=1e-3):
    """Find the principal subsphere, i.e., lower the dimension by one.

    Parameters
    ----------
    x : (N, d+1) real array
        Data on d-sphere.
    tol : float, default=1e-3
        Convergence tolerance in radian.

    Returns
    -------
    v : (d+1,) real array
        Principal axis.
    r : scalar
        Principal geodesic distance.
    """
    _, D = x.shape
    if D <= 1:
        raise ValueError("Data must be on at least 1-sphere.")
    elif D == 2:
        r = 0
        v = np.mean(x, axis=0)
        v /= np.linalg.norm(v)
    else:
        pole = np.array([0] * (D - 1) + [1])
        R = np.eye(D)
        _x = x
        v, r = _pss(_x)
        while np.arccos(np.dot(pole, v)) > tol:
            # Rotate so that v becomes the pole
            _x, _R = _rotate(_x, v)
            v, r = _pss(_x)
            R = R @ _R.T
        v = R @ v  # re-rotate back
    return v, r


def proj(x, v, r):
    """Minimum-geodesic projection of points to subsphere.

    Parameters
    ----------
    x : (N, d+1) real array
        Data on unit d-sphere.
    v : (d+1,) real array
        Subsphere axis.
    r : scalar
        Subsphere geodesic distance.

    Returns
    -------
    A : (N, d+1) real array
        Points projected onto the small-subsphere on unit d-sphere.
    """
    geod = np.arccos(x @ v)[..., np.newaxis]
    return (np.sin(r) * x + np.sin(geod - r) * v) / np.sin(geod)


def residual(x, v, r):
    """Signed residuals of dimension reduction to subsphere.

    Parameters
    ----------
    x : (N, d+1) real array
        Data on d-sphere.
    v : (d+1,) real array
        Subsphere axis.
    r : scalar
        Subsphere geodesic distance.

    Returns
    -------
    xi : (N,) array
        Signed residual.
    """
    _, D = x.shape
    if D <= 1:
        raise ValueError("Data must be on at least 1-sphere.")
    elif D == 2:
        xi = np.arctan2(x @ (v @ [[0, 1], [-1, 0]]), x @ v)
    else:
        xi = np.arccos(np.dot(x, v.T)) - r
    return xi


def _R(v):
    a = np.zeros_like(v)
    a[-1] = 1.0
    b = v
    c = b - a * (a @ b)
    c /= np.linalg.norm(c)

    A = np.outer(a, c) - np.outer(c, a)
    theta = np.arccos(v[-1])
    Id = np.eye(len(A))
    R = Id + np.sin(theta) * A + (np.cos(theta) - 1) * (np.outer(a, a) + np.outer(c, c))
    return R


def to_unit_sphere(x, v, r):
    """Transform projected data on subsphere to unit hypersphere.

    Parameters
    ----------
    x : (N, d+1) real array
        Data projected on subsphere.
    v : (d+1,) real array
        Subsphere axis.
    r : scalar
        Subsphere geodesic distance.

    Returns
    -------
    (N, d) real array
        Data points on unit hypersphere.
    """
    R = _R(v)
    return x @ (1 / np.sin(r) * R[:-1:, :]).T


def from_unit_sphere(x, v, r):
    """Transform data on unit hypersphere to projected data on subsphere.

    Parameters
    ----------
    x : (N, d) real array
        Data points on unit hypersphere.
    v : (d+1,) real array
        Subsphere axis.
    r : scalar
        Subsphere geodesic distance.

    Returns
    -------
    (N, d+1) real array
        Data projected on subsphere.
    """
    R = _R(v)
    vec = np.hstack([np.sin(r) * x, np.full(len(x), np.cos(r)).reshape(-1, 1)])
    return (R.T @ vec.T).T


def _pss(pts):
    # Projection
    x_dag = Log(pts)
    v_dag_init = np.mean(x_dag, axis=0)
    r_init = np.mean(np.linalg.norm(x_dag - v_dag_init, axis=1))
    init = np.concatenate([v_dag_init, [r_init]])
    # Optimization
    opt = least_squares(_loss, init, args=(x_dag,), method="lm").x
    v_dag_opt, r_opt = opt[:-1], opt[-1]
    v_opt = Exp(v_dag_opt.reshape(1, -1)).reshape(-1)
    r_opt = np.mod(r_opt, np.pi)
    return v_opt, r_opt


def _loss(params, x_dag):
    v_dag, r = params[:-1], params[-1]
    return np.linalg.norm(x_dag - v_dag.reshape(1, -1), axis=1) - r


def _rotate(pts, v):
    R = _R(v)
    return (R @ pts.T).T, R


def Exp(z):
    """Exponential map of hypersphere at (0, 0, ..., 0, 1).

    Parameters
    ----------
    z : (N, d) real array
        Vectors on tangent space.

    Returns
    -------
    (N, d+1) real array
        Points on d-sphere.
    """
    norm = np.linalg.norm(z, axis=1)[..., np.newaxis]
    return np.hstack([np.sin(norm) / norm * z, np.cos(norm)])


def Log(x):
    """Log map of hypersphere at (0, 0, ..., 0, 1).

    Parameters
    ----------
    x : (N, d+1) real array
        Points on d-sphere.

    Returns
    -------
    (N, d) real array
        Vectors on tangent space.
    """
    thetas = np.arccos(x[:, -1:])
    return thetas / np.sin(thetas) * x[:, :-1]
