"""Functions for principal nested spheres analysis."""

import numpy as np
from scipy.optimize import least_squares

__all__ = [
    "pss",
    "proj",
    "embed",
    "to_unit_sphere",
    "reconstruct",
    "from_unit_sphere",
    "pns",
    "residual",
    "Exp",
    "Log",
]


def pss(x, tol=1e-3):
    """Find the principal subsphere from data on a hypersphere.

    Parameters
    ----------
    x : (N, d+1) real array
        Extrinsic coordinates of data on a ``d``-dimensional hypersphere,
        embedded in a ``d+1``-dimensional space.
    tol : float, default=1e-3
        Convergence tolerance in radian.

    Returns
    -------
    v : (d+1,) real array
        Estimated principal axis of the subsphere in extrinsic coordinates.
    r : scalar in [0, pi]
        Geodesic distance from the pole by *v* to the estimated principal subsphere.

    See Also
    --------
    proj : Project *x* onto the found principal subsphere.

    Examples
    --------
    >>> from skpns.pns import pss
    >>> from skpns.util import circular_data, unit_sphere
    >>> x = circular_data()
    >>> v, _ = pss(x)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... ax = plt.figure().add_subplot(projection='3d', computed_zorder=False)
    ... ax.plot_surface(*unit_sphere(), color='skyblue', alpha=0.6, edgecolor='gray')
    ... ax.scatter(*x.T, marker="x")
    ... ax.scatter(*v)
    """
    _, D = x.shape
    if D <= 1:
        raise ValueError("Data must be on at least 1-sphere.")
    elif D == 2:
        r = np.int_(0)
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
    return v.astype(x.dtype), r.astype(x.dtype)


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


def proj(x, v, r):
    """Minimum-geodesic projection of points to a subsphere.

    Parameters
    ----------
    x : (N, d+1) real array
        Extrinsic coordinates of data on a ``d``-dimensional hypersphere,
        embedded in a ``d+1``-dimensional space.
    v : (d+1,) real array
        Subsphere axis.
    r : scalar
        Subsphere geodesic distance.

    Returns
    -------
    A : (N, d+1) real array
        Extrinsic coordinates of data on a ``d``-dimensional hypersphere,
        projected onto the found principal subsphere.

    See Also
    --------
    pss : Find *v* and *r* for the principal subsphere.

    Examples
    --------
    >>> from skpns.pns import pss, proj
    >>> from skpns.util import circular_data, unit_sphere
    >>> x = circular_data()
    >>> A = proj(x, *pss(x))
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... ax = plt.figure().add_subplot(projection='3d', computed_zorder=False)
    ... ax.plot_surface(*unit_sphere(), color='skyblue', alpha=0.6, edgecolor='gray')
    ... ax.scatter(*x.T, marker="x")
    ... ax.scatter(*A.T, marker=".")
    """
    rho = np.arccos(x @ v)[..., np.newaxis]
    return (np.sin(r) * x + np.sin(rho - r) * v) / np.sin(rho)


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
    return R.astype(v.dtype)


def embed(x, v, r):
    """Embed data on a hypersphere to a sub-hypersphere.

    Parameters
    ----------
    x : (N, d+1) real array
        Data on a hypersphere.
    v : (d+1,) real array
        Sub-hypersphere axis.
    r : scalar
        Sub-hypersphere geodesic distance.

    Returns
    -------
    (N, d) real array
        Data on a sub-hypersphere.

    See Also
    --------
    pss : Find *v* and *r* for the principal subsphere.
    proj : Project data on a principal subsphere.
    reconstruct : Inverse operation of this function.

    Examples
    --------
    >>> from skpns.pns import pss, proj, embed
    >>> from skpns.util import circular_data, unit_sphere
    >>> x = circular_data()
    >>> v, r = pss(x)
    >>> A = proj(x, v, r)
    >>> x_low = embed(x, v, r)
    >>> A_low = embed(A, v, r)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... fig = plt.figure()
    ... ax1 = fig.add_subplot(121, projection='3d', computed_zorder=False)
    ... ax1.plot_surface(*unit_sphere(), color='skyblue', alpha=0.6, edgecolor='gray')
    ... ax1.scatter(*x.T, marker="x")
    ... ax1.scatter(*A.T, marker=".", zorder=10)
    ... ax2 = fig.add_subplot(122)
    ... ax2.scatter(*x_low.T, marker="x")
    ... ax2.scatter(*A_low.T, marker=".", zorder=10)
    ... ax2.set_aspect("equal")
    """
    R = _R(v)
    return x @ (1 / np.sin(r) * R[:-1:, :]).T


def to_unit_sphere(x, v, r):
    """alias of :func:`embed`."""
    return embed(x, v, r)


def reconstruct(x, v, r):
    """Reconstruct data on a sub-hypersphere to a hypersphere.

    Parameters
    ----------
    x : (N, d) real array
        Data on a sub-hypersphere.
    v : (d+1,) real array
        Sub-hypersphere axis.
    r : scalar
        Sub-hypersphere geodesic distance.

    Returns
    -------
    (N, d+1) real array
        Data on a hypersphere.

    See Also
    --------
    embed : Inverse operation of this function.

    Examples
    --------
    >>> from skpns.pns import reconstruct
    >>> from skpns.util import circular_data, unit_sphere
    >>> x = circular_data(dim=2)
    >>> v = np.array([1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)])
    >>> r = 0.15 * np.pi
    >>> x_high = reconstruct(x, v, r)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... fig = plt.figure()
    ... ax1 = fig.add_subplot(121)
    ... ax1.scatter(*x.T)
    ... ax1.set_aspect("equal")
    ... ax2 = fig.add_subplot(122, projection='3d', computed_zorder=False)
    ... ax2.plot_surface(*unit_sphere(), color='skyblue', alpha=0.6, edgecolor='gray')
    ... ax2.scatter(*x_high.T)
    """
    R = _R(v)
    vec = np.hstack([np.sin(r) * x, np.full(len(x), np.cos(r)).reshape(-1, 1)])
    return vec @ R


def from_unit_sphere(x, v, r):
    """alias of :func:`reconstruct`."""
    return reconstruct(x, v, r)


def pns(x, tol=1e-3):
    r"""Principal nested spheres analysis.

    Parameters
    ----------
    x : (N, d+1) real array
        Data on d-sphere :math:`S^d \subset \mathbb{R}^{d+1}`.
    tol : float, default=1e-3
        Convergence tolerance in radian.

    Yields
    ------
    v : (d+1-i,) real array
        Estimated principal axis :math:`\hat{v}`.
    r : scalar
        Estimated principal geodesic distance :math:`\hat{r}`.
    x : (N, d-i) real array
        Transformed data :math:`x^\dagger` on low-dimensional unit hypersphere.

    Notes
    -----
    Let :math:`k = 1, \ldots, d-1`.
    The entire :math:`\hat{v}` are

    .. math::

        \hat{v}_{1} \in S^{d} \subset \mathbb{R}^{d+1},\quad
        \ldots,\quad
        \hat{v}_{k} \in S^{d-k+1} \subset \mathbb{R}^{d-k+2},\quad
        \ldots,\quad
        \hat{v}_{d} \in S^{1} \subset \mathbb{R}^{2},

    the entire :math:`\hat{r}` are

    .. math::

        \hat{r}_{1},\quad
        \ldots,\quad
        \hat{r}_{d} \in \mathbb{R},

    and the transformed data are

    .. math::

        x^\dagger_{1} \in S^{d-1} \subset \mathbb{R}^{d},\quad
        \ldots,\quad
        x^\dagger_{k} \in S^{d-k} \subset \mathbb{R}^{d-k+1},\quad
        \ldots,\quad
        x^\dagger_{d} \subset \mathbb{R},

    Note that the results from the last iteration are specially handled,
    as described in the original paper.

    Examples
    --------
    >>> from skpns.pns import pns, reconstruct
    >>> from skpns.util import circular_data, unit_sphere, circle
    >>> x = circular_data()
    >>> pns_gen = pns(x)
    >>> v1, r1, A1 = next(pns_gen)
    >>> v2, r2, A2 = next(pns_gen)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... ax = plt.figure().add_subplot(projection='3d', computed_zorder=False)
    ... ax.plot_surface(*unit_sphere(), color='skyblue', alpha=0.6, edgecolor='gray')
    ... ax.scatter(*x.T, marker=".")
    ... ax.scatter(*reconstruct(A1, v1, r1).T, marker="x")
    ... ax.scatter(*reconstruct(reconstruct(A2, v2, r2), v1, r1).T, zorder=10)
    ... ax.plot(*circle(v1, r1), color="tab:red")
    """
    d = x.shape[1] - 1

    for _ in range(1, d):
        v, r = pss(x, tol)
        A = proj(x, v, r)
        x = embed(A, v, r)
        yield v, r, x

    v, r = pss(x, tol)
    x = np.full((len(x), 1), 0, dtype=x.dtype)
    yield v, r, x


def residual(x, v, r):
    """Signed residuals caused by projecting data to a subsphere.

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
        Signed residuals.

    Examples
    --------
    >>> from skpns.pns import pns, reconstruct, residual
    >>> from skpns.util import circular_data, unit_sphere, circle
    >>> x = circular_data()
    >>> pns_gen = pns(x)
    >>> v1, r1, A1 = next(pns_gen)
    >>> res1 = residual(x, v1, r1)
    >>> v2, r2, A2 = next(pns_gen)
    >>> res2 = residual(A1, v2, r2)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... fig = plt.figure()
    ... ax1 = fig.add_subplot(121, projection='3d', computed_zorder=False)
    ... ax1.plot_surface(*unit_sphere(), color='skyblue', alpha=0.6, edgecolor='gray')
    ... ax1.scatter(*x.T, c=res2)
    ... ax1.plot(*circle(v1, r1), color="tab:red")
    ... ax1.scatter(*reconstruct(reconstruct(A2, v2, r2), v1, r1).T, color="tab:red")
    ... ax2 = fig.add_subplot(122)
    ... ax2.set_xlim(-np.pi/2, np.pi/2)
    ... ax2.set_ylim(-np.pi, np.pi)
    ... ax2.scatter(res1, res2, c=res2)
    ... ax2.axvline(0, color="tab:red")
    ... ax2.scatter(0, 0, color="tab:red")
    """
    _, D = x.shape
    if D <= 1:
        raise ValueError("Data must be on at least 1-sphere.")
    elif D == 2:
        xi = np.arctan2(x @ (v @ [[0, 1], [-1, 0]]), x @ v)
    else:
        rho = np.arccos(np.dot(x, v.T))
        xi = rho - r
    return xi


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
