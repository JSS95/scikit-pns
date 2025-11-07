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
    "residual",
    "pns",
    "Exp",
    "Log",
]


def pss(x, tol=1e-3):
    r"""Find the principal subsphere from data on a hypersphere.

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

    Notes
    -----
    This function determines the best fitting subsphere
    :math:`\hat{A}_{d-k} = A_{d-k}(\hat{v}_k, \hat{r}_k) \subset S^{d-k+1}` for
    :math:`k = 1, 2, \ldots, d`.

    The Fréchet mean :math:`\hat{A}_0` of the lowest level best fitting subsphere
    :math:`\hat{A}_1` is also determined by this function.

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
    r"""Minimum-geodesic projection of points to a subsphere.

    Parameters
    ----------
    x : (N, m+1) real array
        Extrinsic coordinates of data on a ``d``-dimensional hypersphere,
        embedded in a ``d+1``-dimensional space.
    v : (m+1,) real array
        Subsphere axis.
    r : scalar
        Subsphere geodesic distance.

    Returns
    -------
    xP : (N, m+1) real array
        Extrinsic coordinates of data on a ``d``-dimensional hypersphere,
        projected onto the found principal subsphere.

    See Also
    --------
    pss : Find *v* and *r* for the principal subsphere.
    embed : Reduce the number of components of the projected data by one.

    Notes
    -----
    This is the function
    :math:`P: S^{d-k+1} \to A_{d-k}(v_k, r_k ) \subset S^{d-k+1}` for
    :math:`k = 1, 2, \ldots, d-1` in the original paper.
    Here, :math:`A_{d-k}(v_k, r_k)` is a subsphere of the hypersphere :math:`S^{d-k+1}`.
    The input and output data dimension are :math:`m+1`, where :math:`m = d-k+1`.

    Note that the projection at :math:`k=d` is not supported by this function.
    This projection is trivial, as it collapses all points in :math:`S^1` to the
    Fréchet mean.

    This function projects the data onto any subsphere. To project to the principal
    subsphere :math:`\hat{A}_{d-k} = A_{d-k}(\hat{v}_k, \hat{r}_k)`, pass the results
    from :func:`pss`.

    The resulting points have same number of components but their rank is reduced
    by one in the manifold. Use :func:`embed` to further map
    :math:`x \in A_{d-k}(v_k, r_k) \subset S^{d-k+1}` to :math:`x^\dagger \in S^{d-k}`.

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
    r"""Embed data on a sub-hypersphere to a low-dimensional unit hypersphere.

    Parameters
    ----------
    x : (N, m+1) real array
        Data :math:`x \in A_{m-1} \subset S^m \subset \mathbb{R}^{m+1}`,
        on a subsphere :math:`A_{m-1}` of a unit hypersphere :math:`S^m`.
    v : (m+1,) real array
        Sub-hypersphere axis.
    r : scalar
        Sub-hypersphere geodesic distance.

    Returns
    -------
    (N, m) real array
        Data :math:`x^\dagger` on a low-dimensional unit hypersphere :math:`S^{m-1}`.

    See Also
    --------
    pss : Find *v* and *r* for the principal subsphere.
    proj : Project data on a principal subsphere.
    reconstruct : Inverse operation of this function.

    Notes
    -----
    This is the function
    :math:`f_k: A_{d-k}(v_k, r_k) \subset S^{d-k+1} \to S^{d-k}` for
    :math:`k = 1, 2, \ldots, d-1` in the original paper.
    Here, :math:`A_{d-k}(v_k, r_k)` is a subsphere of the hypersphere :math:`S^{d-k+1}`.
    The input is :math:`x \in S^m \subset \mathbb{R}^{m+1}`
    and the output is :math:`x^\dagger \in S^{m-1} \subset \mathbb{R}^{m}`,
    where :math:`m = d-k+1`.

    Examples
    --------
    >>> from skpns.pns import pss, proj, embed
    >>> from skpns.util import circular_data, unit_sphere
    >>> x = circular_data()
    >>> v, r = pss(x)
    >>> A = proj(x, v, r)
    >>> A_low = embed(A, v, r)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... fig = plt.figure()
    ... ax1 = fig.add_subplot(121, projection='3d', computed_zorder=False)
    ... ax1.plot_surface(*unit_sphere(), color='skyblue', alpha=0.6, edgecolor='gray')
    ... ax1.scatter(*A.T, marker=".", zorder=10)
    ... ax2 = fig.add_subplot(122)
    ... ax2.scatter(*A_low.T, marker=".", zorder=10)
    ... ax2.set_aspect("equal")
    """
    R = _R(v)
    return x @ (1 / np.sin(r) * R[:-1:, :]).T


def to_unit_sphere(x, v, r):
    """alias of :func:`embed`."""
    return embed(x, v, r)


def reconstruct(x, v, r):
    r"""Reconstruct data on a low-dimensional unit hypersphere. to a sub-hypersphere.

    Parameters
    ----------
    x : (N, m) real array
        Data :math:`x^\dagger` on a low-dimensional unit hypersphere :math:`S^{m-1}`.
    v : (m+1,) real array
        Sub-hypersphere axis.
    r : scalar
        Sub-hypersphere geodesic distance.

    Returns
    -------
    (N, m+1) real array
        Data :math:`x \in A_{m-1} \subset S^m \subset \mathbb{R}^{m+1}`,
        on a subsphere :math:`A_{m-1}` of a unit hypersphere :math:`S^m`.

    See Also
    --------
    embed : Inverse operation of this function.

    Notes
    -----
    This is the function
    :math:`f^{-1}_k: S^{d-k} \to A_{d-k}(v_k, r_k) \subset S^{d-k+1}` for
    :math:`k = 1, 2, \ldots, d-1` in the original paper.
    Here, :math:`A_{d-k}(v_k, r_k)` is a subsphere of the hypersphere :math:`S^{d-k+1}`.
    The input is :math:`x^\dagger \in S^{m-1} \subset \mathbb{R}^{m}`
    and the output is :math:`x \in S^m \subset \mathbb{R}^{m+1}`,
    where :math:`m = d-k+1`.

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


def residual(x, v, r):
    r"""Signed residuals caused by projecting data to a subsphere.

    Parameters
    ----------
    x : (N, m+1) real array
        Data on m-sphere.
    v : (m+1,) real array
        Subsphere axis.
    r : scalar
        Subsphere geodesic distance.

    Returns
    -------
    xi : (N,) array
        Signed residuals.

    Notes
    -----
    This is the signed unscaled residual :math:`\xi = \rho(x, v) - r`
    in the original paper.

    Let :math:`k = 1, \ldots, d`.
    The inputs are :math:`x \in S^{d-k+1}`, :math:`v_k \in S^{d-k+1}` and
    :math:`r_k \in \mathbb{R}`, and the output is :math:`\xi_{d-k}`.

    Examples
    --------
    >>> from skpns.pns import residual
    >>> from skpns.util import circular_data, unit_sphere, circle
    >>> x = circular_data()
    >>> v = np.array([1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)])
    >>> r = 0.15 * np.pi
    >>> res = residual(x, v, r)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... fig = plt.figure()
    ... ax1 = fig.add_subplot(121, projection='3d', computed_zorder=False)
    ... ax1.plot_surface(*unit_sphere(), color='skyblue', alpha=0.6, edgecolor='gray')
    ... ax1.scatter(*x.T, c=res)
    ... ax1.plot(*circle(v, r), color="tab:red")
    ... ax2 = fig.add_subplot(122)
    ... ax2.scatter(np.arange(len(res)), res, c=res)
    ... ax2.axhline(0, color="tab:red")
    """
    _, D = x.shape
    if D <= 1:
        raise ValueError("Data must be on at least 1-sphere.")
    elif D == 2:
        rho = np.arctan2(x @ (v @ [[0, 1], [-1, 0]]), x @ v)
    else:
        rho = np.arccos(np.dot(x, v.T))
    return rho - r


def pns(x, tol=1e-3, residual="none"):
    r"""Principal nested spheres analysis.

    Parameters
    ----------
    x : (N, d+1) real array
        Data on a d-sphere.
    tol : float, default=1e-3
        Convergence tolerance in radians.
    residual : {'none', 'scaled', 'unscaled'}
        If 'none', do not yield residuals.
        If 'scaled', yield scaled residuals :math:`\Xi`.
        If 'unscaled', yield unscaled residuals :math:`\xi`.

    Yields
    ------
    v : (d+1-i,) real array
        Estimated principal axis :math:`\hat{v}`.
    r : scalar
        Estimated principal geodesic distance :math:`\hat{r}`.
    xd : (N, d-i) real array
        Transformed data :math:`x^\dagger` on low-dimensional unit hypersphere.
    res : (N,) real array
        Residuals. See the description of parameter *residual*.

    See Also
    --------
    reconstruct : Reconstruct the transformed data onto higher-dimensional spheres.

    Notes
    -----
    The input data is :math:`x \in S^d \subset \mathbb{R}^{d+1}`.

    At :math:`k`-th iteration for :math:`k=1, \ldots, d`, this generator yields:

    1. The principal axis :math:`\hat{v}_{k} \in S^{d-k+1} \subset \mathbb{R}^{d-k+2}`,
    2. The principal geodesic distance :math:`\hat{r}_k \in \mathbb{R}`, and
    3. The embedded data :math:`x_k^\dagger \in S^{d-k} \subset \mathbb{R}^{d-k+1}`.
    4. (Optional) Scaled residual :math:`\Xi(d-k)`,
       or unscaled residual :math:`\xi_{d-k}`.

    Data projected onto each principal nested sphere in the original space,
    :math:`\hat{\mathfrak{A}}_{d-k} \subset S^d`,
    can be found by recursively calling :func:`reconstruct` on :math:`x_k^\dagger`.

    Examples
    --------
    Use :func:`reconstruct` to map reduced data onto the original sphere.

    .. plot::
        :include-source:
        :context: reset

        >>> from skpns.pns import pns, reconstruct
        >>> from skpns.util import circular_data, unit_sphere, circle
        >>> x = circular_data()
        >>> pns_gen = pns(x, residual="none")
        >>> v1, r1, xd1 = next(pns_gen)
        >>> v2, r2, xd2 = next(pns_gen)
        >>> import matplotlib.pyplot as plt  # doctest: +SKIP
        ... ax = plt.figure().add_subplot(projection='3d', computed_zorder=False)
        ... ax.plot_surface(*unit_sphere(), color='skyblue', edgecolor='gray')
        ... ax.scatter(*x.T)
        ... ax.scatter(*reconstruct(xd1, v1, r1).T, marker="x")
        ... ax.scatter(*reconstruct(reconstruct(xd2, v2, r2), v1, r1).T, zorder=10)
        ... ax.plot(*circle(v1, r1), color="tab:red")

    Unscaled residuals do not distinguish the scale of the data on the original sphere.

    .. plot::
        :include-source:
        :context: close-figs

        >>> X = circular_data(scale="large")
        >>> (V1, R1, XD1, XI1), (V2, R2, XD2, XI2) = list(pns(X, residual="unscaled"))
        >>> x = circular_data(scale="small")
        >>> (v1, r1, xd1, xi1), (v2, r2, xd2, xi2) = list(pns(x, residual="unscaled"))
        >>> fig = plt.figure()  # doctest: +SKIP
        ... ax1 = fig.add_subplot(121, projection='3d', computed_zorder=False)
        ... ax1.plot_surface(*unit_sphere(), color='skyblue', edgecolor='gray')
        ... ax1.scatter(*X.T)
        ... ax1.scatter(*x.T)
        ... ax2 = fig.add_subplot(122)
        ... ax2.scatter(XI2, XI1)
        ... ax2.scatter(xi2, xi1)
        ... ax2.set_xlim(-np.pi, np.pi)
        ... ax2.set_ylim(-np.pi/2, np.pi/2)

    Scaled residuals distinguish different arc-lengths of principal subspheres.

    .. plot::
        :include-source:
        :context: close-figs

        >>> X = circular_data(scale="large")
        >>> (V1, R1, XD1, XI1), (V2, R2, XD2, XI2) = list(pns(X, residual="scaled"))
        >>> x = circular_data(scale="small")
        >>> (v1, r1, xd1, xi1), (v2, r2, xd2, xi2) = list(pns(x, residual="scaled"))
        >>> fig = plt.figure()  # doctest: +SKIP
        ... ax1 = fig.add_subplot(121, projection='3d', computed_zorder=False)
        ... ax1.plot_surface(*unit_sphere(), color='skyblue', edgecolor='gray')
        ... ax1.scatter(*X.T)
        ... ax1.scatter(*x.T)
        ... ax2 = fig.add_subplot(122)
        ... ax2.scatter(XI2, XI1)
        ... ax2.scatter(xi2, xi1)
        ... ax2.set_xlim(-np.pi, np.pi)
        ... ax2.set_ylim(-np.pi/2, np.pi/2)
    """
    d = x.shape[1] - 1

    sin_r = 1
    for _ in range(1, d):  # k=1, ..., (d-1)
        v, r = pss(x, tol)  # v_k, r_k
        rho = np.arccos(x @ v)[..., np.newaxis]
        P = (np.sin(r) * x + np.sin(rho - r) * v) / np.sin(rho)  # points on A_{d-k}
        x_dagger = embed(P, v, r)

        xi = rho - r  # xi_{d-k}
        Xi = sin_r * xi  # Xi(d-k), i.e., Xi(d-1), ..., Xi(1)

        if residual == "none":
            ret = v, r, x_dagger
        elif residual == "scaled":
            ret = v, r, x_dagger, Xi
        elif residual == "unscaled":
            ret = v, r, x_dagger, xi
        yield ret
        x = x_dagger
        sin_r *= np.sin(r)

    # k=d
    v, r = pss(x, tol)
    rho = np.arctan2(x @ (v @ [[0, 1], [-1, 0]]), x @ v)[..., np.newaxis]
    x_dagger = np.full((len(x), 1), 0, dtype=x.dtype)

    xi = rho - r
    Xi = sin_r * xi  # Xi(0)
    if residual == "none":
        ret = v, r, x_dagger
    elif residual == "scaled":
        ret = v, r, x_dagger, Xi
    elif residual == "unscaled":
        ret = v, r, x_dagger, xi
    yield ret


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
