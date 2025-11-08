"""Scikit-learn wrappers for PNS."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .pns import embed, pns, proj, reconstruct

__all__ = [
    "ExtrinsicPNS",
    "PNS",
    "IntrinsicPNS",
]


class ExtrinsicPNS(TransformerMixin, BaseEstimator):
    """Principal nested spheres (PNS) analysis with extrinsic coordinates.

    Reduces the dimensionality of data on a high-dimensional hypersphere
    while preserving its spherical geometry.

    The resulting data are represented by extrinsic coordinates.
    For example, `n_components=2` transforms data onto a 2D unit circle,
    represented by x and y coordinates.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep.
        Data are transformed onto a unit hypersphere embedded in this dimensional space.
    tol : float, default=1e-3
        Optimization tolerance.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Stores the embedding vectors.
    v_ : list of (n_features - 1) arrays
        Principal directions of nested spheres.
    r_ : ndarray of shape (n_features - 1,)
        Principal radii of nested spheres.

    Examples
    --------
    >>> from skpns import ExtrinsicPNS
    >>> from skpns.util import circular_data, unit_sphere
    >>> X = circular_data()
    >>> pns = ExtrinsicPNS(n_components=2)
    >>> X_reduced = pns.fit_transform(X)
    >>> X_inv = pns.inverse_transform(X_reduced)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... fig = plt.figure()
    ... ax1 = fig.add_subplot(121, projection='3d', computed_zorder=False)
    ... ax1.plot_surface(*unit_sphere(), color='skyblue', alpha=0.6, edgecolor='gray')
    ... ax1.scatter(*X_inv.T, zorder=10)
    ... ax1.scatter(*X.T)
    ... ax2 = fig.add_subplot(122)
    ... ax2.scatter(*X_reduced.T)
    ... ax2.set_aspect('equal')
    """

    def __init__(self, n_components=2, tol=1e-3):
        self.n_components = n_components
        self.tol = tol

    def _fit_transform(self, X):
        self._n_features = X.shape[1]
        self.v_ = []
        self.r_ = []

        D = X.shape[1]
        pns_ = pns(X, self.tol)
        for _ in range(D - self.n_components):
            v, r, X = next(pns_)
            self.v_.append(v)
            self.r_.append(r)
        self.embedding_ = X

    def fit(self, X, y=None):
        """Find principal nested spheres for the data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data on (n_features - 1)-dimensional hypersphere.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns a fitted instance of self.
        """
        self._fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        """Fit the model with data in X and transform X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data on (n_features - 1)-dimensional hypersphere.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            X transformed in the new space.
        """
        self._fit_transform(X)
        return self.embedding_

    def transform(self, X):
        """Transform X onto the fitted subsphere.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data on (n_features - 1)-dimensional hypersphere.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            X transformed in the new space.
        """
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"Input dimension {X.shape[1]} does not match "
                f"fitted dimension {self._n_features}."
            )

        for v, r in zip(self.v_, self.r_):
            A, _ = proj(X, v, r)
            X = embed(A, v, r)
        return X

    def inverse_transform(self, X):
        """Transform the low-dimensional data back to the original hypersphere.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)

        Returns
        -------
        X_new : array-like of shape (n_samples, n_features)
        """
        for v, r in zip(reversed(self.v_), reversed(self.r_)):
            X = reconstruct(X, v, r)
        return X

    def to_hypersphere(self, X):
        """Alias for :meth:`inverse_transform`."""
        return self.inverse_transform(X)


PNS = ExtrinsicPNS


class IntrinsicPNS(TransformerMixin, BaseEstimator):
    r"""Principal nested spheres (PNS) analysis with intrinsic coordinates.

    Reduces the dimensionality of data on a high-dimensional hypersphere
    while preserving its spherical geometry.

    The resulting data are intrinsic Euclidean coordinates, which are the
    scaled residuals in each dimension. For example, `n_components=2`
    represents data on the surface of a 3D sphere.

    Parameters
    ----------
    n_components : int, default=None
        Number of components to keep.
        Data are transformed onto a Euclidean space in this dimension,
        representing the surface of a hypersphere with the same dimension.
        If None, all components are kept, i.e., extrinsic coordinates are
        converted to intrinsic coordinates without loosing dimenisonality.
    tol : float, default=1e-3
        Optimization tolerance.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, d)
        The embedding vectors,
        :math:`\Xi(0), \Xi(1), \ldots, \Xi(d-1)`,
        where the input data is on d-sphere.
    v_ : list of arrays
        Principal directions of nested spheres,
        :math:`\hat{v}_1, \hat{v}_2, \ldots, \hat{v}_d`.
    r_ : ndarray
        Principal radii of nested spheres,
        :math:`\hat{r}_1, \hat{r}_2, \ldots, \hat{r}_d`.

    Notes
    -----
    The resulting data is

    .. math::

        \hat{X}_\mathrm{PNS} =
        \begin{bmatrix}
            \Xi(0) \\
            \Xi(1) \\
            \vdots \\
            \Xi(n)
        \end{bmatrix},

    using notation in the original paper, where :math:`n` is *n_components*.
    The coordinates lie in :math:`[-\pi, \pi] \times [-\pi/2, \pi/2]^{n-1}`.

    Examples
    --------
    >>> from skpns import IntrinsicPNS
    >>> from skpns.util import circular_data, unit_sphere
    >>> X = circular_data()
    >>> pns = IntrinsicPNS()
    >>> Xi = pns.fit_transform(X)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... fig = plt.figure()
    ... ax1 = fig.add_subplot(121, projection='3d', computed_zorder=False)
    ... ax1.plot_surface(*unit_sphere(), color='skyblue', edgecolor='gray')
    ... ax1.scatter(*X.T, c=Xi[:, 0])
    ... ax2 = fig.add_subplot(122)
    ... ax2.scatter(*Xi.T, c=Xi[:, 0])
    ... ax2.set_xlim(-np.pi, np.pi)
    ... ax2.set_ylim(-np.pi/2, np.pi/2)
    """

    def __init__(self, n_components=None, tol=1e-3):
        self.n_components = n_components
        self.tol = tol

    def _fit_transform(self, X):
        if self.n_components is None:
            self.n_components = X.shape[1] - 1

        self._n_features = X.shape[1]  # d+1
        self.v_ = []
        self.r_ = []

        residuals = []
        for v, r, _, Xi in pns(X, self.tol, residual="scaled"):
            self.v_.append(v)
            self.r_.append(r)
            residuals.append(Xi)
        self.embedding_ = np.flip(np.concatenate(residuals, axis=-1), axis=-1)

    def fit(self, X, y=None):
        self._fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        self._fit_transform(X)
        return self.embedding_[:, : self.n_components]

    def transform(self, X, y=None):
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"Input dimension {X.shape[1]} does not match "
                f"fitted dimension {self._n_features}."
            )

        d = X.shape[1] - 1
        residuals = []

        sin_r = 1
        for k in range(1, d):
            v, r = self.v_[k - 1], self.r_[k - 1]
            P, xi = proj(X, v, r)
            X = embed(P, v, r)
            Xi = sin_r * xi
            residuals.append(Xi)
            sin_r *= np.sin(r)

        v, r = self.v_[k], self.r_[k]
        _, xi = proj(X, v, r)
        Xi = sin_r * xi
        residuals.append(Xi)

        ret = np.flip(np.concatenate(residuals, axis=-1), axis=-1)
        return ret[:, : self.n_components]

    def inverse_transform(self, Xi):
        """
        Examples
        --------
        >>> from skpns import IntrinsicPNS
        >>> from skpns.util import circular_data, unit_sphere
        >>> X = circular_data()
        >>> pns = IntrinsicPNS()
        >>> Xi = pns.fit_transform(X)
        >>> X_inv = pns.inverse_transform(Xi)
        >>> import matplotlib.pyplot as plt  # doctest: +SKIP
        ... fig = plt.figure()
        ... ax1 = fig.add_subplot(121, projection='3d', computed_zorder=False)
        ... ax1.plot_surface(*unit_sphere(), color='skyblue', edgecolor='gray')
        ... ax1.scatter(*X.T)
        ... ax2 = fig.add_subplot(122, projection='3d', computed_zorder=False)
        ... ax2.plot_surface(*unit_sphere(), color='skyblue', edgecolor='gray')
        ... ax2.scatter(*X_inv.T)
        """
        d = self._n_features - 1
        _, n = Xi.shape
        if n > d:
            raise ValueError(
                f"Input dimension {n} is larger than fitted dimension {d}."
            )
        Xi = np.concatenate(
            [Xi, np.zeros((Xi.shape[0], self._n_features - n - 1))], axis=-1
        )  # Now, each column in Xi is Xi(0), ..., Xi(d-1).

        # Un-scale Xi, i.e., xi(d-k) = Xi(d-k) / prod_{i=1}^{k-1}(sin(r_i)).
        sin_rs = np.sin(self.r_[:-1])  # sin(r_1), sin(r_2), ..., sin(r_{d-1})
        xi = Xi.copy()  # xi(0), ..., xi(d-1)
        prod_sin_r = np.prod(sin_rs)
        for i in range(d - 1):
            xi[:, i] /= prod_sin_r
            prod_sin_r /= sin_rs[-i - 1]
        xi[:, d - 1] /= prod_sin_r
        # 여기까지는 문제없이 unscaled residual를 잘 복원함을 확인함....

        # Starting from the lowest dimension,
        # 1. Convert to cartesian coordinates.
        # 2. Reconstruct to one higher dimension, with north pole different from v.
        # 3. Rotate for v.
        # 4. Un-project with residuals.
        # 5. Go to 2.

        # Initialize: rotate xi(0) and convert to cartesian
        xi[:, 0] += _cartesian_to_hyperspherical(self.v_[-1][np.newaxis, ...])[0]
        x_dagger = _hyperspherical_to_cartesian(xi[:, :1])

        # Step 2 to Step 5
        for i in range(d - 1):
            k = i + 1  # 1, 2, ..., d - 1
            A = reconstruct(x_dagger, self.v_[-1 - k], self.r_[-1 - k])
            x_dagger = _inv_proj(A, np.sin(xi[:, k]), self.v_[-1 - k], self.r_[-1 - k])

        return x_dagger


def _cartesian_to_hyperspherical(X):
    """
    Convert (N, d+1) Cartesian coordinates on a d-sphere
    to hyperspherical coordinates (N, d).

    Convention:
      - xi[..., 0]: azimuthal angle in [-pi, pi]
      - xi[..., 1:]: centered elevation angles in [-pi/2, pi/2]
    """
    N, D = X.shape
    d = D - 1
    xi = np.zeros((N, d))

    xi[:, 0] = np.arctan2(X[:, 1], X[:, 0])
    for i in range(1, d - 1):
        denom = np.linalg.norm(X[:, i:], axis=1)
        xi[:, i] = np.arctan(X[:, i + 1] / denom)
    xi[:, -1] = np.arctan2(X[:, -1], np.linalg.norm(X[:, :-1], axis=1))

    return xi


def _hyperspherical_to_cartesian(xi):
    """
    Convert (N, d) hyperspherical coordinates back to
    Cartesian coordinates (N, d+1) on a unit d-sphere.

    Convention:
      - xi[..., 0]: azimuthal angle in [-pi, pi]
      - xi[..., 1:]: centered elevation angles in [-pi/2, pi/2]
    """
    N, d = xi.shape
    X = np.zeros((N, d + 1))

    for n in range(N):
        angles = xi[n]
        cos_elev = np.cumprod(np.cos(angles[1:][::-1]))[::-1]
        X[n, 0] = np.cos(angles[0]) * (cos_elev[0] if d > 1 else 1)
        X[n, 1] = np.sin(angles[0]) * (cos_elev[0] if d > 1 else 1)
        for k in range(1, d):
            X[n, k + 1] = np.sin(angles[k]) * (cos_elev[k] if k < d - 1 else 1)
    return X


def _inv_proj(P, xi, v, r):
    """Inverse of pns.proj().

    Parameters
    ----------
    P, xi : Results from pns.proj()
    v, r : Principal axis and geodesic distance.
    """
    rho = (xi + r)[..., np.newaxis]
    return (P * np.sin(rho) - np.sin(rho - r) * v) / np.sin(r)
