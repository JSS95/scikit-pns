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
            A = proj(X, v, r)
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

    # def transform(self, X, y=None):
    #     if X.shape[1] != self._n_features:
    #         raise ValueError(
    #             f"Input dimension {X.shape[1]} does not match "
    #             f"fitted dimension {self._n_features}."
    #         )

    #     residuals = []
    #     for i in range(self._n_features - 2):
    #         v = self.v_[i]
    #         r = self.r_[i]
    #         # proj()
    #         geod = np.arccos(X @ v)[..., np.newaxis]
    #         A = (np.sin(r) * X + np.sin(geod - r) * v) / np.sin(geod)
    #         residuals.append(geod - r)
    #         X = embed(A, v, r)

    #     # deal with the last dimension
    #     v, r = self.v_[-1], self.r_[-1]
    #     residuals.append(np.arctan2(X @ (v @ [[0, 1], [-1, 0]]), X @ v).reshape(-1, 1))

    #     residuals = np.flip(np.concatenate(residuals, axis=-1), axis=-1)
    #     return residuals[:, :self.n_components]

    # # def inverse_transform(self, Xi):
    # #     """
    # #     Examples
    # #     --------
    # #     >>> from skpns import IntrinsicPNS
    # #     >>> from skpns.util import circular_data, unit_sphere
    # #     >>> X = circular_data()
    # #     >>> pns = IntrinsicPNS(2)
    # #     """
    # #     _, n = Xi.shape
    # #     if n + 1 > self._n_features:
    # #         raise ValueError(
    # #             f"Input extrinsic dimension {n + 1} is larger than "
    # #             f"fitted dimension {self._n_features}."
    # #         )
    # #     rs = list(reversed(self.r_))
    # #     vs = list(reversed(self.v_))

    # #     # Deal with the first dimension
    # #     v, r = vs[0], rs[0]
    # #     X[:, 0]

    # # #     # De-centralize data
    # # #     X = X.copy()
    # # #     X[:, :-1] += self.r_[:-1]
    # # #     X[:, -1] += self._cartesian_to_spherical(self.v_[-1].reshape(1, -1))[0]

    # # #     # Convert to extrinsic coordinates
    # # #     X_extrinsic = self._spherical_to_cartesian(X)
    # # #     if X_extrinsic.shape[1] < self._n_features:
    # # #         # Map to the original dimension
    # # #         vs = reversed(self.v_[: self._n_features - n - 1])
    # # #         rs = reversed(self.r_[: self._n_features - n - 1])
    # # #         for v, r in zip(vs, rs):
    # # #             X_extrinsic = reconstruct(X_extrinsic, v, r)
    # # #     else:
    # # #         pass
    # # #         # Data is already in original coordinates. Just rotate by v.

    # # #     # X[:, -1] += self._cartesian_to_spherical(self.v_[-1].reshape(1, -1))[0]
    # # #     # X_extrinsic = self._spherical_to_cartesian(X)

    # # #     # if X_extrinsic.shape[1] == self._n_features:
    # # #     #     # Data is already in original coordinates.
    # # #     #     # Just rotate by v.
    # # #     #     X_extrinsic, _ = _rotate(X_extrinsic, self.v_[0])
    # # #     # else:
    # # #     #     # Map to the original dimension
    # # #     #     vs = reversed(self.v_[: self._n_features - n - 1])
    # # #     #     rs = reversed(self.r_[: self._n_features - n - 1])
    # # #     #     for v, r in zip(vs, rs):
    # # #     #         X_extrinsic = reconstruct(X_extrinsic, v, r)
    # # #     # return X_extrinsic

    # # # def _spherical_to_cartesian(self, thetas):
    # # #     N, dim = thetas.shape
    # # #     n = dim + 1
    # # #     X = np.zeros((N, n))
    # # #     X[:, 0] = np.cos(thetas[:, 0])
    # # #     for i in range(1, n - 1):
    # # #         X[:, i] = np.prod(np.sin(thetas[:, :i]), axis=1) * np.cos(thetas[:, i])
    # # #     X[:, -1] = np.prod(np.sin(thetas), axis=1)
    # # #     return X

    # # # def _cartesian_to_spherical(self, X):
    # # #     N, n = X.shape
    # # #     thetas = np.zeros((N, n - 1))
    # # #     for i in range(n - 2):
    # # #         denom = np.prod(np.sin(thetas[:, :i]), axis=1) if i > 0 else 1.0
    # # #         arg = np.clip(X[:, i] / denom, -1.0, 1.0)
    # # #         thetas[:, i] = np.arccos(arg)
    # # #     thetas[:, -1] = np.arctan2(X[:, -1], X[:, -2])
    # # #     return thetas
