"""Scikit-learn wrappers for PNS."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .pns import embed, pns, proj, pss, reconstruct

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

    The resulting data are represented by intrinsic coordinates.
    For example, `n_components=2` transforms data onto the surface of a 3D sphere,
    represented by spherical coordinates.

    The transformed data are in hyperspherical coordinates, with the range of angles in
    each dimension being

    .. math::

        [-\pi/2, \pi/2], \ldots, [-\pi/2, \pi/2], [-\pi, \pi].

    Parameters
    ----------
    n_components : int, default=None
        Number of components to keep.
        Data are transformed onto a unit hypersphere in this dimension, embedded in
        `n_components + 1` dimensions.
        If None, all components are kept.
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
    >>> from skpns import IntrinsicPNS
    >>> from skpns.util import circular_data, unit_sphere
    >>> X = circular_data()
    >>> pns = IntrinsicPNS()
    >>> X_transformed = pns.fit_transform(X)
    >>> import matplotlib.pyplot as plt
    ... fig = plt.figure()
    ... ax1 = fig.add_subplot(121, projection='3d', computed_zorder=False)
    ... ax1.plot_surface(*unit_sphere(), color='skyblue', alpha=0.6, edgecolor='gray')
    ... ax1.scatter(*X.T, c=X_transformed[:, -1])
    ... ax2 = fig.add_subplot(122)
    ... ax2.set_xlim(-np.pi/2, np.pi/2)
    ... ax2.set_ylim(-np.pi, np.pi)
    ... ax2.scatter(*X_transformed.T, c=X_transformed[:, -1])
    """

    def __init__(self, n_components=None, tol=1e-3):
        self.n_components = n_components
        self.tol = tol

    def _fit_transform(self, X):
        if self.n_components is None:
            self.n_components = X.shape[1] - 1

        self._n_features = X.shape[1]
        self.v_ = []
        self.r_ = []

        residuals = []
        for _ in range(self._n_features - 2):
            v, r = pss(X, self.tol)
            self.v_.append(v)
            self.r_.append(r)

            # Projection
            geod = np.arccos(X @ v)[..., np.newaxis]
            A = (np.sin(r) * X + np.sin(geod - r) * v) / np.sin(geod)
            residuals.append(geod - r)
            X = embed(A, v, r)

        # deal with the last dimension
        v, r = pss(X, self.tol)
        self.v_.append(v)
        self.r_.append(r)
        residuals.append(np.arctan2(X @ (v @ [[0, 1], [-1, 0]]), X @ v).reshape(-1, 1))
        self.embedding_ = np.concatenate(residuals, axis=-1)[:, -self.n_components :]

    def fit(self, X, y=None):
        self._fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        self._fit_transform(X)
        return self.embedding_

    def transform(self, X, y=None):
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"Input dimension {X.shape[1]} does not match "
                f"fitted dimension {self._n_features}."
            )

        residuals = []
        for i in range(self._n_features - 2):
            v = self.v_[i]
            r = self.r_[i]
            # proj()
            geod = np.arccos(X @ v)[..., np.newaxis]
            A = (np.sin(r) * X + np.sin(geod - r) * v) / np.sin(geod)
            residuals.append(geod - r)
            X = embed(A, v, r)

        # deal with the last dimension
        v, r = self.v_[-1], self.r_[-1]
        residuals.append(np.arctan2(X @ (v @ [[0, 1], [-1, 0]]), X @ v).reshape(-1, 1))

        residuals = np.concatenate(residuals, axis=-1)
        return residuals[:, -self.n_components :]
