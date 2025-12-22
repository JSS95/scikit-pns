"""Scikit-learn wrappers for PNS."""

import numpy as np
import pns as pnspy
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    "ExtrinsicPNS",
    "InverseExtrinsicPNS",
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
    maxiter : int, optional
        Maximum number of iterations for the optimization.
        If None, the number of iterations is not checked.
    lm_kwargs : dict, optional
        Additional keyword arguments to be passed for Levenberg-Marquardt optimization.
        Follows the signature of :func:`scipy.optimize.least_squares`.

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
    >>> from pns.util import circular_data, unit_sphere
    >>> X = circular_data([0, -1, 0])
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

    def __init__(self, n_components=2, tol=1e-3, maxiter=None, lm_kwargs=None):
        self.n_components = n_components
        self.tol = tol
        self.maxiter = maxiter
        self.lm_kwargs = lm_kwargs

    def _fit_transform(self, X):
        self._n_features = X.shape[1]
        self.v_, self.r_, _, self.embedding_ = pnspy.pns(
            X,
            self.n_components,
            tol=self.tol,
            maxiter=self.maxiter,
            lm_kwargs=self.lm_kwargs,
        )

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
        return pnspy.extrinsic_transform(X, self.v_, self.r_)

    def inverse_transform(self, X):
        """Transform the low-dimensional data back to the original hypersphere.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)

        Returns
        -------
        X_new : array-like of shape (n_samples, n_features)
        """
        return pnspy.inverse_extrinsic_transform(X, self.v_, self.r_)


class InverseExtrinsicPNS(TransformerMixin, BaseEstimator):
    """Inverse converter of :class:`ExtrinsicPNS`.

    This class is for building ONNX graph and not intended to be used directly.
    Use :meth:`ExtrinsicPNS.inverse_transform` instead in Python runtime.

    Parameters
    ----------
    extrinsic_pns : ExtrinsicPNS
        Fitted :class:`ExtrinsicPNS` instance.

    Examples
    --------
    >>> from skpns import ExtrinsicPNS, InverseExtrinsicPNS
    >>> from pns.util import circular_data
    >>> from skl2onnx import to_onnx
    >>> X = circular_data().astype('float32')
    >>> pns = ExtrinsicPNS(n_components=2).fit(X)
    >>> onnx = to_onnx(InverseExtrinsicPNS(pns), X[:1])
    """

    def __init__(self, extrinsic_pns):
        self.extrinsic_pns = extrinsic_pns
        self.v_ = extrinsic_pns.v_
        self.r_ = extrinsic_pns.r_
        self.n_components = extrinsic_pns._n_features

    def transform(self, X):
        return self.extrinsic_pns.inverse_transform(X)


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
    maxiter : int, optional
        Maximum number of iterations for the optimization.
        If None, the number of iterations is not checked.

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
    lm_kwargs : dict, optional
        Additional keyword arguments to be passed for Levenberg-Marquardt optimization.
        Follows the signature of :func:`scipy.optimize.least_squares`.

    Notes
    -----
    The resulting data is the transposed matrix of

    .. math::

        \hat{X}_\mathrm{PNS} =
        \begin{bmatrix}
            \Xi(0) \\
            \Xi(1) \\
            \vdots \\
            \Xi(n)
        \end{bmatrix},

    with notations in the original paper, where :math:`n` is *n_components*.
    The coordinates lie in :math:`[-\pi, \pi] \times [-\pi/2, \pi/2]^{n-1}`,
    i.e., the azimuthal angle is the first coordinate.

    Examples
    --------
    >>> import numpy as np
    >>> from skpns import IntrinsicPNS
    >>> from pns.util import circular_data, unit_sphere
    >>> X = circular_data([0, -1, 0])
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

    def __init__(self, n_components=None, tol=1e-3, maxiter=None, lm_kwargs=None):
        self.n_components = n_components
        self.tol = tol
        self.maxiter = maxiter
        self.lm_kwargs = lm_kwargs

    def _fit_transform(self, X):
        if self.n_components is None:
            self.n_components = X.shape[1] - 1

        self._n_features = X.shape[1]
        self.v_, self.r_, xi, _ = pnspy.pns(
            X,
            1,
            tol=self.tol,
            maxiter=self.maxiter,
            lm_kwargs=self.lm_kwargs,
        )

        sin_r = 1
        for i in range(xi.shape[1] - 1):
            xi[:, i] *= sin_r
            sin_r *= np.sin(self.r_[i])
        xi[:, -1] *= sin_r

        self.embedding_ = np.flip(xi, axis=-1)

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
        return self.embedding_[:, : self.n_components]

    def transform(self, X, y=None):
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
        return pnspy.intrinsic_transform(X, self.v_, self.r_)

    def inverse_transform(self, Xi):
        """Transform the low-dimensional data back to the original hypersphere.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)

        Returns
        -------
        X_new : array-like of shape (n_samples, n_features)

        Examples
        --------
        >>> from skpns import IntrinsicPNS
        >>> from pns.util import circular_data, unit_sphere
        >>> X = circular_data([0, -1, 0])
        >>> pns = IntrinsicPNS(1)
        >>> Xi = pns.fit_transform(X)
        >>> X_inv = pns.inverse_transform(Xi)
        >>> import matplotlib.pyplot as plt  # doctest: +SKIP
        ... ax = plt.figure().add_subplot(projection='3d', computed_zorder=False)
        ... ax.plot_surface(*unit_sphere(), color='skyblue', edgecolor='gray')
        ... ax.scatter(*X.T)
        ... ax.scatter(*X_inv.T)
        """
        return pnspy.inverse_intrinsic_transform(Xi, self.v_, self.r_)
