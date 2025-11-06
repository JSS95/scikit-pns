"""Utility functions for data generation and transformation."""

import numpy as np

from .pns import _R

__all__ = [
    "circular_data",
    "unit_sphere",
    "circle",
]


def circular_data(dim=3):
    """Circular data in 3D, or its embedding in 2D.

    Parameters
    ----------
    dim : {3, 2}
        Data dimension.

    Returns
    -------
    ndarray of shape (100, dim)
        Data coordinates.

    Examples
    --------
    >>> from skpns.util import circular_data, unit_sphere
    >>> x_3d = circular_data(3)
    >>> x_2d = circular_data(2)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... fig = plt.figure()
    ... ax1 = fig.add_subplot(121, projection='3d', computed_zorder=False)
    ... ax2 = fig.add_subplot(122)
    ... ax1.plot_surface(*unit_sphere(), color='skyblue', alpha=0.6, edgecolor='gray')
    ... ax1.scatter(*x_3d.T, marker="x")
    ... ax2.scatter(*x_2d.T, marker="x")
    ... ax2.set_aspect("equal")
    """
    # 3D data around north pole
    t = np.random.uniform(0.1 * np.pi, 0.2 * np.pi, 100)
    p = np.random.uniform(0, 3 * np.pi / 2, 100)
    x = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)]).T
    # Rotate data in altitude angle
    v = np.array([1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)])
    north_pole = np.array([0.0, 0.0, 1.0])
    u = v - north_pole
    u /= np.linalg.norm(u)
    H = np.eye(3) - 2 * np.outer(u, u)
    x = (H @ x.T).T
    if dim == 3:
        pass
    elif dim == 2:
        # Embed data in 2D
        R = _R(v)
        r = np.mean(t)
        x = x @ (1 / np.sin(r) * R[:-1:, :]).T
    else:
        raise ValueError("Invalid dimension.")
    return x


def unit_sphere():
    """Helper function to plot a unit sphere.

    Returns
    -------
    x, y, z : array
        Coordinates for unit sphere.
    """
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def circle(v, theta, n=100):
    """Helper function to plot a circle in 3D.

    Parameters
    ----------
    v : (3,) array
        Unit vector to center of circle in 3D.
    theta : scalar
        Geodesic distance.
    n : int, default=100
        Number of points.
    """
    phi = np.linspace(0, 2 * np.pi, n)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.full_like(phi, np.cos(theta))
    circle = np.stack([x, y, z], axis=1)

    north_pole = np.array([0.0, 0.0, 1.0])
    u = v - north_pole
    u /= np.linalg.norm(u)
    H = np.eye(3) - 2 * np.outer(u, u)
    return H @ circle.T
