"""
Utilities
---------
"""

import numpy as np

__all__ = [
    "circular_data",
    "unit_sphere",
    "circle",
]


def circular_data():
    """Circular data in 3D.

    Returns
    -------
    ndarray of shape (100, 3)
        Data coordinates.
    """
    t = np.random.uniform(0.1 * np.pi, 0.2 * np.pi, 100)
    p = np.random.uniform(0, 3 * np.pi / 2, 100)
    circle = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)]).T

    v = np.array([1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)])
    north_pole = np.array([0.0, 0.0, 1.0])
    u = v - north_pole
    u /= np.linalg.norm(u)
    H = np.eye(3) - 2 * np.outer(u, u)
    return (H @ circle.T).T


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
