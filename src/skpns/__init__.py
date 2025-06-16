"""Principal nested spheres analysis."""

from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    "PNS",
]


class PNS(TransformerMixin, BaseEstimator):
    """Principal nested spheres (PNS) analysis [1]_.

    References
    ----------
    .. [1] Jung, Sungkyu, Ian L. Dryden, and James Stephen Marron.
       "Analysis of principal nested spheres." Biometrika 99.3 (2012): 551-568.
    """
