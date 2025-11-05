"""Principal nested spheres (PNS) analysis [1]_ for scikit-learn.

.. [1] Jung, Sungkyu, Ian L. Dryden, and James Stephen Marron.
       "Analysis of principal nested spheres." Biometrika 99.3 (2012): 551-568.
"""

from .sklearn import PNS

__all__ = [
    "PNS",
]

try:
    from skl2onnx import update_registered_converter

    from .onnx import pns_converter, pns_shape_calculator

    update_registered_converter(PNS, "SkpnsPNS", pns_shape_calculator, pns_converter)

except ModuleNotFoundError:
    pass
