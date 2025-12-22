"""Principal nested spheres (PNS) analysis [1]_ for scikit-learn.

The main API classes are :class:`IntrinsicPNS` and :class:`ExtrinsicPNS`.
Low-level functions are available in :mod:`skpns.pns`.

.. [1] Jung, Sungkyu, Ian L. Dryden, and James Stephen Marron.
       "Analysis of principal nested spheres." Biometrika 99.3 (2012): 551-568.
"""

from .sklearn import ExtrinsicPNS, IntrinsicPNS, InverseExtrinsicPNS

__all__ = [
    "ExtrinsicPNS",
    "IntrinsicPNS",
    "InverseExtrinsicPNS",
]

try:
    from skl2onnx import update_registered_converter

    from .onnx import (
        extrinsicpns_converter,
        intrinsicpns_converter,
        inverse_extrinsicpns_converter,
        shape_calculator,
    )

    update_registered_converter(
        IntrinsicPNS,
        "SkpnsIntrinsicPNS",
        shape_calculator,
        intrinsicpns_converter,
    )
    update_registered_converter(
        ExtrinsicPNS,
        "SkpnsExtrinsicPNS",
        shape_calculator,
        extrinsicpns_converter,
    )
    update_registered_converter(
        InverseExtrinsicPNS,
        "SkpnsInverseExtrinsicPNS",
        shape_calculator,
        inverse_extrinsicpns_converter,
    )

except ModuleNotFoundError:
    pass
