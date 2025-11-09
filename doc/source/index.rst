.. scikit-pns documentation master file, created by
   sphinx-quickstart on Mon Jun 16 06:52:48 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

************************
scikit-pns documentation
************************

.. plot:: plot-header.py
    :include-source: False

.. automodule:: skpns

Installation
============

scikit-pns can be installed using pip::

    pip install scikit-pns

Quickstart
==========

scikit-pns is imported as :mod:`skpns`.

.. code-block:: python

    from skpns import IntrinsicPNS
    from skpns.util import circular_data
    X = circular_data()
    X_new = IntrinsicPNS().fit_transform(X)

ONNX support
============

Transformers can be converted to ONNX models.

.. note::

    To use this feature, you need to install scikit-pns with ``[onnx]`` optional dependency::

        pip install scikit-pns[onnx]

.. plot::

    import numpy as np
    from skpns import ExtrinsicPNS, IntrinsicPNS
    from skpns.util import circular_data
    from skl2onnx import to_onnx
    import matplotlib.pyplot as plt

    # Train and save model
    X = circular_data().astype(np.float32)  # Must be float32

    int_pns = IntrinsicPNS(2).fit(X)
    with open("int_pns.onnx", "wb") as f:
        f.write(to_onnx(int_pns, X[:1]).SerializeToString())

    ext_pns = ExtrinsicPNS(2).fit(X)
    with open("ext_pns.onnx", "wb") as f:
        f.write(to_onnx(ext_pns, X[:1]).SerializeToString())

    # Load model
    import onnxruntime as rt

    ext_sess = rt.InferenceSession("ext_pns.onnx", providers=["CPUExecutionProvider"])
    ext_onnx = ext_sess.run([ext_sess.get_outputs()[0].name], {ext_sess.get_inputs()[0].name: X})[0]

    int_sess = rt.InferenceSession("int_pns.onnx", providers=["CPUExecutionProvider"])
    int_onnx = int_sess.run([int_sess.get_outputs()[0].name], {int_sess.get_inputs()[0].name: X})[0]

    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    ax1.plot(*int_pns.transform(X).T, "o", label="Python runtime")
    ax1.plot(*int_onnx.T, "x", label="ONNX runtime")
    ax1.set_xlim(-np.pi, np.pi)
    ax1.set_ylim(-np.pi / 2, np.pi / 2)
    ax1.legend()
    ax1.set_title("IntrinsicPNS")

    ax2 = fig.add_subplot(122)
    ax2.plot(*ext_pns.transform(X).T, "o", label="Python runtime")
    ax2.plot(*ext_onnx.T, "x", label="ONNX runtime")
    ax2.set_aspect("equal")
    ax2.legend()
    ax2.set_title("ExtrinsicPNS")

    fig.show()


Module reference
================

High-level API
--------------

.. autoclass:: skpns.IntrinsicPNS
    :members:

.. autoclass:: skpns.ExtrinsicPNS
    :members:

Low-level functions
-------------------

.. automodule:: skpns.pns
    :members:

Utilities
---------

.. automodule:: skpns.util
    :members:
