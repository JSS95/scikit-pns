.. scikit-pns documentation master file, created by
   sphinx-quickstart on Mon Jun 16 06:52:48 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

************************
scikit-pns documentation
************************

.. plot:: plot-header.py
    :include-source: False

`scikit-pns` provides :class:`.PNS`, which is a scikit-learn transformer for princiapl nested spheres analysis.

ONNX compatibility
==================

PNS can be converted to ONNX and saved.

.. note::

    To use this feature, you need to install scikit-pns with ``[onnx]`` optional dependency::

        pip install scikit-pns[onnx]

.. plot::

    import numpy as np
    from skpns import PNS
    from skpns.util import circular_data
    from skl2onnx import to_onnx
    import matplotlib.pyplot as plt

    # Train and save model
    X = circular_data().astype(np.float32)  # Must be float32
    pns = PNS(2).fit(X)
    Xpred = pns.transform(X)

    onx = to_onnx(pns, X[:1])
    with open("pns.onnx", "wb") as f:
        f.write(onx.SerializeToString())

    # Load model
    import onnxruntime as rt

    sess = rt.InferenceSession("pns.onnx", providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    Xpred_onnx = sess.run([label_name], {input_name: X})[0]

    plt.plot(*Xpred.T, "o")
    plt.plot(*Xpred_onnx.T, "x")
    plt.gca().set_aspect("equal")
    plt.show()


Module reference
================

.. automodule:: skpns
    :members:

.. automodule:: skpns.pns
    :members:

.. automodule:: skpns.util
    :members:
