import matplotlib.pyplot as plt
import numpy as np

from skpns import PNS
from skpns.util import circle, circular_data, unit_sphere

pns = PNS(n_components=2)
X = circular_data()
X_new = pns.fit_transform(X)
v, r = pns.v_[0], pns.r_[0]

fig = plt.figure()
ax = fig.add_subplot(projection="3d", computed_zorder=False)
ax.plot_surface(*unit_sphere(), color="skyblue", alpha=0.6, edgecolor="gray")
ax.scatter(*X.T, marker=".", color="tab:blue")
ax.plot(*circle(v, r), color="tab:orange")
ax.scatter(*pns.to_hypersphere(X_new).T, marker="x", color="tab:green")

xs, ys, zs = X.T
ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
