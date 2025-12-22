import matplotlib.pyplot as plt
from pns.util import circular_data, unit_sphere

from skpns import ExtrinsicPNS

pns = ExtrinsicPNS(n_components=2)
X = circular_data([0, -1, 0])
X_new = pns.fit_transform(X)
v, r = pns.v_[0], pns.r_[0]

fig = plt.figure()
ax = fig.add_subplot(projection="3d", computed_zorder=False)
ax.plot_surface(*unit_sphere(), color="skyblue", alpha=0.6, edgecolor="gray")
ax.scatter(*X.T, marker=".", color="tab:blue")
ax.scatter(*pns.inverse_transform(X_new).T, marker="x", color="tab:green")
