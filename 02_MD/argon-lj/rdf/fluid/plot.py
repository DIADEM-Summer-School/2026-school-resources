import numpy as np
import matplotlib.pyplot as plt

filename = "rdf.dat"
data = np.loadtxt(filename, skiprows=4)

x = data[:, 1] # r
y = data[:, 2] # gr

fig, ax = plt.subplots(figsize=(6, 6), facecolor="#111111")
ax.set_facecolor("#111111")
ax.plot(x, y, "o-", color="white", markersize=7, linewidth=2)
ax.set_xlabel(r"$r$", fontsize=22, color="white", labelpad=12)
ax.set_ylabel(r"$g(r)$", fontsize=22, color="white", labelpad=12)
ax.tick_params(axis="both", which="major", direction="in", colors="white", labelsize=20, length=8, width=1.5, top=True, right=True)
ax.minorticks_on()
ax.tick_params(axis="both", which="minor", direction="in", colors="white", length=4, width=1.0, top=True, right=True)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(20)
    label.set_color("white")
for spine in ax.spines.values():
    spine.set_color("white")
    spine.set_linewidth(1.5)
ax.grid(False)
plt.tight_layout()
plt.savefig("rdf.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.show()

