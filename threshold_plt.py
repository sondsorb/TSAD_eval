import numpy as np
from matplotlib import pyplot as plt


x = np.arange(48)

y = np.sin(0.7 + x / 12) + np.sin(x / 4 + 29) + 0.1 * np.sin(1.25*x) * (np.cos(np.sqrt(1.25*x) + 2)) + x / 32 + 0.12


figsize=(3.4,2)
plt.figure(figsize=figsize)

plt.plot(x, y)

for t in [0.5, 1, 1.5, 2, 2.5]:
    plt.plot(x, x + t - x, ".k")
    for i in range(len(x)):
        if t < y[i]:
            plt.plot([x[i]], [t], ".r")

plt.tight_layout()
plt.savefig("thr2.pdf")
plt.show()
