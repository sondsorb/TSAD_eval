import numpy as np
from matplotlib import pyplot as plt


x = np.arange(60)

y = np.sin(0.7 + x / 15) + np.sin(x / 5 + 29) + 0.1 * np.sin(x) * (np.cos(np.sqrt(x) + 2)) + x / 40


plt.plot(x, y)

for t in [0.5, 1, 1.5, 2, 2.5]:
    plt.plot(x, x + t - x, ".k")
    for i in range(len(x)):
        if t < y[i]:
            plt.plot([x[i]], [t], ".r")

plt.show()
