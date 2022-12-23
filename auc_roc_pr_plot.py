import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt


class Two_1d_normal_distributions:
    def __init__(self, P_ampl, N_ampl, P_mu, N_mu, P_std, N_std, color="b"):
        self.P_ampl = P_ampl
        self.N_ampl = N_ampl
        self.P_mu = P_mu
        self.N_mu = N_mu
        self.P_std = P_std
        self.N_std = N_std

        self.color = color
        self.N_color = "k"
        self.P_color = "r"

    def make(self, delta=0.05, steps=10001, start=-8, stop=8):
        index = 0
        self.fpr = []
        self.precision = []
        self.recall = []
        self.x_fpr = []
        self.x_precision = []
        self.x_recall = []
        self.xs = 0
        self.x_threshold = []
        self.o_fpr = []
        self.o_precision = []
        self.o_recall = []
        self.os = 0
        self.o_threshold = []
        for threshold in np.linspace(start, stop, steps):
            TN = self.N_ampl * norm.cdf(threshold, loc=self.N_mu, scale=self.N_std)
            FP = self.N_ampl - TN
            FN = self.P_ampl * norm.cdf(threshold, loc=self.P_mu, scale=self.P_std)
            TP = self.P_ampl - FN
            self.fpr.append(FP / (FP + TN))
            self.precision.append(TP / (TP + FP))
            self.recall.append(TP / (TP + FN))

            if (FN) / (self.P_ampl) >= self.xs * delta:
                self.x_fpr.append(FP / (FP + TN))
                self.x_precision.append(TP / (TP + FP))
                self.x_recall.append(TP / (TP + FN))
                self.xs += 1
                # print(xs, TN+FN)
                self.x_threshold.append(threshold)
            if (TN / self.N_ampl) >= self.os * delta:
                self.o_fpr.append(FP / (FP + TN))
                self.o_precision.append(TP / (TP + FP))
                self.o_recall.append(TP / (TP + FN))
                self.os += 1
                self.o_threshold.append(threshold)
                # print(os, TN+FN)

    def plot_roc_pr(self, roc_ax, pr_ax):
        roc_ax.plot(self.fpr, self.recall, self.color)
        pr_ax.plot(self.precision, self.recall, self.color)
        roc_ax.plot(self.x_fpr, self.x_recall, "x")
        pr_ax.plot(self.x_precision, self.x_recall, "x")
        roc_ax.plot(self.o_fpr, self.o_recall, "o")
        pr_ax.plot(self.o_precision, self.o_recall, "o")

    def plot_distributions(self, ax, start=-8, stop=8, steps=1001, normalize=True):
        grid = np.linspace(start, stop, steps)

        ax.plot(
            grid,
            norm.pdf(grid, loc=self.N_mu, scale=self.N_std)
            * (1 if normalize else self.N_ampl),
            color=self.N_color,
            label=f"pdf_N/{self.N_ampl}",
        )
        ax.plot(
            grid,
            norm.pdf(grid, loc=self.P_mu, scale=self.P_std)
            * (1 if normalize else self.P_ampl),
            color=self.P_color,
            label=f"pdf_P/{self.P_ampl}",
        )
        ax.plot(self.o_threshold, np.zeros(self.os), "o")
        ax.plot(self.x_threshold, np.zeros(self.xs), "x")
        # plt.plot(self.o_threshold, norm.pdf((np.array(self.o_threshold) - self.N_mu)/self.N_std), "o")
        # plt.plot(self.x_threshold, norm.pdf((np.array(self.x_threshold) - self.P_mu)/self.P_std), "x")
        ax.legend()

    def plot_cdf(self, ax, start=-8, stop=8, steps=1001, normalize=True):
        grid = np.linspace(start, stop, steps)

        # ax.plot(grid, norm.pdf(grid, loc=1, scale=2))
        # ax.plot(grid, norm.pdf(grid, loc=2, scale=3))
        # ax.plot(grid, norm.pdf(grid, loc=3, scale=1))
        ax.plot(
            grid,
            norm.cdf(grid, loc=self.N_mu, scale=self.N_std)
            * (1 if normalize else self.N_ampl),
            color=self.N_color,
            label=f"pdf_N/{self.N_ampl}",
        )
        ax.plot(
            grid,
            norm.cdf(grid, loc=self.P_mu, scale=self.P_std)
            * (1 if normalize else self.P_ampl),
            color=self.P_color,
            label=f"pdf_P/{self.P_ampl}",
        )


if __name__ == "__main__":

    t1 = Two_1d_normal_distributions(1, 50, 1.5, -1, 2, 1, color="b")
    t2 = Two_1d_normal_distributions(1, 50, 1, -1, 1, 1, color="g")

    t1.make(steps=1001, delta=0.1)
    t2.make(steps=1001, delta=0.1)

    fig, axes = plt.subplots(2, 1)
    t1.plot_roc_pr(axes[0], axes[1])
    t2.plot_roc_pr(axes[0], axes[1])
    plt.show()

    fig, axes = plt.subplots(2, 2)
    t1.plot_distributions(axes[0][0])
    t2.plot_distributions(axes[0][1])
    t1.plot_distributions(axes[1][0], normalize=False)
    t2.plot_distributions(axes[1][1], normalize=False)
    plt.show()

    # t1.plot_cdf(plt)
    # plt.show()
