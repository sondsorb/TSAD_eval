import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

from metrics import f1_from_pr


class Two_1d_normal_distributions:
    def __init__(self, P_ampl, N_ampl, P_mu, N_mu, P_std, N_std, color="b", betas=None):
        self.P_ampl = P_ampl
        self.N_ampl = N_ampl
        self.P_mu = P_mu
        self.N_mu = N_mu
        self.P_std = P_std
        self.N_std = N_std
        self.betas = (1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16, 32) if betas == None else betas

        self.color = color
        self.N_color = "k"
        self.P_color = "r"

    def make(self, delta=0.05, steps=10001, start=-8, stop=8):
        index = 0

        # For plotting grahps
        self.fpr = []
        self.precision = []
        self.recall = []

        # For plotting x´s and o´s on the graphs
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

        # Track maximum f scores for various beta values
        self.max_f = {i: 0 for i in self.betas}
        self.max_f_fpr = {i: 0 for i in self.betas}
        self.max_f_precision = {i: 0 for i in self.betas}
        self.max_f_recall = {i: 0 for i in self.betas}
        self.max_f_thresholds = {i: 0 for i in self.betas}

        for threshold in np.linspace(start, stop, steps):
            TN = self.N_ampl * norm.cdf(threshold, loc=self.N_mu, scale=self.N_std)
            FP = self.N_ampl - TN
            FN = self.P_ampl * norm.cdf(threshold, loc=self.P_mu, scale=self.P_std)
            TP = self.P_ampl - FN
            self.fpr.append(FP / (FP + TN))
            self.precision.append(TP / (TP + FP))
            self.recall.append(TP / (TP + FN))

            for beta in self.betas:
                if f1_from_pr(p=self.precision[-1], r=self.recall[-1], beta=beta) > self.max_f[beta]:
                    self.max_f[beta] = f1_from_pr(p=self.precision[-1], r=self.recall[-1], beta=beta)
                    self.max_f_fpr[beta] = self.fpr[-1]
                    self.max_f_precision[beta] = self.precision[-1]
                    self.max_f_recall[beta] = self.recall[-1]
                    self.max_f_thresholds[beta] = threshold

            if (FN) / (self.P_ampl) >= self.xs * delta + delta * 0.5:
                self.x_fpr.append(FP / (FP + TN))
                self.x_precision.append(TP / (TP + FP))
                self.x_recall.append(TP / (TP + FN))
                self.xs += 1
                # print(xs, TN+FN)
                self.x_threshold.append(threshold)
            if (TN / self.N_ampl) >= self.os * delta + delta * 0.5:
                self.o_fpr.append(FP / (FP + TN))
                self.o_precision.append(TP / (TP + FP))
                self.o_recall.append(TP / (TP + FN))
                self.os += 1
                self.o_threshold.append(threshold)
                # print(os, TN+FN)

    def plot_roc_pr(self, roc_ax, pr_ax, plot_xs=True, plot_os=True, plot_fs=False):
        roc_ax.plot(self.fpr, self.recall, self.color, zorder=1)
        pr_ax.plot(self.precision, self.recall, self.color, zorder=1)
        if plot_xs:
            roc_ax.plot(self.x_fpr, self.x_recall, "x", color=self.color, zorder=1)
            pr_ax.plot(self.x_precision, self.x_recall, "x", color=self.color, zorder=1)
        if plot_os:
            roc_ax.plot(self.o_fpr, self.o_recall, "o", color=self.color, fillstyle="none", zorder=1)
            pr_ax.plot(self.o_precision, self.o_recall, "o", color=self.color, fillstyle="none", zorder=1)
        if plot_fs:
            roc_ax.plot(
                list(self.max_f_fpr.values()),
                list(self.max_f_recall.values()),
                ".",
                linestyle="None",
                zorder=2,
                color="k",
            )  # self.color)
            pr_ax.plot(
                list(self.max_f_precision.values()), list(self.max_f_recall.values()), ".", zorder=2, color="k"
            )  # self.color)
            for beta in self.betas:
                # roc_ax.plot([self.max_f_fpr[beta]],[self.max_f_recall[beta]], marker=f"$1/{int(1/beta)}$" if beta<1 else f"${beta}$", linestyle= "None", zorder=2, color="k")
                # pr_ax.plot([self.max_f_precision[beta]], [self.max_f_recall[beta]], marker=f"$1/{int(1/beta)}$" if beta<1 else f"${beta}$", zorder=2, color="k")
                if self.color == "forestgreen":  # need to place the numbers differently
                    roc_ax.text(
                        self.max_f_fpr[beta],
                        self.max_f_recall[beta],
                        f" $1/{int(1/beta)}$" if beta < 1 else f" ${beta}$",
                        horizontalalignment="left",
                        verticalalignment="top",
                        color=self.color,
                    )
                    pr_ax.text(
                        self.max_f_precision[beta],
                        self.max_f_recall[beta],
                        f"$1/{int(1/beta)}$" if beta < 1 else f"${beta}$",
                        horizontalalignment="left",
                        verticalalignment="bottom",
                        color=self.color,
                    )
                else:
                    roc_ax.text(
                        self.max_f_fpr[beta],
                        self.max_f_recall[beta],
                        f"$1/{int(1/beta)}$" if beta < 1 else f"${beta}$",
                        horizontalalignment="right",
                        verticalalignment="bottom",
                        color=self.color,
                    )
                    pr_ax.text(
                        self.max_f_precision[beta],
                        self.max_f_recall[beta],
                        f"$1/{int(1/beta)}$ " if beta < 1 else f"${beta}$ ",
                        horizontalalignment="right",
                        verticalalignment="top",
                        color=self.color,
                    )

            # adjust axes to get the numbers within the figure
            xmin, xmax = roc_ax.get_xlim()
            xmin, xmax = pr_ax.get_xlim()
            roc_ax.set_xlim([xmin - 0.04, xmax])
            pr_ax.set_xlim([xmin - 0.01, xmax])
        pr_ax.set_xlabel("Precision")
        pr_ax.set_ylabel("Recall")
        roc_ax.set_xlabel("False positive rate")
        roc_ax.set_ylabel("Recall")

    def plot_roc_prec(self):
        plt.plot(self.fpr, self.recall, self.color, zorder=1)
        plt.plot(self.fpr, self.precision, self.color, zorder=1)
        plt.show()

    def plot_roc_pr_lines(self, ax):
        ax.plot(self.fpr, np.array(self.recall) + 1, self.color, zorder=1)
        ax.plot(np.array(self.precision) * (-1) + 1, self.recall, self.color, zorder=1)
        for i in range(0, len(self.recall), 5):
            ax.plot(
                [self.fpr[i], 1 - self.precision[i]],
                [self.recall[i] + 1, self.recall[i]],
                marker="o",
                color=self.N_color,
                zorder=1,
                alpha=0.3,
            )

    def plot_distributions(
        self, axes, start=-5, stop=7, steps=1001, normalize=True, plot_xs=True, plot_os=True, plot_fs=False, threshold=0
    ):
        grid = np.linspace(start, stop, steps)
        fill_alpha = 0.2

        y = lambda x: norm.pdf(x, loc=self.N_mu, scale=self.N_std) * (1 if normalize else self.N_ampl)
        axes[0].plot(
            grid,
            y(grid),
            color=self.N_color,
            label=f"pdf_N/{self.N_ampl}",
        )

        axes[0].fill_between(
            grid[grid <= threshold], 0, y(grid[grid <= threshold]), alpha=fill_alpha, lw=0, color="darkgreen"
        )
        axes[0].fill_between(
            grid[grid >= threshold], 0, y(grid[grid >= threshold]), alpha=fill_alpha, lw=0, color="orchid"
        )
        tn_x = min(self.N_mu, threshold - 0.75)
        fp_x = max(self.N_mu, threshold + 0.75)
        axes[0].text(tn_x, y(tn_x) / 2 - 0.005, "TN", horizontalalignment="center", verticalalignment="top")
        axes[0].text(fp_x, y(fp_x) / 2 - 0.005, "FP", horizontalalignment="center", verticalalignment="top")
        # add thresholdline, on the whole y-range
        ymin, ymax = axes[0].get_ylim()
        axes[0].plot([threshold, threshold], [ymin - 1, ymax + 1], "--", color="gray", lw=1)
        axes[0].set_ylim([ymin - 0.02, ymax])

        # same for anomal distributions
        y = lambda x: norm.pdf(x, loc=self.P_mu, scale=self.P_std) * (1 if normalize else self.P_ampl)
        axes[1].plot(
            grid,
            y(grid),
            color=self.P_color,
            label=f"pdf_P/{self.P_ampl}",
        )
        axes[1].fill_between(
            grid[grid <= threshold], 0, y(grid[grid <= threshold]), alpha=fill_alpha, lw=0, color="chocolate"
        )
        axes[1].fill_between(
            grid[grid >= threshold], 0, y(grid[grid >= threshold]), alpha=fill_alpha, lw=0, color="darkcyan"
        )
        fn_x = min(self.P_mu, threshold - 0.75)
        tp_x = max(self.P_mu, threshold + 0.75)
        axes[1].text(fn_x, y(fn_x) / 2, "FN", horizontalalignment="center", verticalalignment="top")
        axes[1].text(tp_x, y(tp_x) / 2, "TP", horizontalalignment="center", verticalalignment="top")

        # add thresholdline, on the whole y-range
        ymin, ymax = axes[1].get_ylim()
        axes[1].plot([threshold, threshold], [ymin - 1, ymax + 1], "--", color="gray", lw=1)
        axes[1].set_ylim([ymin, ymax])


    def plot_cdf(self, ax, start=-6, stop=8, steps=1001, normalize=True):
        grid = np.linspace(start, stop, steps)

        ax.plot(
            grid,
            norm.cdf(grid, loc=self.N_mu, scale=self.N_std) * (1 if normalize else self.N_ampl),
            color=self.N_color,
            label=f"pdf_N/{self.N_ampl}",
        )
        ax.plot(
            grid,
            norm.cdf(grid, loc=self.P_mu, scale=self.P_std) * (1 if normalize else self.P_ampl),
            color=self.P_color,
            label=f"pdf_P/{self.P_ampl}",
        )


if __name__ == "__main__":

    # Make detector distributions

    t1 = Two_1d_normal_distributions(
        1, 49, 1.8, -1, 2, 1, color="mediumblue", betas=(1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16)
    )
    t2 = Two_1d_normal_distributions(
        1, 49, 1, -1, 1, 1, color="forestgreen", betas=(1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16)
    )

    t1.make(steps=1001, delta=0.1)
    t2.make(steps=1001, delta=0.1)

    
    #  Make roc and pr plots

    figsize = (4, 4)

    roc_fig, roc_ax = plt.subplots(figsize=figsize)
    pr_fig, pr_ax = plt.subplots(figsize=figsize)
    t1.plot_roc_pr(roc_ax, pr_ax, False, False, True)
    t2.plot_roc_pr(roc_ax, pr_ax, False, False, True)
    roc_fig.tight_layout()
    pr_fig.tight_layout()
    roc_fig.savefig("auc_roc_f.pdf")
    pr_fig.savefig("auc_pr_f.pdf")
    plt.show()
    plt.close("all")


    # Make distribution plots

    figsize = (5, 3)

    for beta in t1.betas:  # [16,4,1,1/4,1/8]:
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
        t1.plot_distributions(
            [axes[0][0], axes[1][0]], plot_xs=False, plot_os=False, plot_fs=True, threshold=t1.max_f_thresholds[beta]
        )
        t2.plot_distributions(
            [axes[0][1], axes[1][1]], plot_xs=False, plot_os=False, plot_fs=True, threshold=t2.max_f_thresholds[beta]
        )

        axes[0][0].set_title(f"Blue detector", color=t1.color)
        axes[0][1].set_title("Green detector", color=t2.color)
        axes[1][0].set_xlabel("Anomaly \n score", color=t1.color)
        axes[1][1].set_xlabel("Anomaly \n score", color=t2.color)
        shadowaxes = fig.add_subplot(111, xticks=[], yticks=[], frame_on=False)
        shadowaxes.set_ylabel("Probability density", labelpad=25)
        fig.tight_layout()
        axes[0][0].set_ylabel("Normal\nsamples", labelpad=25)
        axes[1][0].set_ylabel("Anomalous\nsamples", labelpad=25)

        plt.subplots_adjust(hspace=0.0)
        plt.savefig(f"auc_distributions_b{beta}.pdf")
        plt.show()
