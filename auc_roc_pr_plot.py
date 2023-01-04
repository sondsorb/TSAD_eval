import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

from metrics import f1_from_pr


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
        self.betas = (1/8,1/4,1/2, 1,2,4,8,16,32)
        self.max_f = {i:0 for i in self.betas}
        self.max_f_fpr = {i:0 for i in self.betas}
        self.max_f_precision = {i:0 for i in self.betas}
        self.max_f_recall = {i:0 for i in self.betas}

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

            if (FN) / (self.P_ampl) >= self.xs * delta + delta*0.5:
                self.x_fpr.append(FP / (FP + TN))
                self.x_precision.append(TP / (TP + FP))
                self.x_recall.append(TP / (TP + FN))
                self.xs += 1
                # print(xs, TN+FN)
                self.x_threshold.append(threshold)
            if (TN / self.N_ampl) >= self.os * delta + delta*0.5:
                self.o_fpr.append(FP / (FP + TN))
                self.o_precision.append(TP / (TP + FP))
                self.o_recall.append(TP / (TP + FN))
                self.os += 1
                self.o_threshold.append(threshold)
                # print(os, TN+FN)

    def plot_roc_pr(self, roc_ax, pr_ax, plot_xs=True, plot_ys=True, plot_fs=False):
        roc_ax.plot(self.fpr, self.recall, self.color, zorder=1)
        pr_ax.plot(self.precision, self.recall, self.color, zorder=1)
        if plot_xs:
            roc_ax.plot(self.x_fpr, self.x_recall, "x", color=self.color, zorder=1)
            pr_ax.plot(self.x_precision, self.x_recall, "x", color=self.color, zorder=1)
        if plot_xs:
            roc_ax.plot(self.o_fpr, self.o_recall, "o", color=self.color, fillstyle="none", zorder=1)
            pr_ax.plot(self.o_precision, self.o_recall, "o", color=self.color, fillstyle="none", zorder=1)
        if plot_fs:
            roc_ax.plot(list(self.max_f_fpr.values()), list(self.max_f_recall.values()), ".", linestyle= "None", zorder=2, color="k")#self.color)
            pr_ax.plot(list(self.max_f_precision.values()), list(self.max_f_recall.values()), ".", zorder=2, color="k")#self.color)
            for beta in self.betas:
                #roc_ax.plot([self.max_f_fpr[beta]],[self.max_f_recall[beta]], marker=f"$1/{int(1/beta)}$" if beta<1 else f"${beta}$", linestyle= "None", zorder=2, color="k")
                #pr_ax.plot([self.max_f_precision[beta]], [self.max_f_recall[beta]], marker=f"$1/{int(1/beta)}$" if beta<1 else f"${beta}$", zorder=2, color="k")
                roc_ax.text(self.max_f_fpr[beta]+0.01,self.max_f_recall[beta]-0.02, f"$1/{int(1/beta)}$" if beta<1 else f"${beta}$")
                pr_ax.text(self.max_f_precision[beta]+0.01, self.max_f_recall[beta]+0.01,f"$1/{int(1/beta)}$" if beta<1 else f"${beta}$")

    def plot_distributions(self, ax, start=-5, stop=8, steps=1001, normalize=True):
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
        ax.plot(self.o_threshold, np.ones(self.os)*(-0.01), "o", fillstyle="none", color=self.color)
        ax.plot(self.x_threshold, np.ones(self.xs)*(-0.02), "x", color=self.color)
        # plt.plot(self.o_threshold, norm.pdf((np.array(self.o_threshold) - self.N_mu)/self.N_std), "o")
        # plt.plot(self.x_threshold, norm.pdf((np.array(self.x_threshold) - self.P_mu)/self.P_std), "x")
        ax.legend()

    def plot_cdf(self, ax, start=-6, stop=8, steps=1001, normalize=True):
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

    t1 = Two_1d_normal_distributions(1, 49, 1.8, -1, 2, 1, color="b")
    t2 = Two_1d_normal_distributions(1, 49, 1, -1, 1, 1, color="g")

    t1.make(steps=1001, delta=0.1)
    t2.make(steps=1001, delta=0.1)

    #Local visualtization:

    #fig, axes = plt.subplots(2, 1)
    #t1.plot_roc_pr(axes[0], axes[1])
    #t2.plot_roc_pr(axes[0], axes[1])
    #plt.show()

    #fig, axes = plt.subplots(1,2)
    #t1.plot_distributions(axes[0])
    #t2.plot_distributions(axes[1])
    #plt.show()


    # Save plots:

    figsize=(5,5)

    roc_fig, roc_ax = plt.subplots(figsize=figsize)
    plt.tight_layout()
    pr_fig, pr_ax = plt.subplots(figsize=figsize)
    plt.tight_layout()
    t1.plot_roc_pr(roc_ax, pr_ax)
    t2.plot_roc_pr(roc_ax, pr_ax)
    roc_fig.savefig("auc_roc.pdf")
    pr_fig.savefig("auc_pr.pdf")
    plt.show()
    plt.close("all")

    roc_fig, roc_ax = plt.subplots(figsize=figsize)
    plt.tight_layout()
    pr_fig, pr_ax = plt.subplots(figsize=figsize)
    plt.tight_layout()
    t1.plot_roc_pr(roc_ax, pr_ax, False, False, True)
    t2.plot_roc_pr(roc_ax, pr_ax, False, False, True)
    roc_fig.savefig("auc_roc_f.pdf")
    pr_fig.savefig("auc_pr_f.pdf")
    plt.show()
    plt.close("all")

    plt.figure(figsize=figsize)
    t1.plot_distributions(plt)
    plt.tight_layout()
    plt.savefig("auc_distributions_1.pdf")
    plt.show()
    plt.close("all")
    plt.figure(figsize=figsize)
    t2.plot_distributions(plt, stop=5)
    plt.tight_layout()
    plt.savefig("auc_distributions_2.pdf")
    plt.show()
