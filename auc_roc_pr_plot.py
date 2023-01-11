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
        self.betas = (1/8,1/4,1/2, 1,2,4,8,16,32) if betas == None else betas

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
        self.max_f = {i:0 for i in self.betas}
        self.max_f_fpr = {i:0 for i in self.betas}
        self.max_f_precision = {i:0 for i in self.betas}
        self.max_f_recall = {i:0 for i in self.betas}
        self.max_f_thresholds = {i:0 for i in self.betas}

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
            roc_ax.plot(list(self.max_f_fpr.values()), list(self.max_f_recall.values()), ".", linestyle= "None", zorder=2, color="k")#self.color)
            pr_ax.plot(list(self.max_f_precision.values()), list(self.max_f_recall.values()), ".", zorder=2, color="k")#self.color)
            for beta in self.betas:
                #roc_ax.plot([self.max_f_fpr[beta]],[self.max_f_recall[beta]], marker=f"$1/{int(1/beta)}$" if beta<1 else f"${beta}$", linestyle= "None", zorder=2, color="k")
                #pr_ax.plot([self.max_f_precision[beta]], [self.max_f_recall[beta]], marker=f"$1/{int(1/beta)}$" if beta<1 else f"${beta}$", zorder=2, color="k")
                if self.color == "g":
                    roc_ax.text(self.max_f_fpr[beta],self.max_f_recall[beta], f"$1/{int(1/beta)}$" if beta<1 else f"${beta}$",horizontalalignment="left", verticalalignment="top")
                else:
                    roc_ax.text(self.max_f_fpr[beta],self.max_f_recall[beta], f"$1/{int(1/beta)}$" if beta<1 else f"${beta}$", horizontalalignment="right", verticalalignment="bottom")
                pr_ax.text(self.max_f_precision[beta], self.max_f_recall[beta],f"$1/{int(1/beta)}$" if beta<1 else f"${beta}$",horizontalalignment="left", verticalalignment="bottom")
        pr_ax.set_xlabel("Precision")
        pr_ax.set_ylabel("Recall")
        roc_ax.set_xlabel("False positive rate")
        roc_ax.set_ylabel("Recall")

    def plot_roc_prec(self):
        plt.plot(self.fpr, self.recall, self.color, zorder=1)
        plt.plot(self.fpr, self.precision, self.color, zorder=1)
        plt.show()

    def plot_roc_pr_lines(self, ax):
        ax.plot(self.fpr, np.array(self.recall)+1, self.color, zorder=1)
        ax.plot(np.array(self.precision)*(-1)+1, self.recall, self.color, zorder=1)
        #for i in range(len(self.x_fpr)):
        #    plt.plot([self.x_fpr[i], 1-self.x_precision[i]], [self.x_recall[i]+1,self.x_recall[i]], marker="x", color=self.P_color, zorder=1, alpha=0.3)
        #for i in range(len(self.o_fpr)):
        #    plt.plot([self.o_fpr[i], 1-self.o_precision[i]], [self.o_recall[i]+1,self.o_recall[i]], marker="o", color=self.N_color, zorder=1, alpha=0.3)
        for i in range(0,len(self.recall), 5):
            ax.plot([self.fpr[i], 1-self.precision[i]], [self.recall[i]+1,self.recall[i]], marker="o", color=self.N_color, zorder=1, alpha=0.3)


    def plot_distributions(self, ax, start=-5, stop=8, steps=1001, normalize=True, plot_xs=True, plot_os=True, plot_fs=False):
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
        if plot_os:
            ax.plot(self.o_threshold, np.ones(self.os)*(-0.01), "o", fillstyle="none", color=self.color)
        if plot_xs:
            ax.plot(self.x_threshold, np.ones(self.xs)*(-0.02), "x", color=self.color)
        if plot_fs:
            #ax.plot(list(self.max_f_thresholds.values()),np.ones(len(self.betas))*(-0.02), "|", linestyle= "None", zorder=2, color="k")#self.color)
            #for i, beta in enumerate(self.betas):
            #    ax.text(self.max_f_thresholds[beta]-0.07,-0.025+0.01*(-1)**i, f"$1/{int(1/beta)}$" if beta<1 else f"${beta}$")
            ax.plot(list(self.max_f_thresholds.values()),np.ones(len(self.betas))*(0), "|", linestyle= "None", zorder=2, color="k")#self.color)
            ax.plot(list(self.max_f_thresholds.values())[1::2],np.ones(len(self.betas[1::2]))*(-0.005), "|", linestyle= "None", zorder=2, color="k")#self.color)
            for beta in self.betas[1::2]:
                ax.text(self.max_f_thresholds[beta],-0.025, f"$1/{int(1/beta)}$" if beta<1 else f"${beta}$", horizontalalignment='center')
        #ax.legend()
        ax.xlabel("Threshold")
        ax.ylabel("Probability density")
        #ax.grid()

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

    t1 = Two_1d_normal_distributions(1, 49, 1.8, -1, 2, 1, color="b", betas = (1/8,1/4,1/2,1,2,4,8,16))
    t2 = Two_1d_normal_distributions(1, 49, 1, -1, 1, 1, color="g", betas = (1/8,1/4,1/2,1,2,4,8,16))

    t1.make(steps=1001, delta=0.1)
    t2.make(steps=1001, delta=0.1)

    #Local visualtization:

    #fig, axes = plt.subplots(2, 1)
    #t1.plot_roc_pr(axes[0], axes[1])
    #t2.plot_roc_pr(axes[0], axes[1])
    #plt.show()

    #fig, axes = plt.subplots(1,2)
    #t1.plot_distributions(axes[0], plot_xs=False, plot_os=False,plot_fs=True)
    #t2.plot_distributions(axes[1], plot_xs=False, plot_os=False,plot_fs=True)
    #plt.show()

    #quit()

    # Save plots:

    figsize=(5,5)

    #plt.figure(figsize=figsize)
    #t1.plot_roc_pr_lines(plt)
    #plt.tight_layout()
    #plt.savefig("lines_1.pdf")
    #plt.show()
    #plt.close("all")
    #plt.figure(figsize=figsize)
    #t2.plot_roc_pr_lines(plt)
    #plt.tight_layout()
    #plt.savefig("lines_2.pdf")
    #plt.show()
    #plt.close("all")

    #roc_fig, roc_ax = plt.subplots(figsize=figsize)
    #plt.tight_layout()
    #pr_fig, pr_ax = plt.subplots(figsize=figsize)
    #plt.tight_layout()
    #t1.plot_roc_pr(roc_ax, pr_ax)
    #t2.plot_roc_pr(roc_ax, pr_ax)
    #roc_fig.savefig("auc_roc.pdf")
    #pr_fig.savefig("auc_pr.pdf")
    #plt.show()
    #plt.close("all")

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

    plt.figure(figsize=figsize)
    t1.plot_distributions(plt, plot_xs=False, plot_os=False,plot_fs=True)
    plt.tight_layout()
    plt.savefig("auc_distributions_1.pdf")
    plt.show()
    plt.close("all")
    plt.figure(figsize=figsize)
    t2.plot_distributions(plt, stop=5, plot_xs=False, plot_os=False,plot_fs=True)
    plt.tight_layout()
    plt.savefig("auc_distributions_2.pdf")
    plt.show()
