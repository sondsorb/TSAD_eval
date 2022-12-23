import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

class Two_normal_distributions:
    def __init__(self, P_ampl, N_ampl, P_mu, N_mu, P_std, N_std):
        self.P_ampl=P_ampl
        self.N_ampl=N_ampl
        self.P_mu=P_mu
        self.N_mu=N_mu
        self.P_std=P_std
        self.N_std=N_std

    def plot(self, delta=0.05, steps=10001, start=-5, stop=5):
        index = 0
        fpr = []
        precision = []
        recall = []
        x_fpr = []
        x_precision = []
        x_recall = []
        xs = 0
        x_threshold = []
        o_fpr = []
        o_precision = []
        o_recall = []
        os = 0
        o_threshold = []
        for threshold in np.linspace(start, stop, steps):
            TN = self.N_ampl*norm.cdf((threshold-self.N_mu)/self.N_std)
            FP = self.N_ampl - TN
            FN = self.P_ampl*norm.cdf((threshold-self.P_mu)/self.P_std)
            TP = self.P_ampl - FN
            fpr.append( FP/(FP+TN) )
            precision.append( TP/(TP+FP) )
            recall.append( TP/(TP+FN) )

            if (FN)/(self.P_ampl) >= xs*delta:
                x_fpr.append( FP/(FP+TN) )
                x_precision.append( TP/(TP+FP) )
                x_recall.append( TP/(TP+FN) )
                xs += 1
                #print(xs, TN+FN)
                x_threshold.append(threshold)
            if (TN/self.N_ampl) >= os*delta:
                o_fpr.append( FP/(FP+TN) )
                o_precision.append( TP/(TP+FP) )
                o_recall.append( TP/(TP+FN) )
                os += 1
                o_threshold.append(threshold)
                #print(os, TN+FN)
        plt.plot(fpr, recall)
        plt.plot(precision, recall)
        plt.plot(x_fpr, x_recall, 'x')
        plt.plot(x_precision, x_recall, 'x')
        plt.plot(o_fpr, o_recall, 'o')
        plt.plot(o_precision, o_recall, 'o')
        plt.show()

        grid = np.linspace(start, stop, steps)
        plt.plot(grid, norm.pdf((grid - self.N_mu)/self.N_std), label=f"pdf_N/{self.N_ampl}")
        plt.plot(grid, norm.pdf((grid - self.P_mu)/self.P_std), label=f"pdf_P/{self.P_ampl}")
        plt.plot(o_threshold, norm.pdf((np.array(o_threshold) - self.N_mu)/self.N_std), "o")
        plt.plot(x_threshold, norm.pdf((np.array(x_threshold) - self.P_mu)/self.P_std), "x")
        plt.legend()
        plt.show()




if __name__ == '__main__':

    t = Two_normal_distributions(1,50,2,-1,1,1)

    t.plot(steps = 1001, delta=0.1)


