from dataclasses import dataclass
import numpy as np

from makefig import *
import metrics


@dataclass
class Table_content:

    figure_contents: [Figure_content]
    metrics: [str]
    results: []


class Table:
    def __init__(self, table_content, filename="table1.txt", scale=None):
        self.string = ""
        self.filename = filename
        self.content = table_content
        self.scale = scale

        self.row_length = len(self.content.metrics) + 1
        self.n_rows = len(self.content.results)
        self.rows_added = 0

    def make(self):
        self.add_line(f"""\\begin{{tabular}}[t]{{{"c"*self.row_length}}}""")
        self.add_all_content()
        self.add_line("\\end{tabular}")

    def add_line(self, line):
        self.string += line
        self.string += "\n"

    def write(self):
        self.make()
        if self.filename != None:
            with open(self.filename, "w") as file:
                file.write(self.string)

    def __str__(self):
        return self.string

    def add_all_content(self):

        self.add_hline()
        self.add_top_row()
        self.add_hline()
        for i in range(self.n_rows):
            self.add_next_row()
        self.add_hline()

    def add_hline(self):
        self.add_line("\\hline")

    def add_top_row(self):
        self.add_fig(0)

        for entry in self.content.metrics:
            self.add_entry(entry)
        self.end_row()

    def add_next_row(self):
        self.add_fig(self.rows_added + 1)

        for i, entry in enumerate(self.content.results[self.rows_added]):
            self.add_entry(entry, bold=self._is_optimal(entry, i))
        self.end_row()

        self.rows_added += 1

    def add_fig(self, number):
        figure = Figure(self.content.figure_contents[number], scale=self.scale)
        figure.make()
        self.string += figure.string

    def add_entry(self, entry, bold=False):
        if type(entry) in (float, np.float64):
            entry = round(entry, 2)
        if bold:
            self.string += f"&\\textbf{{{entry}}}"
        else:
            self.string += f"&{entry}"

    def _is_optimal(self, entry, i):
        if self._lower_the_better(i):
            return entry == min(np.array(self.content.results)[:, i])
        return entry == max(np.array(self.content.results)[:, i])

    def _lower_the_better(self,i):
        return self.content.metrics[i] == f"\\tempdist"

    def end_row(self):
        self.add_line("\\\\")


def distance_problem():
    figure_contents = []

    anomalies = [[[[270, 275], [290, 295]]], [[[240, 245], [290, 295]]], [[[265, 270], [285, 290]]]]
    metrics = ["AF"]
    results = [[0.7], [0.1]]  # NOTE not correct!

    figure_contents = [Figure_content(None, 300, anomaly) for anomaly in anomalies]
    table_content = Table_content(figure_contents, metrics, results)

    table = Table(table_content)
    table.write()


def create_table(anomalies, metric_list, length, name=None, scale=None):
    results = [
        [metric(length, anomalies[0], predicted_anomalies).get_score() for metric in metric_list]
        for predicted_anomalies in anomalies[1:]
    ]

    figure_contents = make_figure_content(length, anomalies)
    table_content = Table_content(
        figure_contents, [metric(length, anomalies[0], anomalies[1]).name for metric in metric_list], results
    )

    table = Table(table_content, name, scale)
    table.write()
    print(table)


def make_figure_content(
    length, anomalies
):  # TODO cleanup this (this function only exist because "anomalies" in makefig is a mess
    figure_anomalies = []
    for ts in anomalies:
        ts = metrics.Binary_detection(length, ts, [])
        figure_anomalies.append([ts.get_gt_anomalies_segmentwise().tolist()])
    return [Figure_content(None, length, anomaly) for anomaly in figure_anomalies]


######################
## Example problems ##
######################

All_metrics = [
    metrics.Pointwise_metrics,
    metrics.PointAdjust,
    metrics.DelayThresholdedPointAdjust,
    metrics.PointAdjustKPercent,
    metrics.LatencySparsityAware,
    metrics.Segmentwise_metrics,
    metrics.Composite_f,
    metrics.Time_Tolerant,
    metrics.Range_PR,
    metrics.TaF,
    metrics.eTaF,
    metrics.Affiliation,
    metrics.NAB_score,
    metrics.Temporal_Distance
]

def PA_problem():
    anomalies = [[[25, 39]], [12, 36]]
    metric_list = [metrics.Pointwise_metrics, metrics.PointAdjust]
    length = 48
    name = "pa_problem.txt"

    create_table(anomalies, metric_list, length, name, scale=2)


class Range_PR_front(metrics.Range_PR):
    def __init__(self, *args):
        super().__init__(*args, bias="front", alpha=0)
def late_early_prediction():
    early_metrics = [metrics.PointAdjust, Range_PR_front, metrics.NAB_score]
    # anomalies = [[[5, 9], [15, 19]], [8,9], [7,8], [6,7], [5,6], [8,9,18,19],[7,8,17,18],[6,7,16,17], [5,6,15,16]]
    anomalies = [[[5, 9], [15, 19]], [9], [7], [5], [9, 19], [7, 17], [5, 15]]
    length = 22
    create_table(anomalies, early_metrics, length, scale=2)

def length_problem_1():
    # anomalies = [[[5,6], [15, 19]], [15,16,17,18,19], [5,6,15,16]]
    anomalies = [[[3, 4], [11, 12], [19, 23]], [[19, 23]], [3, 4, 11, 12]]
    length = 24
    create_table(anomalies, All_metrics, length, scale=2)


def length_problem_2():
    anomalies = [[2, 4, 6, 15, 16, 17, 18, 19], [2, 3, 4, 5, 6], [2, 15, 16, 17, 18, 19]]
    length = 22
    create_table(anomalies, All_metrics, length, scale=2)


def short_predictions():
    anomalies = [[[14, 20]], [3, 16], [2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 20]]
    length = 22
    create_table(anomalies, All_metrics, length, scale=2)


def detection_over_covering():
    anomalies = [[[5, 12], [20, 27]], [[5, 12]], [5, 20]]  # , [[5,12],[20,27]]
    length = 28
    create_table(anomalies, All_metrics, length, scale=2)


def close_fp():
    anomalies = [[12, 13, 14, 15], [7, 8], [8, 9], [9, 10], [10, 11]]
    metric_list = [
        # metrics.Pointwise_metrics, metrics.PointAdjust, metrics.Segmentwise_metrics, metrics.Composite_f, metrics.NAB_score,
        metrics.Affiliation,
        metrics.Time_Tolerant,
        metrics.Temporal_Distance
    ]
    length = 17
    create_table(anomalies, metric_list, length, scale=2)

def concise():
    anomalies = [[4, 5, 8, 9, 12,15], [4, 5, 8, 9, 15], [[3, 18]]]
    length = 22
    create_table(anomalies, All_metrics, length, scale=2)

def af_problem():
    anomalies = [[29, 30, 35, 36], [25, 26, 35, 36], [29, 30, 34, 35]]
    metric_list = [
        metrics.Affiliation,
        metrics.Temporal_Distance,
    ]
    length = 38
    create_table(anomalies, metric_list, length, scale=2)

# def labelling_problem():
#    anomalies = [[[6,7],[30,39]], [[15,16],[30,33]],[6,7,30,40], [[6,7],[15,16],[30,39]]]
#    metric_list = [
#            metrics.Pointwise_metrics, metrics.PointAdjust, metrics.Segmentwise_metrics, metrics.Composite_f, #metrics.NAB_score,
#            metrics.Range_PR,
#            metrics.Affiliation]
#    length = 50
#
#    create_table(anomalies, metric_list, length, scale=2)
#
#    anomalies[0] = [6,7,30,40]
#
#    create_table(anomalies, metric_list, length, scale=2)


def labelling_problem():
    anomalies = [[[18, 23]], [18, 24]]
    length = 30

    create_table(anomalies, All_metrics, length, scale=2)

    anomalies = [[18, 24], [[18, 23]]]

    create_table(anomalies, All_metrics, length, scale=2)


##############################################
# methods including thresholding strategires #
##############################################


class Nonbinary_Table(Table):
    def __init__(self, anomaly_scores, *args):
        self.anomaly_scores = anomaly_scores
        super().__init__(*args)
        self.x_factor = 1 / 10
        self.y_factor = 1 / 5 * 2 / self.scale

    def add_fig(self, number):
        if number > 0:
            self.add_anomaly_score_fig(number)
        else:
            figure = Figure(self.content.figure_contents[number], scale=self.scale)
            figure.make()
            self.string += figure.string

    def add_anomaly_score_fig(self, number):
        self.add_line(
            f"\\begin{{tikzpicture}}[scale={self.scale}, baseline=-\\the\\dimexpr\\fontdimen22\\textfont2\\relax]"
        )
        self.add_line("\\foreach \\i/\\a in")
        self.add_line(
            str([(round(i * self.x_factor,3), round(a * self.y_factor,3)) for i, a in enumerate(self.anomaly_scores[number - 1])])
            .replace(",", "/")
            .replace(")/", ",")
            .replace("(", "")
            .replace("[", "{")
            .replace("]", "}")
            .replace(")}", "}{")
        )
        self.add_line("\\coordinate (now) at (\\i,\\a) {};")
        self.add_line("  \\ifthenelse{\\equal{\\i}{0.0}}{}{")
        self.add_line("  \\draw[-, blue] (prev) -- (now);")
        self.add_line("  }")
        self.add_line("  \\coordinate (prev) at (\\i,\\a) {};")
        self.add_line("}")
        self.add_line("\\end{tikzpicture}")

rng = np.random.default_rng()
RANDOM_TS = rng.uniform(0,1,9999)
def random_anomaly_score(length,binary_prediction, noise_amplitude = 0.5, presmoothing_kernel = [.25,.5,.25], postsmoothing_kernel = [.25,.5,.25]):

    anomaly_score = metrics.Binary_anomalies(length,binary_prediction).anomalies_binary
    anomaly_score += smooth(RANDOM_TS[:len(anomaly_score)]*noise_amplitude, presmoothing_kernel)#rng.uniform(0, noise_amplitude, len(anomaly_score)), presmoothing_kernel)
    return smooth(anomaly_score, postsmoothing_kernel)

def smooth(anomaly_score, kernel):
    return np.convolve(
        anomaly_score,
        kernel,
        # [0.25, 0.5, 0.25],
        # [.1,0.2,0.4,0.2,.1],
        # [.05,0.1,0.7,0.1,.05],
        "same",
    )

def gaussian_smoothing(binary_prediction, length, std = 1):
    anomaly_score = metrics.Binary_anomalies(length,binary_prediction).anomalies_binary
    smooth_score = np.zeros(length)
    indices = np.arange(length)
    for i, point_val in enumerate(anomaly_score):
        smooth_score += point_val * np.exp(-(indices-i)**2/(2*std))
    return smooth_score

Nonbinary_metrics= [
    metrics.PatK_pw,
    metrics.Best_threshold_pw,
    metrics.AUC_ROC,
    metrics.AUC_PR_pw,
    metrics.VUS_ROC,
    metrics.VUS_PR,
]
def nonbinary_labelling_problem():
    length = 30
    anomaly_scores = [random_anomaly_score(length,[18, 24], postsmoothing_kernel=[1])]
    gt = [[18, 23]]
    create_nonbinary_table(gt, anomaly_scores, Nonbinary_metrics, length, scale=2)

    gt = [18, 24]
    anomaly_scores = [random_anomaly_score(length,[[18, 23]], postsmoothing_kernel=[1])]
    create_nonbinary_table(gt, anomaly_scores, Nonbinary_metrics, length, scale=2)

def nonbinary_detection_over_covering():
    gt = [[5,12], [35,42]]
    length = 45
    anomaly_scores = [random_anomaly_score(length, x, noise_amplitude=0, postsmoothing_kernel=[1]) for x in [[[5,12]], [5,35]] ]
    create_nonbinary_table(gt, anomaly_scores, Nonbinary_metrics, length, scale=1.5)

def auc_roc_problem():
    gt = [14,15,16]
    length = 45
    anomaly_scores = [random_anomaly_score(length, x, postsmoothing_kernel=[1], noise_amplitude=0) for x in [[14,15], [[12,20]]] ]
    create_nonbinary_table(gt, anomaly_scores, Nonbinary_metrics, length, scale=1.5)


def nonbinary_length_problem_1():

    gt = [[3,4], [11,12], [19, 25]]
    length = 27
    anomaly_scores = [random_anomaly_score(length, x, noise_amplitude=0) for x in [[[19,25]], [3,4,11,12]] ]
    create_nonbinary_table(gt, anomaly_scores, Nonbinary_metrics, length, scale=2)

def score_value_problem():

    gt = [[8,10]]
    length = 21
    x = np.arange(21)
    anomaly_scores = 1/(1+abs(x-10)), (10.1-abs(x-10))**0.5/3
    create_nonbinary_table(gt, anomaly_scores, Nonbinary_metrics, length, scale=2)

def create_nonbinary_table(gt, anomaly_scores, metric_list, length, name=None, scale=None):
    results = []
    for anomaly_score in anomaly_scores: 
        results_this_line = []
        metric_names = []
        for metric in metric_list:
            this_metric = metric(gt, anomaly_score)
            results_this_line.append(this_metric.get_score())
            metric_names.append(this_metric.name)
        results.append(results_this_line)

    figure_contents = make_figure_content(length, [gt])
    table_content = Table_content(
        figure_contents, metric_names, results
    )

    table = Nonbinary_Table(anomaly_scores, table_content, name, scale)
    table.write()
    print(table)

def nonbinary_close_fp():

    length = 17
    gt = [12, 13, 14]
    anomaly_scores = [gaussian_smoothing(x, length, std=2) for x in [[8], [9], [10], [11]]]
    create_nonbinary_table(gt, anomaly_scores, Nonbinary_metrics, length, scale=2)

def auc_roc_problem_2():
    length = 64
    gt = [[8,12]]
    anomaly_scores = [random_anomaly_score(length, pred, noise_amplitude=0, postsmoothing_kernel=[1]) for length in [32,64] for pred in [[8,9,10,11],[[3,13]]]]
    create_nonbinary_table(gt, anomaly_scores, Nonbinary_metrics, length, scale=0.8)

def auc_roc_problem_3():
    length = 128
    gt = [[10,14]]
    anomaly_scores = [gaussian_smoothing(pred, length, std=10) for length in [16,32,64,128] for pred in [[0]]]
    create_nonbinary_table(gt, anomaly_scores, Nonbinary_metrics, length, scale=0.8)

def nonbinary_short_predictions():
    length = 22
    gt = [[14, 20]]
    anomaly_scores = [random_anomaly_score(length, x, noise_amplitude=0) for x in [[3, 16], [2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 20]]]
    create_nonbinary_table(gt,anomaly_scores, Nonbinary_metrics, length, scale=2)


if __name__ == "__main__":
    #nonbinary_short_predictions()
    #nonbinary_detection_over_covering()
    print("auc_roc_problem")
    auc_roc_problem()
    print("auc_roc_problem_2")
    auc_roc_problem_2()
    quit()
    #print("PA problem")
    #PA_problem()
    print("late_early_prediction")
    late_early_prediction()
    print("length_problem_1")
    length_problem_1()
    print("length_problem_2")
    length_problem_2()
    print("short_predictions")
    short_predictions()
    print("detection_over_covering")
    detection_over_covering()
    print("close_fp")
    close_fp()
    print("concise")
    concise()
    print("af_problem")
    af_problem()
    print("labelling_problem")
    labelling_problem()

    ##threshold_test()
    print("nonbinary_labelling")
    nonbinary_labelling_problem()
    print("nonbinary_detection_over_covering")
    nonbinary_detection_over_covering()
    print("auc_roc_problem")
    auc_roc_problem()
    print("nonbinary_length_problem_1")
    nonbinary_length_problem_1()
    print("score_value_problem")
    score_value_problem()
    print("nonbinary_close_fp")
    nonbinary_close_fp()
    print("auc_roc_problem_2")
    auc_roc_problem_2()
    print("auc_roc_problem_3")
    auc_roc_problem_3()
