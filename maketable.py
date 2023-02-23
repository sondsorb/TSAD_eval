from dataclasses import dataclass
import numpy as np

from makefig import *
import metrics


@dataclass
class Table_content:
    figure_contents: [metrics.Binary_anomalies]
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
        self.add_hline("top")
        self.add_top_row()
        self.add_hline("mid")
        for i in range(self.n_rows):
            self.add_next_row()
        self.add_hline("bottom")

    def add_hline(self, linetype="h"):
        if linetype == "h":
            self.add_line("\\hline")
        elif linetype == "top":
            self.add_line("\\toprule")
        elif linetype == "mid":
            self.add_line("\\midrule")
        elif linetype == "bottom":
            self.add_line("\\bottomrule")
        else:
            raise ValueError

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
        figure = Binary_anomalies_figure(self.content.figure_contents[number], scale=self.scale)
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

    def _lower_the_better(self, i):
        return self.content.metrics[i] == f"\\tempdist"

    def end_row(self):
        self.add_line("\\\\")


def create_table(anomalies, metric_list, length, name=None, scale=None):
    results = []
    for predicted_anomalies in anomalies[1:]:
        results.append([])
        metric_names = []
        for metric in metric_list:
            m = metric(
                length, anomalies[0], predicted_anomalies
            )  # first entry in anomalies is GT, the rest are predictions
            results[-1].append(m.get_score())
            metric_names.append(m.name)

    figure_contents = [metrics.Binary_anomalies(length, ts) for ts in anomalies]
    table_content = Table_content(figure_contents, metric_names, results)

    table = Table(table_content, name, scale)
    table.write()
    print(table)


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
    metrics.Temporal_Distance,
]
All_metrics_except_nab = list(All_metrics)
All_metrics_except_nab.remove(metrics.NAB_score)


def PA_problem():
    anomalies = [[[17, 26]], [8, 24]]
    metric_list = [metrics.Pointwise_metrics, metrics.PointAdjust]
    length = 32
    name = "pa_problem.txt"

    create_table(anomalies, metric_list, length, name, scale=2)


class Range_PR_front(metrics.Range_PR):
    def __init__(self, *args):
        super().__init__(*args, bias="front", alpha=0)


def late_early_prediction():
    early_metrics = [metrics.PointAdjust, Range_PR_front, metrics.NAB_score]
    anomalies = [[[5, 9], [15, 19]], [9], [7], [5], [9, 19], [7, 17], [5, 15]]
    length = 22
    create_table(anomalies, early_metrics, length, scale=2)


def length_problem_1():
    anomalies = [[[3, 4], [11, 12], [19, 23]], [[19, 23]], [3, 4, 11, 12]]
    length = 24
    create_table(anomalies, All_metrics, length, scale=2)


def length_problem_2():
    anomalies = [[2, 4, 6, 15, 16, 17, 18, 19], [2, 3, 4, 5, 6], [2, 15, 16, 17, 18, 19]]
    length = 22
    create_table(anomalies, All_metrics_except_nab, length, scale=2)


def short_predictions():
    anomalies = [[[14, 20]], [2, 15], [2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 20]]
    length = 22
    create_table(anomalies, All_metrics, length, scale=2)


def detection_over_covering():
    anomalies = [[[5, 12], [20, 27]], [[5, 12]], [5, 20]]  # , [[5,12],[20,27]]
    length = 28
    create_table(anomalies, All_metrics, length, scale=2)


def close_fp():
    anomalies = [[12, 13, 14, 15], [7, 8], [8, 9], [9, 10], [10, 11]]
    metric_list = [
        metrics.Affiliation,
        metrics.Time_Tolerant,
        metrics.Temporal_Distance,
    ]
    length = 17
    create_table(anomalies, metric_list, length, scale=2)


def concise():
    anomalies = [[4, 5, 8, 9, 12, 15], [4, 5, 8, 9, 15], [[3, 18]]]
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


def labelling_problem():
    anomalies = [[[18, 23]], [18, 24]]
    length = 30

    create_table(anomalies, All_metrics_except_nab, length, scale=2)

    anomalies = [[18, 24], [[18, 23]]]

    create_table(anomalies, All_metrics_except_nab, length, scale=2)


##############################################
# methods including thresholding strategires #
##############################################


class Nonbinary_Table(Table):
    def __init__(self, anomaly_scores, *args):
        self.anomaly_scores = anomaly_scores
        super().__init__(*args)
        self.x_factor = 1 / 10
        self.y_factor = 1 / 5 * 2 / self.scale
        self.y_shift = -0.1 / self.scale

    def add_fig(self, number):
        if number > 0:
            self.add_anomaly_score_fig(number)
        else:
            figure = Binary_anomalies_figure(self.content.figure_contents[number], scale=self.scale)
            figure.make()
            self.string += figure.string

    def add_anomaly_score_fig(self, number):
        self.add_line(
            f"\\begin{{tikzpicture}}[scale={self.scale}, baseline=-\\the\\dimexpr\\fontdimen22\\textfont2\\relax]"
        )
        self.add_line("\\foreach \\i/\\a in")
        self.add_line(
            str(
                [
                    (round(i * self.x_factor, 3), round(a * self.y_factor + self.y_shift, 3))
                    for i, a in enumerate(self.anomaly_scores[number - 1])
                ]
            )
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


def create_nonbinary_table(gt, anomaly_scores, metric_list, length, name=None, scale=None, AUC_TEST=False):
    results = []
    for anomaly_score in anomaly_scores:
        results_this_line = []
        metric_names = []
        for metric in metric_list:
            this_metric = metric(gt, anomaly_score)
            results_this_line.append(this_metric.get_score())
            metric_names.append(this_metric.name)
        results.append(results_this_line)

    figure_contents = [metrics.Binary_anomalies(length, gt)]
    table_content = Table_content(figure_contents, metric_names, results)

    table = Nonbinary_Table(anomaly_scores, table_content, name, scale)
    table.write()
    if AUC_TEST:  # left alignment of variable length anomaly scores
        table.string = table.string.replace("ccc", "lcc")
    print(table)


rng = np.random.default_rng()
RANDOM_TS = rng.uniform(0, 1, 9999)


def random_anomaly_score(
    length,
    binary_prediction,
    noise_amplitude=0.5,
    presmoothing_kernel=[0.25, 0.5, 0.25],
    postsmoothing_kernel=[0.25, 0.5, 0.25],
):
    anomaly_score = metrics.Binary_anomalies(length, binary_prediction).anomalies_full_series
    anomaly_score += smooth(RANDOM_TS[: len(anomaly_score)] * noise_amplitude, presmoothing_kernel)
    return smooth(anomaly_score, postsmoothing_kernel)


def smooth(anomaly_score, kernel):
    return np.convolve(
        anomaly_score,
        kernel,
        "same",
    )


def gaussian_smoothing(binary_prediction, length, std=1):
    anomaly_score = metrics.Binary_anomalies(length, binary_prediction).anomalies_full_series
    smooth_score = np.zeros(length)
    indices = np.arange(length)
    for i, point_val in enumerate(anomaly_score):
        smooth_score += point_val * np.exp(-((indices - i) ** 2) / (2 * std))
    return smooth_score


Nonbinary_metrics = [
    metrics.PatK_pw,
    metrics.Best_threshold_pw,
    metrics.AUC_ROC,
    metrics.AUC_PR_pw,
    metrics.VUS_ROC,
    metrics.VUS_PR,
]
ROC_metrics = [
    metrics.AUC_ROC,
    metrics.VUS_ROC,
]


def nonbinary_labelling_problem():
    length = 30
    anomaly_scores = [random_anomaly_score(length, [18, 24], postsmoothing_kernel=[1])]
    gt = [[18, 23]]
    create_nonbinary_table(gt, anomaly_scores, Nonbinary_metrics, length, scale=2)

    gt = [18, 24]
    anomaly_scores = [random_anomaly_score(length, [[18, 23]], postsmoothing_kernel=[1])]
    create_nonbinary_table(gt, anomaly_scores, Nonbinary_metrics, length, scale=2)


def nonbinary_detection_over_covering():
    gt = [[5, 12], [20, 27]]
    length = 28
    anomaly_scores = [metrics.Binary_anomalies(length, x).anomalies_full_series for x in [[[5, 12]], [5, 20]]]
    create_nonbinary_table(gt, anomaly_scores, Nonbinary_metrics, length, scale=1.2)


def auc_roc_problem():
    gt = [14, 15, 16]
    length = 45
    anomaly_scores = [metrics.Binary_anomalies(length, x).anomalies_full_series for x in [[14, 15], [[12, 20]]]]
    create_nonbinary_table(gt, anomaly_scores, Nonbinary_metrics, length, scale=1.5)


def nonbinary_length_problem_1():
    gt = [[3, 4], [11, 12], [19, 25]]
    length = 27
    anomaly_scores = [random_anomaly_score(length, x, noise_amplitude=0) for x in [[[19, 25]], [3, 4, 11, 12]]]
    create_nonbinary_table(gt, anomaly_scores, Nonbinary_metrics, length, scale=1.2)


def score_value_problem():
    gt = [[8, 10]]
    length = 21
    x = np.arange(21)
    anomaly_scores = 1 / (1 + abs(x - 10)), (10.1 - abs(x - 10)) ** 0.5 / 3
    create_nonbinary_table(gt, anomaly_scores, Nonbinary_metrics, length, scale=1.5)


def nonbinary_close_fp():
    length = 17
    gt = [12, 13, 14]
    anomaly_scores = [gaussian_smoothing(x, length, std=2) for x in [[8], [9], [10], [11]]]
    create_nonbinary_table(gt, anomaly_scores, Nonbinary_metrics, length, scale=2)


def nonbinary_nonsmooth_close_fp():
    length = 17
    gt = [12, 13, 14]
    anomaly_scores = [metrics.Binary_anomalies(length, x).anomalies_full_series for x in [[8], [9], [10], [11]]]
    create_nonbinary_table(gt, anomaly_scores, Nonbinary_metrics, length, scale=2)


def auc_roc_problem_2():
    length = 64
    gt = [[10, 14]]
    anomaly_scores = [
        metrics.Binary_anomalies(length, pred).anomalies_full_series
        for length in [32, 64]
        for pred in [[10, 11, 12, 13], [[3, 15]]]
    ]
    create_nonbinary_table(gt, anomaly_scores, ROC_metrics, length, scale=1.2, AUC_TEST=True)


def auc_roc_problem_3():
    length = 64
    gt = [[4, 7]]
    anomaly_scores = [gaussian_smoothing(pred, length, std=10) for length in [8, 16, 32, 64] for pred in [[0]]]
    create_nonbinary_table(gt, anomaly_scores, ROC_metrics, length, scale=1.2, AUC_TEST=True)


def nonbinary_short_predictions():
    length = 22
    gt = [[14, 20]]
    anomaly_scores = [
        random_anomaly_score(length, x, noise_amplitude=0)
        for x in [[2, 15], [2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 20]]
    ]
    create_nonbinary_table(gt, anomaly_scores, Nonbinary_metrics, length, scale=1.5)


if __name__ == "__main__":
    print("\\newcommand{\\showLengthProblemI}[0]{")
    length_problem_1()
    print("}")

    print("\\newcommand{\\showLengthProblemII}[0]{")
    length_problem_2()
    print("}")

    print("\\newcommand{\\showShortPrecitions}[0]{")
    short_predictions()
    print("}")

    print("\\newcommand{\\showDetectionOverCovering}[0]{")
    detection_over_covering()
    print("}")

    print("\\newcommand{\\showLabellingProblem}[0]{")
    labelling_problem()
    print("}")

    print("\\newcommand{\\showNonbinaryDetectionOverCovering}[0]{")
    nonbinary_detection_over_covering()
    print("}")

    print("\\newcommand{\\showNonbinaryLengthI}[0]{")
    nonbinary_length_problem_1()
    print("}")

    print("\\newcommand{\\showScoreValueProblem}[0]{")
    score_value_problem()
    print("}")

    print("\\newcommand{\\showNonbinaryCloseFp}[0]{")
    nonbinary_close_fp()
    print("}")

    print("\\newcommand{\\showNonbinaryNonsmoothCloseFp}[0]{")
    nonbinary_nonsmooth_close_fp()
    print("}")

    print("\\newcommand{\\showAucRocProblemII}[0]{")
    auc_roc_problem_2()
    print("}")

    print("\\newcommand{\\showAucRocProblemIII}[0]{")
    auc_roc_problem_3()
    print("}")

    print("\\newcommand{\\showNonbinaryShortPredictions}[0]{")
    nonbinary_short_predictions()
    print("}")
