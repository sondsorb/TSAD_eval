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

        for entry in self.content.results[self.rows_added]:
            self.add_entry(entry)
        self.end_row()

        self.rows_added += 1

    def add_fig(self, number):
        figure = Figure(self.content.figure_contents[number], scale=self.scale)
        figure.make()
        self.string += figure.string

    def add_entry(self, entry):
        if type(entry) in (float, np.float64):
            entry = round(entry, 2)
        self.string += f"&{entry}"

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
    table_content = Table_content(figure_contents, [metric(length, [], []).name for metric in metric_list], results)

    table = Table(table_content, name, scale)
    table.write()
    print(table)


def make_figure_content(
    length, anomalies
):  # TODO cleanup this (this function only exist because "anomalies" in makefig is a mess
    figure_anomalies = []
    for ts in anomalies:
        ts = metrics.Detected_anomalies(length, ts, [])
        figure_anomalies.append([ts.get_gt_anomalies_segmentwise().tolist()])
    return [Figure_content(None, length, anomaly) for anomaly in figure_anomalies]


######################
## Example problems ##
######################

def PA_problem():
    anomalies = [[[25, 39]], [12, 36]]
    metric_list = [metrics.Pointwise_metrics, metrics.PointAdjust]
    length = 48
    name = "pa_problem.txt"

    create_table(anomalies, metric_list, length, name, scale=2)


def late_early_prediction():

    #anomalies = [[[5, 9], [15, 19]], [8,9], [7,8], [6,7], [5,6], [8,9,18,19],[7,8,17,18],[6,7,16,17], [5,6,15,16]]
    anomalies = [[[5, 9], [15, 19]], [9], [7], [5], [9,19],[7,17], [5,15]]
    metric_list = [metrics.PointAdjust, metrics.NAB_score]
    length = 22

    create_table(anomalies, metric_list, length, scale=2)

def length_problem_1():

    anomalies = [[[5,6], [15, 19]], [15,16,17,18,19], [5,6,15,16]]
    metric_list = [metrics.Pointwise_metrics, metrics.PointAdjust, metrics.Segmentwise_metrics, metrics.Composite_f, metrics.NAB_score, metrics.Affiliation]
    length = 22

    create_table(anomalies, metric_list, length, scale=2)

def length_problem_2():

    anomalies = [[2,4,6,15,16,17,18,19], [2,3,4,5,6], [2,15,16,17,18,19]]
    metric_list = [metrics.Pointwise_metrics, metrics.PointAdjust, metrics.Segmentwise_metrics, metrics.Composite_f, metrics.Affiliation]#, metrics.NAB_score]
    length = 22

    create_table(anomalies, metric_list, length, scale=2)

def short_predictions():

    anomalies = [[ [14, 20]], [3,16], [2,3,4,5,6,7,15,16,17,18,19,20]]
    metric_list = [metrics.Pointwise_metrics, metrics.PointAdjust, metrics.Segmentwise_metrics, metrics.Composite_f, metrics.NAB_score, metrics.Affiliation]
    length = 22

    create_table(anomalies, metric_list, length, scale=2)


def detection_over_covering():
    anomalies = [[5,12], [20,27]], [[5,12]], [5,20], [[5,12],[20,27]]
    metric_list = [metrics.Pointwise_metrics, metrics.PointAdjust, metrics.Segmentwise_metrics, metrics.Composite_f, metrics.NAB_score, metrics.Affiliation]
    length = 28

    create_table(anomalies, metric_list, length, scale=2)

def close_fp():
    anomalies = [[12,13,14,15],[7,8],[8,9],[9,10],[10,11]]
    metric_list = [
            #metrics.Pointwise_metrics, metrics.PointAdjust, metrics.Segmentwise_metrics, metrics.Composite_f, metrics.NAB_score, 
            metrics.Affiliation]
    length = 17

    create_table(anomalies, metric_list, length, scale=2)

def concise():
    anomalies = [[4,5,7,8,10,12], [4,5,7,8,10], [[3,15]]]
    metric_list = [
            metrics.Pointwise_metrics, metrics.PointAdjust, metrics.Segmentwise_metrics, metrics.Composite_f, #metrics.NAB_score,
            metrics.Affiliation]
    length = 17

    create_table(anomalies, metric_list, length, scale=2)

if __name__ == "__main__":
    #PA_problem()
    #late_early_prediction()
    #length_problem_1()
    #length_problem_2()
    #short_predictions()
    #detection_over_covering()
    #close_fp()
    concise()
