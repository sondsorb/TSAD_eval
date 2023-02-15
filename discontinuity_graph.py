from maketable import *


###########################
### Discontinuity graph ###
###########################
class Discontinuity_table(Table):
    def __init__(self, metric_names, results, marks=[]):
        self.metric_names = metric_names
        self.results = results
        self.marks = marks
        super().__init__(Table_content([], [], []), scale=2)
        self.x_factor = 1 / 20 * self.scale
        self.y_factor = 1 / 2
        self.y_shift = -0.2

        self.row_length = 2
        self.n_rows = len(metric_names)

    def add_top_row(self):
        self.string += "Metric"
        self.string += "&"
        self.string += "Score"
        self.end_row()

    def add_next_row(self):
        self.string += self.metric_names[self.rows_added]
        self.string += "&"
        self.add_graph(self.rows_added + 1)
        self.end_row()
        self.rows_added += 1

    def add_graph(self, number):
        self.add_line(f"\\begin{{tikzpicture}}[baseline=-\\the\\dimexpr\\fontdimen22\\textfont2\\relax]")
        for x in self.marks:
            self.add_line(
                f"\draw[-, gray] ({x*self.x_factor},{self.y_shift}) -- ({x*self.x_factor},{0.2*self.y_factor + self.y_shift});"
            )
        self.add_line(
            f"\draw[-, gray] (0,{self.y_shift}) -- ({self.x_factor*(len(self.results[self.metric_names[number-1]])-1)},{self.y_shift});"
        )
        self.add_line("\\foreach \\i/\\a in")
        self.add_line(
            str(
                [
                    (round(i * self.x_factor, 3), round(a * self.y_factor + self.y_shift, 3))
                    for i, a in enumerate(self.results[self.metric_names[number - 1]])
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
        self.add_line("  \\draw[-, teal, thick] (prev) -- (now);")
        self.add_line("  }")
        self.add_line("  \\coordinate (prev) at (\\i,\\a) {};")
        self.add_line("}")
        self.add_line("\\end{tikzpicture}")


def discontinuity_graphs():
    result = {}
    ts_length = 100
    pred_length = 5
    gt_length = 20
    gt_start = 40
    marks = [35, 40, 55, 60]
    assert pred_length % 2 == 1
    metric_names = []
    All_metrics.remove(metrics.Range_PR)
    all_metrics_and_rffront = [*All_metrics, metrics.Range_PR, Range_PR_front]
    for metric in all_metrics_and_rffront:
        if metric == metrics.TaF:
            metric_names.append(metric(5, [3, 4], [3], delta=10).name)
        elif metric == metrics.Time_Tolerant:
            metric_names.append(metric(5, [3, 4], [3], d=10).name)
        else:
            metric_names.append(metric(5, [3, 4], [3]).name)
        current_result = []
        for pred_mid in range(pred_length // 2, ts_length - pred_length // 2):
            gt = [[gt_start, gt_start + gt_length - 1]]
            pred = [[pred_mid - pred_length // 2, pred_mid + pred_length // 2]]
            if metric == metrics.TaF:
                current_result.append(metric(ts_length, gt, pred, delta=10).get_score())
            elif metric == metrics.Time_Tolerant:
                current_result.append(metric(ts_length, gt, pred, d=10).get_score())
            else:
                current_result.append(metric(ts_length, gt, pred).get_score())
        current_result = np.array(current_result)
        current_result = (current_result - min(current_result)) / (max(current_result) - min(current_result))
        result[metric_names[-1]] = current_result

    table = Discontinuity_table(metric_names, result, marks)
    table.write()
    print(table)


discontinuity_graphs()
