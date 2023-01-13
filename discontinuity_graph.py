from maketable import *


###########################
### Discontinuity graph ###
###########################
class Discontinuity_table(Table):
    def __init__(self, metric_names, results, marks=[]):
        self.metric_names = metric_names
        self.results = results
        self.marks = marks
        super().__init__(Table_content([],[],[]), scale=2)
        self.x_factor = 1 / 20
        self.y_factor = 1 / 2 / self.scale

        self.row_length = 2
        self.n_rows = len(metric_names)

    def add_top_row(self):
        self.string += ("Metric")
        self.string += "&"
        self.string += ("Score")
        self.end_row()

    def add_next_row(self):
        self.string +=(self.metric_names[self.rows_added])
        self.string += "&"
        self.add_graph(self.rows_added + 1)
        self.end_row()
        self.rows_added += 1

    def add_graph(self, number):
        self.add_line(
            f"\\begin{{tikzpicture}}[scale={self.scale}, baseline=-\\the\\dimexpr\\fontdimen22\\textfont2\\relax]"
        )
        for x in self.marks:
            self.add_line(f"\draw[-, gray] ({x*self.x_factor},0) -- ({x*self.x_factor},{0.2*self.y_factor});")
        self.add_line(f"\draw[-, gray] (0,0) -- ({self.x_factor*(len(self.results[self.metric_names[number-1]])-1)},0);")
        self.add_line("\\foreach \\i/\\a in")
        self.add_line(
            str([(i * self.x_factor, a * self.y_factor) for i, a in enumerate(self.results[self.metric_names[number - 1]])])
            .replace(",", "/")
            .replace(")/", ",")
            .replace("(", "")
            .replace("[", "{")
            .replace("]", "}")
            .replace(")}", "}{")
        )
        self.add_line("\\coordinate (now) at (\\i,\\a) {};")
        self.add_line("  \\ifthenelse{\\equal{\\i}{0.0}}{}{")
        self.add_line("  \\draw[-, teal] (prev) -- (now);")
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
    marks=[35,40,55,60]
    metric_names=[]
    for metric in [*All_metrics, Range_PR_front]:
        metric_names.append(metric(5, [3,4], [3]).name)
        current_result = np.zeros(ts_length-pred_length)
        for pred_mid in range(ts_length-pred_length):
            gt = [[gt_start, gt_start+gt_length-1]]
            pred=[[max(0,pred_mid-pred_length//2), pred_mid+pred_length//2]]
            current_result[pred_mid] = metric(ts_length, gt, pred).get_score()
        current_result = (current_result-min(current_result))/(max(current_result)-min(current_result))
        result[metric_names[-1]] = current_result

    table = Discontinuity_table(metric_names,result, marks)
    table.write()
    print(table)



discontinuity_graphs()
