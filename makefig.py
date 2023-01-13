from dataclasses import dataclass
from typing import Union


@dataclass
class Figure_content:

    row_titles: [str]  # | None

    time_series_length: int
    anomalies: []


class Figure:
    def __init__(self, figure_content, scale=3):
        self.string = ""
        self.filename = "figure1.txt"
        self.content = figure_content
        self.n_rows = len(self.content.anomalies)

        self.rows_added = 0
        self.current_height = 0

        self.scale = scale
        self.steplength = .2*self.scale
        self.title_placement = -.7*self.scale

        self.circle_radius = 0.05*self.scale
        self.point_step_length = .1*self.scale

    def make(self):
        self.add_line(
            f"\\begin{{tikzpicture}}[baseline=-\\the\\dimexpr\\fontdimen22\\textfont2\\relax]"
        )
        self.add_all_content()
        self.add_line("\\end{tikzpicture}")

    def add_line(self, line):
        self.string += line
        self.string += "\n"

    def write(self):
        self.make()
        with open(self.filename, "w") as file:
            file.write(self.string)

    def add_all_content(self):

        for i in range(self.n_rows):
            self.add_next_row()

    def add_next_row(self):

        self.add_row_title()
        self.add_row_line()
        self.add_row_points()
        self.add_row_anomalies()
        self.add_row_explainations()

        self.rows_added += 1
        self.current_height -= self.steplength

    def add_row_title(self):
        if self.content.row_titles != None:
            self.add_line(
                f"\\node (GT) at ({self.title_placement},{self.current_height}) {{{self.content.row_titles[self.rows_added]}}};"
            )

    def add_row_line(self):
        self.add_line(
            f"\\draw (0,{self.current_height}) -- ({round(self.content.time_series_length*self.point_step_length-self.point_step_length,3)},{self.current_height});"
        )

    def add_row_points(self):
        self.add_line(f"\\nomalies[first x=0, second x={round(self.point_step_length,3)}, last x={round(self.content.time_series_length*self.point_step_length-0.01,3)}, y={self.current_height}, radius={round(self.circle_radius,3)}]") # last one NOT included (-0.01)

    def add_row_anomalies(self):
        step=self.point_step_length
        for start, stop in self.content.anomalies[self.rows_added]:
            self.add_line(f"\\anomalies[first x={round(start*step,3)}, second x={round(start*step+step,3)}, last x={round(stop*step+0.01,3)}, y={self.current_height}, radius={round(self.circle_radius,3)}]") # last one included (+0.01

    def add_row_explainations(self):
        pass


if __name__ == "__main__":
    titles = ["Lables", "Prediction 1", "Prediction 2"]
    anomalies = [[[7, 14]], [[11, 12]], [[8, 9]]]
    content = Figure_content(titles, 19, anomalies)
    fig = Figure(content)
    fig.write()
