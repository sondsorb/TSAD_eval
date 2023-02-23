from dataclasses import dataclass
from typing import Union


class Binary_anomalies_figure:
    def __init__(self, binary_anomalies, scale=3):
        self.string = ""
        self.binary_anomalies = binary_anomalies

        self.scale = scale
        self.steplength = 0.2 * self.scale

        self.circle_radius = 0.05 * self.scale
        self.point_step_length = 0.1 * self.scale

    def make(self):
        self.add_line(f"\\begin{{tikzpicture}}[baseline=-\\the\\dimexpr\\fontdimen22\\textfont2\\relax]")
        self.add_content()
        self.add_line("\\end{tikzpicture}")

    def add_line(self, line):
        self.string += line
        self.string += "\n"

    def add_content(self):
        self.add_normal_points()
        self.add_anomalies()

    def add_normal_points(self):
        self.add_line(
            f"\\nomalies[first x=0, second x={round(self.point_step_length,3)}, last x={round(self.binary_anomalies.get_length()*self.point_step_length-0.01,3)}, y=0, radius={round(self.circle_radius,3)}]"
        )  # last one NOT included (-0.01)

    def add_anomalies(self):
        step = self.point_step_length
        for start, stop in self.binary_anomalies.anomalies_segmentwise:
            if stop > start:
                self.add_line(
                    f"\\anomalies[first x={round(start*step,3)}, second x={round(start*step+step,3)}, last x={round(stop*step+0.01,3)}, y=0, radius={round(self.circle_radius,3)}]"
                )  # last one included (+0.01
            else:
                self.add_line(f"\\anomaly{{{round(start*step,3)}}}{{0}}{{{round(self.circle_radius, 3)}}}")
