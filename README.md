# TSAD_eval

This repository contains experiments for several evaluation metrics for time series anomaly detection, for the paper "Navigating the Metric Maze: A Taxonomy of Evaluation Metrics for Anomaly Detection in Time Series", available at https://arxiv.org/abs/2303.01272


### File overview

`metrics.py` contains all the metrics

`tests.py` are tests for the metrics in metrics.py



`auc_roc_pr_plot.py` and `threshold_plt.py` are for pyplot figures in the paper

`makefig.py`, `maketable.py` and `discontinuity_graph.py` are for tikz code for tables in the paper

the rest is the original source code for the more complicated metrics (imported by `metrics.py`)
