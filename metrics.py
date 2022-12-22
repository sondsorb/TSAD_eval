import numpy as np
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, average_precision_score

from nabscore import Sweeper
from affiliation.metrics import pr_from_events as affiliation_pr
from prts import ts_recall, ts_precision


def pointwise_to_segmentwise(pointwise):

    segmentwise = []

    prev = -10
    for point in pointwise:
        if point > prev + 1:
            segmentwise.append([point, point])
        else:
            segmentwise[-1][-1] += 1
        prev = point
    return np.array(segmentwise)


def segmentwise_to_pointwise(segmentwise):

    pointwise = []

    for start, end in segmentwise:
        for point in range(start, end + 1):
            pointwise.append(point)

    return np.array(pointwise)

def pointwise_to_binary(pointwise, length):
    anomalies_binary = np.zeros(length)
    if len(pointwise) > 0:
        anomalies_binary[pointwise] = 1
    return anomalies_binary

class Binary_anomalies:
    def __init__(self, length, anomalies):
        self._length = length
        self._set_anomalies(anomalies)

    def _set_anomalies(self, anomalies):
        anomalies = np.array(anomalies)
        if len(anomalies.shape) == 1:
            anomalies_ptwise = anomalies
            anomalies_segmentwise = pointwise_to_segmentwise(anomalies)
        elif len(anomalies.shape) == 2:
            anomalies_segmentwise = anomalies
            anomalies_ptwise = segmentwise_to_pointwise(anomalies)
        else:
            raise ValueError(f"Illegal shape of anomalies:\n{anomalies}")

        if len(anomalies_ptwise) > 0:
            assert anomalies_ptwise[-1] < self._length
        anomalies_binary=pointwise_to_binary(anomalies_ptwise, self._length)

        if len(anomalies_ptwise) > 0:
            assert all(anomalies_ptwise == np.sort(anomalies_ptwise))
            assert anomalies_ptwise[0] >= 0
            assert len(anomalies_ptwise) == len(np.unique(anomalies_ptwise))
            assert len(anomalies_ptwise) == sum(anomalies_binary)

            assert all(anomalies_segmentwise[:, 0] == np.sort(anomalies_segmentwise[:, 0]))
            assert all(anomalies_segmentwise[:, 1] >= anomalies_segmentwise[:, 0])

        self.anomalies_segmentwise=anomalies_segmentwise
        self.anomalies_ptwise=anomalies_ptwise
        self.anomalies_binary=anomalies_binary

class Binary_detection:
    def __init__(self, length, gt_anomalies, predicted_anomalies):
        self._length = length
        self._gt = Binary_anomalies(length,gt_anomalies)
        self._prediction = Binary_anomalies(length,predicted_anomalies)

    def get_length(self):
        return self._length

    def get_gt_anomalies_ptwise(self):
        return self._gt.anomalies_ptwise

    def get_gt_anomalies_segmentwise(self):
        return self._gt.anomalies_segmentwise

    def get_predicted_anomalies_ptwise(self):
        return self._prediction.anomalies_ptwise

    def get_predicted_anomalies_segmentwise(self):
        return self._prediction.anomalies_segmentwise

    def get_predicted_anomalies_binary(self):
        return self._prediction.anomalies_binary

    def get_gt_anomalies_binary(self):
        return self._gt.anomalies_binary


def f1_from_pr(p, r):
    return (2 * r * p) / (r + p)


def f1_score(*args, tp, fp, fn):
    r = recall(tp=tp, fn=fn)
    p = precision(tp=tp, fp=fp)

    # if r == 0 and p == 0:
    #    return np.nan
    return f1_from_pr(p, r)


def recall(*args, tp, fn):
    return tp / (tp + fn)


def precision(*args, tp, fp):
    return tp / (tp + fp)


class original_PR_metric(Binary_detection):
    # def __init__(self, *args):
    #    super().__init__(*args)
    def get_score(self):
        return f1_score(tp=self.tp, fn=self.fn, fp=self.fp)


class Pointwise_metrics(original_PR_metric):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = "PW"
        self.set_confusion()

    def set_confusion(self):
        gt = np.zeros(self.get_length())
        if len(self.get_gt_anomalies_ptwise()) > 0:
            gt[self.get_gt_anomalies_ptwise()] = 1

        pred = np.zeros(self.get_length())
        if len(self.get_predicted_anomalies_ptwise()) > 0:
            pred[self.get_predicted_anomalies_ptwise()] = 1

        self.tp = np.sum(pred * gt)
        self.fp = np.sum(pred * (1 - gt))
        self.fn = np.sum((1 - pred) * gt)


class PointAdjust(Pointwise_metrics):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = "PA"
        self.adjust()
        self.set_confusion()

    def adjust(self):

        adjusted_prediction = self.get_predicted_anomalies_ptwise().tolist()
        for start, end in self.get_gt_anomalies_segmentwise():
            for i in range(start, end + 1):
                if i in adjusted_prediction:
                    for j in range(start, end + 1):
                        adjusted_prediction.append(j)
                    break

        self._prediction._set_anomalies(np.sort(np.unique(adjusted_prediction)))


class Segmentwise_metrics(original_PR_metric):
    def __init__(self, *args):
        self.name = "SF"
        super().__init__(*args)
        self.set_confusion()

    def set_confusion(self):

        tp = 0
        fn = 0
        for gt_anomaly in self.get_gt_anomalies_segmentwise():
            found = False
            for predicted_anomaly in self.get_predicted_anomalies_segmentwise():
                if self._overlap(gt_anomaly, predicted_anomaly):
                    tp += 1
                    found = True
                    break
            if found == False:
                fn += 1
        fp = 0
        for predicted_anomaly in self.get_predicted_anomalies_segmentwise():
            found = False
            for gt_anomaly in self.get_gt_anomalies_segmentwise():
                if self._overlap(gt_anomaly, predicted_anomaly):
                    found = True
                    break
            if found == False:
                fp += 1
        self.fp = fp
        self.fn = fn
        self.tp = tp

    def _overlap(self, anomaly1, anomaly2):
        return not (anomaly1[1] < anomaly2[0] or anomaly2[1] < anomaly1[0])


class Redefined_PR_metric(Binary_detection):
    def __init__(self, *args):
        super().__init__(*args)

    def get_score(self):
        self.r = self.recall()
        self.p = self.precision()
        return f1_from_pr(self.p, self.r)

    def recall(self):
        raise NotImplementedError

    def precision(self):
        raise NotImplementedError


class Composite_f(Redefined_PR_metric):
    def __init__(self, *args):
        self.name = "CF"
        super().__init__(*args)

        self.pointwise_metrics = Pointwise_metrics(*args)
        self.segmentwise_metrics = Segmentwise_metrics(*args)

    def recall(self):
        return recall(tp=self.segmentwise_metrics.tp, fn=self.segmentwise_metrics.fn)

    def precision(self):
        return precision(tp=self.pointwise_metrics.tp, fp=self.pointwise_metrics.fp)


class Affiliation(Redefined_PR_metric):
    def __init__(self, *args):
        self.name = "AF"
        super().__init__(*args)

    def get_score(self):
        pr_output = affiliation_pr(
            self._reformat_segments(self.get_predicted_anomalies_segmentwise()),
            self._reformat_segments(self.get_gt_anomalies_segmentwise()),
            (0, self.get_length()),
        )
        self.r = pr_output["recall"]
        self.p = pr_output["precision"]
        return f1_from_pr(self.p, self.r)

    def _reformat_segments(self, segments):
        segments = self._include_end_of_segments(segments)
        segments = self._tuplify_segments(segments)
        return segments

    def _include_end_of_segments(self, segments):
        return [[start, end + 1] for start, end in segments]

    def _tuplify_segments(self, segments):
        return [tuple(segment) for segment in segments]


class Range_PR(Redefined_PR_metric):
    def __init__(self, *args, alpha=0.2, bias="flat"):
        super().__init__(*args)
        self.alpha = alpha
        self.bias = bias
        self.set_name()

    def set_name(self):
        self.name = f"RF$_{{{self.bias}}}^{{\\alpha={self.alpha}}}$"

    def set_kwargs(self):
        real = np.zeros(self.get_length())
        real[self.get_gt_anomalies_ptwise()] = 1
        pred = np.zeros(self.get_length())
        pred[self.get_predicted_anomalies_ptwise()] = 1

        self.kwargs = {"real": real, "pred": pred, "alpha": self.alpha, "cardinality": "one", "bias": self.bias}

    def recall(self):
        self.set_kwargs()
        return ts_recall(**self.kwargs)

    def precision(self):
        self.set_kwargs()
        return ts_precision(**self.kwargs)


class TS_aware(Redefined_PR_metric):
    pass


class Enhanced_TS_aware(Redefined_PR_metric):
    pass


class NAB_score(Binary_detection):
    def __init__(self, *args):
        self.name = "NAB/100"
        super().__init__(*args)

        self.sweeper = Sweeper(probationPercent=0, costMatrix={"tpWeight": 1, "fpWeight": 0.11, "fnWeight": 1})

    def get_score(self):
        if len(self.get_predicted_anomalies_ptwise()) == 0:
            return 0  # raw_score == null_score
        if len(self.get_gt_anomalies_ptwise()) == 0:
            return np.nan  # perfect_score == null_score

        scoresByThreshold = self.get_scoresByThreshold(self.get_predicted_anomalies_ptwise())

        null_score = scoresByThreshold[0].score
        raw_score = scoresByThreshold[1].score

        scoresByThreshold = self.get_scoresByThreshold(self.get_gt_anomalies_ptwise())
        assert scoresByThreshold[1].total == scoresByThreshold[1].tp + scoresByThreshold[1].tn
        perfect_score = scoresByThreshold[1].score

        return (raw_score - null_score) / (perfect_score - null_score)

    def get_scoresByThreshold(self, prediction):
        anomaly_scores = pointwise_to_binary(prediction, self.get_length())
        timestamps = np.arange(self.get_length())
        windowLimits = self.get_gt_anomalies_segmentwise()
        dataSetName = "dummyname"
        anomalyList = self.sweeper.calcSweepScore(timestamps, anomaly_scores, windowLimits, dataSetName)
        scoresByThreshold = self.sweeper.calcScoreByThreshold(anomalyList)

        assert scoresByThreshold[0].threshold == 1.1
        assert scoresByThreshold[1].threshold == 1.0

        return scoresByThreshold



class Nonbinary_detection:
    def __init__(self, gt_anomalies, anomaly_score):
        self._length = len(anomaly_score)
        self._gt = Binary_anomalies(self._length,gt_anomalies)
        self._anomaly_score = anomaly_score

    def get_gt_anomalies_ptwise(self):
        return self._gt.anomalies_ptwise

    def get_gt_anomalies_segmentwise(self):
        return self._gt.anomalies_segmentwise

    def get_gt_anomalies_binary(self):
        return self._gt.anomalies_binary

    def get_anomaly_score(self):
        return self._anomaly_score


class Best_threshold_pw(Nonbinary_detection):
    def __init__(self, *args):
        self.name = "pw-best"
        super().__init__(*args)

    def get_score(self):
        scores = []
        for current_anomaly_score in self.get_anomaly_score():
            scores.append(
                self.get_score_given_anomaly_score_and_threshold(threshold=current_anomaly_score)
            )
        return np.nanmax(scores)

    def get_score_given_anomaly_score_and_threshold(self, threshold):
        gt = self.get_gt_anomalies_binary()

        pred = np.array(self.get_anomaly_score()) >= threshold

        return f1_score(tp=pred @ gt, fn=(1 - pred) @ gt, fp=(1 - gt) @ pred)


class AUC_ROC(Nonbinary_detection):
    def __init__(self, *args):
        self.name = "AUC-ROC"
        super().__init__(*args)

    def get_score(self):
        gt = self.get_gt_anomalies_binary()
        return roc_auc_score(gt, self.get_anomaly_score())


class AUC_PR_pw(Nonbinary_detection):
    def __init__(self, *args):
        self.name = "AUC-PR"
        super().__init__(*args)

    def get_score(self):
        gt = self.get_gt_anomalies_binary()
        return average_precision_score(gt, self.get_anomaly_score())


class PatK_pw(Nonbinary_detection):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = f"P@{len(self.get_gt_anomalies_ptwise())}"

    def get_score(self):
        gt = self.get_gt_anomalies_binary()

        k = int(sum(gt))
        threshold = np.sort(self.get_anomaly_score())[-k]

        pred = self.get_anomaly_score() >= threshold
        assert sum(pred) == k, (k, pred)

        return pred @ gt / k
