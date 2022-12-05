import numpy as np
from dataclasses import dataclass

from nabscore import Sweeper
from affiliation.metrics import pr_from_events as affiliation_pr



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


class Detected_anomalies:
    def __init__(self, length, gt_anomalies, predicted_anomalies):
        self._length = length
        self._set_gt_anomalies(gt_anomalies)
        self._set_predicted_anomalies(predicted_anomalies)

    def _set_gt_anomalies(self, anomalies):
        self._gt_anomalies_segmentwise, self._gt_anomalies_ptwise = self._set_anomalies(anomalies)

    def _set_predicted_anomalies(self, anomalies):
        self._predicted_anomalies_segmentwise, self._predicted_anomalies_ptwise = self._set_anomalies(anomalies)

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
            assert all(anomalies_ptwise == np.sort(anomalies_ptwise))
            assert anomalies_ptwise[0] >= 0
            assert anomalies_ptwise[-1] < self._length
            assert len(anomalies_ptwise) == len(np.unique(anomalies_ptwise))

            assert all(anomalies_segmentwise[:, 0] == np.sort(anomalies_segmentwise[:, 0]))
            assert all(anomalies_segmentwise[:, 1] >= anomalies_segmentwise[:, 0])

        return anomalies_segmentwise, anomalies_ptwise

    def get_gt_anomalies_ptwise(self):
        return self._gt_anomalies_ptwise

    def get_gt_anomalies_segmentwise(self):
        return self._gt_anomalies_segmentwise

    def get_predicted_anomalies_ptwise(self):
        return self._predicted_anomalies_ptwise

    def get_predicted_anomalies_segmentwise(self):
        return self._predicted_anomalies_segmentwise


def f1_from_pr(p,r):
    return (2 * r * p) / (r + p)

def f1_score(*args, tp, fp, fn):
    r = recall(tp=tp, fn=fn)
    p = precision(tp=tp, fp=fp)

    #if r == 0 and p == 0:
    #    return np.nan
    return f1_from_pr(p,r)


def recall(*args, tp, fn):
    return tp / (tp + fn)


def precision(*args, tp, fp):
    return tp / (tp + fp)


class original_PR_metric(Detected_anomalies):
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
        gt = np.zeros(self._length)
        if len(self.get_gt_anomalies_ptwise()) > 0:
            gt[self.get_gt_anomalies_ptwise()] = 1

        pred = np.zeros(self._length)
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

        self._set_predicted_anomalies(np.sort(np.unique(adjusted_prediction)))


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


class Redefined_PR_metric(Detected_anomalies):
    def __init__(self, *args):
        super().__init__(*args)

    def get_score(self):
        self.r = self.recall()
        self.p = self.precision()
        return f1_from_pr(self.p,self.r)

    def recall(self):
        raise NotImplementedError

    def precision(self):
        raise NotImplementedError


class Composite_f(Redefined_PR_metric):
    def __init__(self, *args):
        self.name="CF"
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
        pr_output = affiliation_pr(self._reformat_segments(self.get_predicted_anomalies_segmentwise()), self._reformat_segments(self.get_gt_anomalies_segmentwise()), (0, self._length))
        self.r = pr_output["recall"]
        self.p = pr_output["precision"]
        return f1_from_pr(self.p,self.r)

    def _reformat_segments(self,segments):
        segments = self._include_end_of_segments(segments)
        segments = self._tuplify_segments(segments)
        return segments

    def _include_end_of_segments(self, segments):
        return [[start,end+1] for start,end in segments]

    def _tuplify_segments(self,segments):
        return [tuple(segment) for segment in segments]


class Range_PR(Redefined_PR_metric):
    pass


class TS_aware(Redefined_PR_metric):
    pass


class Enhanced_TS_aware(Redefined_PR_metric):
    pass


class NAB_score(Detected_anomalies):
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
        anomaly_scores = np.zeros(self._length)
        anomaly_scores[prediction] = 1
        timestamps = np.arange(self._length)
        windowLimits = self.get_gt_anomalies_segmentwise()
        dataSetName = "dummyname"
        anomalyList = self.sweeper.calcSweepScore(timestamps, anomaly_scores, windowLimits, dataSetName)
        scoresByThreshold = self.sweeper.calcScoreByThreshold(anomalyList)

        assert scoresByThreshold[0].threshold == 1.1
        assert scoresByThreshold[1].threshold == 1.0

        return scoresByThreshold
