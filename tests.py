from metrics import *

import unittest


class Detected_anomalies_tester(unittest.TestCase):
    def test_unsorted(self):
        self.assertRaises(AssertionError, Detected_anomalies, 10, [2, 3, 4], [3, 4, 2])
        self.assertRaises(AssertionError, Detected_anomalies, 10, [3, 4, 2], [2, 3, 4])
        self.assertRaises(AssertionError, Detected_anomalies, 10, [[1, 8]], [[5, 6], [1, 2]])
        self.assertRaises(AssertionError, Detected_anomalies, 10, [[5, 6], [1, 2]], [[1, 8]])

    def test_nonunique(self):
        self.assertRaises(AssertionError, Detected_anomalies, 10, [2, 4, 4], [2, 3, 4])
        self.assertRaises(AssertionError, Detected_anomalies, 10, [2, 3, 4], [2, 4, 4])

    def test_long_anom(self):
        self.assertRaises(AssertionError, Detected_anomalies, 4, [1], [2, 3, 4])
        self.assertRaises(AssertionError, Detected_anomalies, 4, [[2, 4]], [1])
        self.assertRaises(AssertionError, Detected_anomalies, 4, [-1], [1])

    def test_point_to_seq(self):
        anom1 = [3, 4, 5, 7, 8, 11]
        anom2 = [[3, 5], [7, 8], [11, 11]]
        d = Detected_anomalies(12, anom1, anom2)

        self.assertTrue(np.array_equal(np.array(anom1), d.get_predicted_anomalies_ptwise()))
        self.assertTrue(np.array_equal(np.array(anom2), d.get_gt_anomalies_segmentwise()))

    def test_anomaly_binary_form(self):
        anom1 = [3, 4, 5, 7, 8, 11]
        d = Detected_anomalies(12, anom1, anom1)

        self.assertTrue(np.array_equal(d.get_gt_anomalies_binary(), np.array([0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1])))
        self.assertTrue(
            np.array_equal(d.get_predicted_anomalies_binary(), np.array([0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1]))
        )

    def test_empty_anom(self):
        anom1 = [3, 4, 5, 7, 8]
        anom2 = []

        d = Detected_anomalies(12, anom1, anom2)
        self.assertEqual(0, len(d.get_predicted_anomalies_ptwise()))
        self.assertEqual(0, len(d.get_predicted_anomalies_segmentwise()))


class Confusion_metrics_tester(unittest.TestCase):
    def test_metrics(self):
        self.assertEqual(0.75, recall(tp=3, fn=1))
        self.assertEqual(0.75, precision(tp=3, fp=1))
        self.assertEqual(0.6, f1_score(tp=3, fn=1, fp=3))

    def test_requires_names(self):
        self.assertRaises(TypeError, recall, 3, 4)
        self.assertRaises(TypeError, precision, 3, 4)
        self.assertRaises(TypeError, f1_score, 3, 4, 5)

    def test_zerodivision(self):
        self.assertRaises(ZeroDivisionError, recall, tp=0, fn=0)
        self.assertRaises(ZeroDivisionError, precision, tp=0, fp=0)
        self.assertRaises(ZeroDivisionError, f1_score, tp=0, fp=1, fn=1)


class Metrics_tester(unittest.TestCase):
    def test_PW(self):
        pw = Pointwise_metrics(10, [1, 2, 3, 4], [4, 5, 6])

        self.assertEqual(pw.tp, 1)
        self.assertEqual(pw.fp, 2)
        self.assertEqual(pw.fn, 3)

    def test_PA(self):
        pa = PointAdjust(10, [1, 2, 3, 4, 9], [4, 5, 6])
        self.assertEqual(pa.tp, 4)
        self.assertEqual(pa.fp, 2)
        self.assertEqual(pa.fn, 1)

        pa = PointAdjust(10, [1, 2, 3, 4, 5, 6, 7], [4])
        self.assertEqual(pa.get_score(), 1)

    def test_Segment(self):
        # s = Segmentwise_metrics(10, [[1,2],[4,4],[7,9]], [[0,6]])
        # self.assertEqual(s.tp, 2)
        # self.assertEqual(s.fp, 0)
        # self.assertEqual(s.fn, 1)

        s = Segmentwise_metrics(10, [[1, 2], [4, 4], [7, 9]], [[6, 6], [8, 8]])
        self.assertEqual(s.tp, 1)
        self.assertEqual(s.fp, 1)
        self.assertEqual(s.fn, 2)

        s = Segmentwise_metrics(10, [[1, 2], [4, 4], [7, 9]], [])
        self.assertEqual(s.tp, 0)
        self.assertEqual(s.fp, 0)
        self.assertEqual(s.fn, 3)

        s = Segmentwise_metrics(10, [[1, 2], [4, 4], [7, 9]], [[0, 9]])
        self.assertEqual(s.tp, 3)
        self.assertEqual(s.fp, 0)
        self.assertEqual(s.fn, 0)

    def test_CF(self):
        c = Composite_f(10, [0, 2, 3, 5, 7, 9], [3, 6])
        f = c.get_score()
        self.assertEqual(c.p, 0.5)
        self.assertEqual(c.r, 0.2)

    def test_affiliation(self):
        a = Affiliation(10, [2, 3], [2])
        f = a.get_score()
        self.assertEqual(a.p, 1)
        self.assertTrue(a.r < 1)

        a = Affiliation(10, [2, 3], [2, 3, 4])
        f = a.get_score()
        self.assertTrue(a.p < 1)
        self.assertEqual(a.r, 1)

    def test_range_pr(self):
        r = Range_PR(10, [2, 3], [2])
        f = r.get_score()
        self.assertEqual(r.p, 1)
        self.assertTrue(r.r < 1)

        r2 = Range_PR(10, [2, 3], [2, 3])
        f2 = r2.get_score()
        self.assertTrue(f2 > f)

        r = Range_PR(10, [2, 3], [2, 3, 4])
        f = r.get_score()
        self.assertTrue(r.p < 1)
        self.assertEqual(r.r, 1)

    def test_NAB(self):
        n = NAB_score(10, [[3, 6]], [3])
        self.assertAlmostEqual(n.get_score(), 1)

        n = NAB_score(10, [[3, 6]], [])
        self.assertAlmostEqual(n.get_score(), 0)

        n = NAB_score(10, [[3, 6]], [1])
        self.assertAlmostEqual(n.get_score(), -1 * 0.11 / 2)


class Threshold_metric_tester(unittest.TestCase):
    def setUp(self):
        length = 4
        gt = [[2, 3]]
        base_prediction = [[2, 3]]
        self.args = (length, gt, base_prediction)

    #    def test_roc(self):
    #        a = aucroc(true = [0,0,1,1], score = [0.1,0.4,0.35,0.8])
    def test_auc_pr(self):
        auc_pr = AUC_PR_pw(*self.args)

        anomaly_score = [1, 3, 2, 4]
        score = auc_pr.get_score_given_anomaly_score(anomaly_score)
        self.assertAlmostEqual(score, 0.83, 2)

        anomaly_score = [1, 2, 3, 4]
        score = auc_pr.get_score_given_anomaly_score(anomaly_score)
        self.assertEqual(score, 1)

        anomaly_score = [4, 3, 1, 1]
        score = auc_pr.get_score_given_anomaly_score(anomaly_score)
        self.assertEqual(score, 0.5)

    def test_auc_roc(self):
        auc_roc = AUC_ROC(*self.args)

        anomaly_score = [1, 3, 2, 4]
        score = auc_roc.get_score_given_anomaly_score(anomaly_score)
        self.assertAlmostEqual(score, 0.75, 2)

        anomaly_score = [1, 2, 3, 4]
        score = auc_roc.get_score_given_anomaly_score(anomaly_score)
        self.assertEqual(score, 1)

        anomaly_score = [4, 4, 4, 4]
        score = auc_roc.get_score_given_anomaly_score(anomaly_score)
        self.assertEqual(score, 0.5)

    def test_PatK(self):
        patk = PatK_pw(*self.args)

        anomaly_score = [1, 4, 2, 3]
        score = patk.get_score_given_anomaly_score(anomaly_score)
        self.assertEqual(score, 0.5)

        anomaly_score = [1, 2, 3, 4]
        score = patk.get_score_given_anomaly_score(anomaly_score)
        self.assertEqual(score, 1)

        anomaly_score = [3, 4, 1, 2]
        score = patk.get_score_given_anomaly_score(anomaly_score)
        self.assertEqual(score, 0)

        patk = PatK_pw(4, [1, 2, 3], [3])
        anomaly_score = [3, 4, 1, 2]
        score = patk.get_score_given_anomaly_score(anomaly_score)
        self.assertAlmostEqual(score, 2 / 3)

        patk = PatK_pw(4, [], [3])
        self.assertRaises(AssertionError, patk.get_score)

    def test_Threshold_independent_method(self):
        patk = PatK_pw(*self.args)

        score = patk.get_score()
        self.assertEqual(score, 1)

    def test_best_threshold_pw(self):
        metric = Best_threshold_pw(*self.args)

        anomaly_score = [1, 3, 2, 4]
        score = metric.get_score_given_anomaly_score(anomaly_score)
        self.assertAlmostEqual(score, 2 * 2 / 3 * 1 / (1 + 2 / 3))

        anomaly_score = [2, 3, 1, 4]
        score = metric.get_score_given_anomaly_score(anomaly_score)
        self.assertAlmostEqual(score, 2 * 1 / 2 * 1 / (1 + 1 / 2))

        anomaly_score = [4, 3, 1, 2]
        score = metric.get_score_given_anomaly_score(anomaly_score)
        self.assertAlmostEqual(score, 2 * 1 / 2 * 1 / (1 + 1 / 2))


if __name__ == "__main__":
    unittest.main()
