"""Focused regression tests for the shared JIGSAWS layers.

These tests only require numpy plus the local dataset, so they run on a bare
checkout. Method-specific tests that need scikit-learn/hmmlearn live in
``tests/test_methods.py`` once those extras are installed.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from jigsaws.config import Settings, Task
from jigsaws.dataset import Dataset
from jigsaws.evaluation import BinaryMetrics
from jigsaws.kinematics import Instrument, compute_features
from jigsaws.methods.kinematic.method import collect_feature_table, select_task_features

REPO_ROOT = Path(__file__).resolve().parents[1]
HAS_DATA = (REPO_ROOT / "kinematic_analysis/trajectory_plot/da_vici_data_with_iDT_features").is_dir()


class TaskGradingTest(unittest.TestCase):
    def test_binary_thresholds(self) -> None:
        self.assertEqual(Task.SUTURING.grade(19), 0)
        self.assertEqual(Task.SUTURING.grade(20), 2)
        self.assertEqual(Task.KNOT_TYING.grade(15), 0)
        self.assertEqual(Task.NEEDLE_PASSING.grade(15), 0)
        self.assertEqual(Task.NEEDLE_PASSING.grade(16), 2)

    def test_three_tiers(self) -> None:
        self.assertEqual(Task.SUTURING.tier(15), 0)
        self.assertEqual(Task.SUTURING.tier(16), 1)
        self.assertEqual(Task.SUTURING.tier(19), 2)
        self.assertEqual(Task.NEEDLE_PASSING.tier(16), 1)
        self.assertEqual(Task.NEEDLE_PASSING.tier(20), 2)


class MetricsTest(unittest.TestCase):
    def test_confusion_counts(self) -> None:
        metrics = BinaryMetrics.compute(
            np.array([0, 0, 2, 2]), np.array([0, 2, 2, 2])
        )
        self.assertAlmostEqual(metrics.accuracy, 0.75)
        self.assertAlmostEqual(metrics.novice_accuracy, 0.5)
        self.assertAlmostEqual(metrics.expert_accuracy, 1.0)
        self.assertEqual(metrics.true_novice, 1)
        self.assertEqual(metrics.true_expert, 2)

    def test_rejects_unknown_labels(self) -> None:
        with self.assertRaises(ValueError):
            BinaryMetrics.compute(np.array([0, 1]), np.array([0, 2]))


@unittest.skipUnless(HAS_DATA, "JIGSAWS data directory is not present")
class DatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = Settings.resolve(repo_root=REPO_ROOT)
        cls.dataset = Dataset(cls.settings, Task.SUTURING)

    def test_suturing_metadata(self) -> None:
        self.assertEqual(len(self.dataset), 39)
        self.assertEqual(sum(1 for r in self.dataset.records if r.grade == 0), 23)
        self.assertEqual(sum(1 for r in self.dataset.records if r.grade == 2), 16)
        self.assertEqual(
            self.dataset.surgeme_labels,
            ("G1", "G2", "G3", "G4", "G5", "G6", "G8", "G9", "G10", "G11"),
        )

    def test_skill_detection_folds(self) -> None:
        folds = self.dataset.folds(kind="SkillDetection")
        self.assertEqual(len(folds), 5)
        for fold in folds:
            self.assertGreaterEqual(len(fold.test), 1)
            self.assertTrue(set(fold.train).isdisjoint(fold.test))
            self.assertEqual(len(fold.train) + len(fold.test), 39)

    def test_feature_shape(self) -> None:
        table = collect_feature_table(self.dataset)
        self.assertEqual(table.features.shape, (39, 4, 10))
        self.assertEqual(
            select_task_features(table.features, Task.SUTURING.value).shape,
            (39, 28),
        )


@unittest.skipUnless(HAS_DATA, "JIGSAWS data directory is not present")
class KinematicsParityTest(unittest.TestCase):
    def test_suturing_b001_mtf_left(self) -> None:
        path = (
            REPO_ROOT
            / "kinematic_analysis/trajectory_plot/da_vici_data_with_iDT_features"
            / "Suturing/kinematics/AllGestures/Suturing_B001.txt"
        )
        matrix = np.loadtxt(path)
        features = compute_features(matrix, Instrument.MTF_L).as_tuple()
        expected = (
            187.7,
            3.5208273582,
            0.0187577377,
            0.0006460117,
            10326.1259871029,
            1091137191.755884,
            3.7888786793,
            22.8288848277,
            0.4209249185,
            0.2723337223,
        )
        np.testing.assert_allclose(features, expected, rtol=1e-6, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
