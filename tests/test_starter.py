"""CPU checks for the completed starter pipeline; no dataset or GPU is required."""
import ast
import json
import py_compile
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
STARTER = ROOT / "starter"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STARTER))

import benchmark  # noqa: E402
import inference  # noqa: E402
import losses  # noqa: E402
import model as model_lib  # noqa: E402
import train  # noqa: E402
import eval as ev  # noqa: E402

K = ev.NUM_CLASSES


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class TestPredictionContract(unittest.TestCase):
    def test_saved_predictions_are_accepted_by_eval(self):
        rng = np.random.default_rng(0)
        n = 200
        y = rng.integers(0, K, n)
        probs = softmax(rng.normal(size=(n, K)))
        names = [f"img{i}.jpg" for i in range(n)]
        with tempfile.TemporaryDirectory() as directory:
            path = ev.save_predictions(Path(directory) / "sub" / "F01_seed0_test.csv", names, y, probs)
            pred = ev.read_pred(str(path))
            self.assertEqual(pred.seed, 0)
            np.testing.assert_array_equal(pred.y_pred, probs.argmax(1))
            np.testing.assert_allclose(pred.probs, probs, atol=1e-6)

    def test_save_predictions_rejects_logits_and_wrong_shape(self):
        with self.assertRaisesRegex(ValueError, "chuẩn hoá"):
            ev.save_predictions("unused.csv", [f"{i}.jpg" for i in range(10)], np.zeros(10, int),
                                np.random.default_rng(1).normal(size=(10, K)))
        with self.assertRaisesRegex(ValueError, "dạng"):
            ev.save_predictions("unused.csv", ["a.jpg"], [0], np.ones((1, 5)) / 5)


class TestTrainingHelpers(unittest.TestCase):
    def test_paths_and_defaults(self):
        cfg = train.Config(exp_id="F01", seed=2, pred_dir="predictions", out_dir="runs")
        self.assertEqual(train.pred_path(cfg, "test"), Path("predictions/F01_seed2_test.csv"))
        self.assertEqual(train.run_dir(cfg), Path("runs/F01/seed2"))
        self.assertFalse(train.Config().save_test_predictions)
        self.assertEqual((train.Config().epochs, train.Config().batch_size,
                          train.Config().lr_backbone, train.Config().lr_head,
                          train.Config().weight_decay), (12, 64, 1e-4, 1e-3, 0.05))

    def test_parse_overrides(self):
        parsed = train.parse_overrides(["seed=2", "loss=focal", "ema_decay=none", "amp=false",
                                        "lr_head=0.002", "epochs=8"])
        self.assertEqual(parsed, {"seed": 2, "loss": "focal", "ema_decay": None,
                                  "amp": False, "lr_head": 0.002, "epochs": 8})
        with self.assertRaisesRegex(ValueError, "Unknown Config field"):
            train.parse_overrides(["typo=1"])

    def test_class_weighted_optimizer_parameter_groups(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.AdaptiveAvgPool2d(1))
                self.classifier = nn.Linear(4, K)

            def get_classifier(self):
                return self.classifier

        toy = Toy()
        model_lib.freeze_backbone(toy)
        groups = model_lib.param_groups(toy, 1e-4, 1e-3, 0.05)
        self.assertEqual(len(groups), 1)
        self.assertTrue(all(parameter.requires_grad for parameter in groups[0]["params"]))
        self.assertAlmostEqual(groups[0]["lr"], 1e-3)
        model_lib.set_frozen_backbone_eval(toy)
        self.assertFalse(toy.features[1].training)


class TestLosses(unittest.TestCase):
    def test_focal_gamma_zero_is_cross_entropy(self):
        torch.manual_seed(5)
        logits = torch.randn(32, K)
        target = torch.randint(K, (32,))
        focal = losses.FocalLoss(gamma=0)(logits, target)
        ce = nn.functional.cross_entropy(logits, target)
        self.assertLess(float((focal - ce).abs()), 1e-6)

    def test_label_smoothing_zero_is_cross_entropy(self):
        logits = torch.randn(16, K)
        target = torch.randint(K, (16,))
        actual = losses.LabelSmoothingCE(0.0)(logits, target)
        expected = nn.functional.cross_entropy(logits, target)
        self.assertLess(float((actual - expected).abs()), 1e-6)

    def test_class_weights_are_normalized_and_validate_counts(self):
        weights = losses.class_weights([10, 20, 30, 40, 50, 60, 70, 80, 90])
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertGreater(float(weights[0]), float(weights[-1]))
        self.assertEqual(len(losses.class_weights([10] * K, beta=0.99)), K)
        with self.assertRaises(ValueError):
            losses.class_weights([0] * K)

    def test_mixup_and_cutmix_contract(self):
        torch.manual_seed(2)
        x = torch.rand(8, 3, 32, 32)
        y = torch.arange(8) % K
        for mode in ("mixup", "cutmix"):
            mixed, (y_a, y_b, lam) = losses.mix_batch(x, y, alpha=1.0, mode=mode)
            self.assertEqual(mixed.shape, x.shape)
            self.assertEqual(y_a.shape, y.shape)
            self.assertEqual(y_b.shape, y.shape)
            self.assertGreaterEqual(lam, 0.0)
            self.assertLessEqual(lam, 1.0)
            value = losses.mixed_loss(nn.CrossEntropyLoss(), torch.randn(8, K), (y_a, y_b, lam))
            self.assertTrue(torch.isfinite(value))


class TestInferenceAndLatency(unittest.TestCase):
    def test_view_aggregation_and_temperature(self):
        rng = np.random.default_rng(4)
        first = rng.normal(size=(20, K))
        second = rng.normal(size=(20, K))
        probs = inference.aggregate_views([first, second], "prob")
        logits = inference.aggregate_views([first, second], "logit")
        np.testing.assert_allclose(probs.sum(1), 1.0, atol=1e-7)
        np.testing.assert_allclose(logits.sum(1), 1.0, atol=1e-7)
        self.assertFalse(np.allclose(probs, logits))
        labels = first.argmax(1)
        temperature = inference.fit_temperature(first, labels)
        self.assertGreater(temperature, 0.0)
        calibrated = inference.apply_temperature(first, temperature)
        np.testing.assert_allclose(calibrated.sum(1), 1.0, atol=1e-7)

    def test_bn_fusion_preserves_eval_output(self):
        torch.manual_seed(3)
        network = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1, bias=False), nn.BatchNorm2d(4),
                                nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, K)).eval()
        inputs = torch.randn(2, 3, 32, 32)
        expected = network(inputs)
        fused = inference.fuse_conv_bn(network)
        torch.testing.assert_close(fused(inputs), expected, rtol=1e-5, atol=1e-5)

    def test_benchmark_reports_percentiles_after_minimum_samples(self):
        calls = []
        result = benchmark.bench(lambda: calls.append(1), warmup=10, iters=50)
        self.assertEqual(result["n"], 50)
        self.assertEqual(len(calls), 60)
        self.assertGreaterEqual(result["p99"], result["p50"])
        with self.assertRaises(ValueError):
            benchmark.bench(lambda: None, warmup=9, iters=50)


class TestStarterFiles(unittest.TestCase):
    def test_all_python_files_compile_and_have_no_unimplemented_stubs(self):
        files = sorted(STARTER.glob("*.py")) + [ROOT / "eval.py"]
        for file in files:
            py_compile.compile(str(file), doraise=True)
            tree = ast.parse(file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                    self.assertNotEqual(getattr(node.exc.func, "id", ""), "NotImplementedError", file.name)

    def test_no_eval_modification_or_extra_complete_helper_contract(self):
        self.assertTrue((ROOT / "eval.py").is_file())
        for file in STARTER.glob("*.py"):
            self.assertNotIn("import records", file.read_text(encoding="utf-8"), file.name)

    def test_notebook_is_valid_and_test_cell_is_guarded(self):
        notebook = json.loads((STARTER / "lab_day2.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        source = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                source.extend(cell["source"])
        text = "".join(source)
        self.assertIn("eval.py', 'score", text)
        self.assertIn("eval.py', 'grade", text)
        self.assertIn("b7b30f96d466fba86016aa5a26606e0f", (ROOT / "prepare_data.py").read_text(encoding="utf-8"))
        self.assertIn("RUN_TEST_ONCE = False", text)


if __name__ == "__main__":
    unittest.main()
