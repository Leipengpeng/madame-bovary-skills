"""Exact frozen replay and provenance tests using synthetic pixels, never photos."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
from PIL import Image, ImageCms


SCRIPT = (Path(__file__).resolve().parents[1]
          / "scripts" / "apply_frozen_grade.py")
sys.path.insert(0, str(SCRIPT.parent))
try:
    SPEC = importlib.util.spec_from_file_location("parametric_frozen_under_test", SCRIPT)
    frozen = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(frozen)
finally:
    sys.path.pop(0)


class FrozenParametricTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="frozen-parametric-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.report_path = self.root/"grading_report.json"
        self.profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        rng = np.random.default_rng(51)
        texture = cv2.GaussianBlur(rng.integers(45, 205, (320, 320), dtype=np.uint8), (0, 0), .6)
        self.baseline = np.stack([texture+8, texture, texture-7], axis=-1)
        self.candidate = self.baseline.copy()
        self.candidate[125:195, 125:195] = [135, 120, 112]
        for name, values in (("baseline.png", self.baseline), ("reference.png", self.baseline),
                             ("candidate.png", self.candidate)):
            Image.fromarray(values).save(self.root/name)
        self.scene = {
            "id": "05", "input": str(self.root/"baseline.png"),
            "reference": str(self.root/"reference.png"), "output": str(self.root/"recorded.png"),
            "input_hashes": {str(self.root/name): frozen.tone.sha256(self.root/name)
                             for name in ("baseline.png", "reference.png")},
            "size": [320, 320],
            "brightness_curve_Lstar": {"input": [0, 50, 100], "target": [0, 51, 100]},
            "selected": {"status": "selected_by_heldout_metrics_pending_visual_review", "rejected": [],
                         "color_mode": "shared_scale_offset", "strength": .5,
                         "chroma_scale_offset": [1.01, .5, -.5]},
        }
        config_scene = {"id": "05", "input": self.scene["input"], "reference": self.scene["reference"],
                        "exclude_boxes": [[.35, .35, .65, .65]],
                        "skin_boxes": [[.4, .4, .6, .6]],
                        "audit_regions": {"background": [.05, .05, .3, .3]}}
        (self.root/"config.json").write_text(json.dumps({"scenes": [config_scene]}), encoding="utf-8")
        self.record_selection()

    def transform(self, values):
        selected = self.scene["selected"]
        curve = tuple(np.array(self.scene["brightness_curve_Lstar"][key], dtype=np.float64)
                      for key in ("input", "target"))
        return frozen.tone.to_rgb(frozen.tone.apply_transform(
            frozen.tone.lab(values), curve, np.array(selected["chroma_scale_offset"], dtype=np.float64),
            selected["strength"]))[0]

    def save_report(self):
        self.report_path.write_text(json.dumps({"scenes": [self.scene]}), encoding="utf-8")

    def record_selection(self):
        Image.fromarray(self.transform(self.baseline)).save(self.scene["output"], icc_profile=self.profile)
        self.scene["output_sha256"] = frozen.tone.sha256(self.scene["output"])
        self.save_report()

    def args(self, name="result.png"):
        return SimpleNamespace(input=str(self.root/"candidate.png"), report=str(self.report_path),
                               scene_id="05", output=str(self.root/name), candidate_regions=None,
                               report_sha256=frozen.tone.sha256(self.report_path))

    def test_both_parametric_modes_replay_exactly_apply_once_and_keep_guards(self):
        for mode in ("offset_only", "shared_scale_offset"):
            with self.subTest(mode=mode):
                self.scene["selected"]["color_mode"] = mode
                self.scene["selected"]["chroma_scale_offset"][0] = 1 if mode == "offset_only" else 1.01
                self.record_selection()
                args = self.args(mode+".png")
                before = {name: frozen.tone.sha256(self.root/name)
                          for name in ("baseline.png", "reference.png", "recorded.png", "grading_report.json")}
                result = frozen.run(args)
                actual, info = frozen.decoded_rgb(args.output)
                self.assertTrue(np.array_equal(actual, self.transform(self.candidate)))
                self.assertTrue(result["baseline_reproduction"]["exact_pixels"])
                evidence = json.loads(Path(result["evidence"]).read_text(encoding="utf-8"))
                self.assertEqual(evidence["transform_type"], "parametric_curve_chroma")
                self.assertEqual(evidence["verified_report_sha256"], args.report_sha256)
                self.assertEqual(evidence["frozen_parameters"]["brightness_curve_Lstar"],
                                 self.scene["brightness_curve_Lstar"])
                self.assertFalse(evidence["frozen_LUT_contains_base_and_residual"])
                self.assertIsNone(evidence["lut_file"])
                self.assertIn("apply_transform", evidence["function_source_sha256"])
                self.assertEqual(evidence["protection"]["rejected"], [])
                self.assertEqual(evidence["protection"]["thresholds"]["skin_p95_abs_L"], 6)
                self.assertTrue(evidence["geometry_check_against_raw_baseline"]["no_image_warp"])
                marker = json.loads(info[frozen.MARKER])
                self.assertEqual(marker["report_sha256"], args.report_sha256)
                self.assertNotIn("lut_sha256", marker)
                self.assertEqual(info["icc_profile"], self.profile)
                self.assertEqual(before, {name: frozen.tone.sha256(self.root/name) for name in before})
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    frozen.run(args)
                args.input, args.output = args.output, str(self.root/(mode+"-twice.png"))
                with self.assertRaisesRegex(ValueError, "double grading"):
                    frozen.run(args)

    def test_parametric_pin_source_hashes_and_exact_replay_reject_tampering(self):
        for failure in ("missing_pin", "report_pin", "changed_report", "baseline", "reference", "output", "replay"):
            with self.subTest(failure=failure):
                # Restore every fixture after each intentional tamper.
                for name in ("baseline.png", "reference.png"):
                    Image.fromarray(self.baseline).save(self.root/name)
                self.scene["selected"]["strength"] = .5
                self.record_selection()
                args = self.args("failed-"+failure+".png")
                if failure == "missing_pin":
                    args.report_sha256 = None
                elif failure == "report_pin":
                    args.report_sha256 = "0" * 64
                elif failure == "changed_report":
                    self.scene["selected"]["strength"] = .75
                    self.save_report()
                elif failure == "replay":
                    self.scene["selected"]["strength"] = .75
                    self.save_report()
                    args.report_sha256 = frozen.tone.sha256(self.report_path)
                else:
                    name = {"baseline": "baseline.png", "reference": "reference.png", "output": "recorded.png"}[failure]
                    (self.root/name).write_bytes(b"changed dependency")
                with self.assertRaises(ValueError):
                    frozen.run(args)
                self.assertFalse(Path(args.output).exists())
                self.assertFalse(Path(args.output).with_suffix(".frozen-grade.json").exists())

    def test_existing_lut_replay_remains_exact_without_new_report_pin(self):
        grid = np.stack(np.meshgrid(np.linspace(0, 100, 33), np.linspace(-128, 128, 33),
                                    np.linspace(-128, 128, 33), indexing="ij"), axis=-1)
        grid[..., 0] = np.minimum(grid[..., 0]+.5, 100)
        lut_path = self.root/"selected.npz"
        np.savez_compressed(lut_path, Lab_lut=grid, domain_min=[0, -128, -128], domain_max=[100, 128, 128])
        self.scene["selected"] = {"color_mode": "shared_scale_offset+regularized_3D_LUT", "rejected": [],
                                  "lut_file": str(lut_path), "lut_sha256": frozen.tone.sha256(lut_path)}
        replay, _ = frozen.tone.to_rgb(frozen.tone.lut_lookup(frozen.tone.lab(self.baseline), grid))
        Image.fromarray(replay).save(self.scene["output"], icc_profile=self.profile)
        self.scene["output_sha256"] = frozen.tone.sha256(self.scene["output"])
        self.save_report()
        args = self.args("legacy-lut.png")
        args.report_sha256 = None
        result = frozen.run(args)
        actual, info = frozen.decoded_rgb(args.output)
        expected, _ = frozen.tone.to_rgb(frozen.tone.lut_lookup(frozen.tone.lab(self.candidate), grid))
        self.assertTrue(np.array_equal(actual, expected))
        self.assertTrue(result["baseline_reproduction"]["exact_pixels"])
        evidence = json.loads(Path(result["evidence"]).read_text(encoding="utf-8"))
        self.assertEqual(evidence["transform_type"], "lab_lut")
        self.assertTrue(evidence["frozen_LUT_contains_base_and_residual"])
        self.assertEqual(json.loads(info[frozen.MARKER])["lut_sha256"], self.scene["selected"]["lut_sha256"])

    def test_parametric_validator_rejects_invalid_modes_shapes_numbers_and_order(self):
        bad_values = [float("nan"), float("inf"), True, "0.5", 10**1000]
        cases = []
        for value in bad_values+[-.1, 1.1]:
            scene = deepcopy(self.scene)
            scene["selected"]["strength"] = value
            cases.append(scene)
        for chroma in ([1, 0], [1, 0, 0, 0], [.7, 0, 0], [1.2, 0, 0], [1, 4, 0], [1, 0, float("nan")]):
            scene = deepcopy(self.scene)
            scene["selected"]["chroma_scale_offset"] = chroma
            cases.append(scene)
        for x, y in (([0, 50, 50, 100], [0, 25, 50, 100]), ([0, 100], [0]),
                     ([0, 50, 100], [0, 101, 100]), ([0, 50, 100], [0, -1, 100]),
                     ([0, 50, 100], [0, 70, 60]), ([0, float("inf"), 100], [0, 50, 100]),
                     ([1, 100], [0, 100]), ([0, 100], [1, 100])):
            scene = deepcopy(self.scene)
            scene["brightness_curve_Lstar"] = {"input": x, "target": y}
            cases.append(scene)
        for key, value in (("color_mode", "identity_seed"), ("status", "no_candidate_improved_safely"),
                           ("rejected", ["skin_change_limit"]), ("lut_file", "unexpected.npz")):
            scene = deepcopy(self.scene)
            scene["selected"][key] = value
            cases.append(scene)
        scene = deepcopy(self.scene)
        scene["selected"]["color_mode"] = "offset_only"
        cases.append(scene)  # Scale 1.01 is not an offset-only transform.
        for index, scene in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                frozen.contract_validation.validate_parametric_grade(scene)


if __name__ == "__main__":
    unittest.main()
