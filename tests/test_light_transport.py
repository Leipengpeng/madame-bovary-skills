"""Offline bounds tests for explicitly declared simplified flash geometry."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1]/"scripts"/"plan_light_transport.py"
SPEC = importlib.util.spec_from_file_location("light_transport_under_test", SCRIPT)
transport = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transport)


class FlashGeometryTests(unittest.TestCase):
    def geometry(self):
        return {"enabled": True, "basis": "creative_hypothesis", "basis_note": "Test assumptions, not photo measurements.",
                "arrangement": "collinear_parallel_planes", "receiver": "real wall",
                "source_to_subject_m": [2, 4], "subject_to_receiver_m": [1, 3], "emitter_diameter_m": [.01, .02]}

    def test_bounds_match_all_corner_extrema(self):
        geometry = self.geometry()
        result = transport.estimate_flash_geometry(geometry)
        corners = list(itertools.product(geometry["source_to_subject_m"], geometry["subject_to_receiver_m"],
                                         geometry["emitter_diameter_m"]))
        metrics = {"source_to_receiver_m": lambda a,b,d: a+b,
                   "flash_irradiance_ratio_bg_to_subject": lambda a,b,d: (a/(a+b))**2,
                   "shadow_linear_magnification": lambda a,b,d: 1+b/a,
                   "penumbra_width_m": lambda a,b,d: d*b/a}
        for key, formula in metrics.items():
            with self.subTest(metric=key):
                values = [formula(*corner) for corner in corners]
                self.assertAlmostEqual(result[key][0], min(values))
                self.assertAlmostEqual(result[key][1], max(values))
        self.assertIs(result["calculation_performed"], True)
        self.assertIs(result["basis_verified"], False)
        self.assertEqual(result["estimation"], "bounded_geometry")
        self.assertEqual(result["irradiance_model"], "point_like_direct")
        self.assertAlmostEqual(result["max_emitter_size_to_subject_distance_ratio"], .01)

    def test_known_formula_and_zero_gap_corner(self):
        geometry = self.geometry()
        geometry.update(source_to_subject_m=[2, 2], subject_to_receiver_m=[1, 1], emitter_diameter_m=[.01, .01])
        result = transport.estimate_flash_geometry(geometry)
        expected = {"source_to_receiver_m": 3, "flash_irradiance_ratio_bg_to_subject": 4/9,
                    "shadow_linear_magnification": 1.5, "penumbra_width_m": .005}
        for key, value in expected.items():
            for endpoint in result[key]:
                self.assertAlmostEqual(endpoint, value)
        geometry["subject_to_receiver_m"] = [0, 0]
        zero = transport.estimate_flash_geometry(geometry)
        self.assertEqual(zero["source_to_receiver_m"], [2, 2])
        self.assertEqual(zero["flash_irradiance_ratio_bg_to_subject"], [1, 1])
        self.assertEqual(zero["shadow_linear_magnification"], [1, 1])
        self.assertEqual(zero["penumbra_width_m"], [0, 0])

    def test_invalid_intervals_and_unsupported_assumptions_are_rejected(self):
        for key, values in (("source_to_subject_m", [0, 2]), ("subject_to_receiver_m", [-1, 2]),
                            ("emitter_diameter_m", [0, .02]), ("source_to_subject_m", [4, 2]),
                            ("source_to_subject_m", [True, 2]), ("subject_to_receiver_m", [0, float("nan")]),
                            ("emitter_diameter_m", [.01, float("inf")]), ("source_to_subject_m", [2]),
                            ("source_to_subject_m", ["2", 3]), ("enabled", 1), ("basis", "estimated_from_photo"),
                            ("basis_note", "  "), ("receiver", ""), ("arrangement", "arbitrary_3d")):
            with self.subTest(key=key, values=values):
                geometry = self.geometry()
                geometry[key] = values
                with self.assertRaises(ValueError):
                    transport.estimate_flash_geometry(geometry)
        for basis in ("creative_hypothesis", "user_provided", "measured"):
            geometry = self.geometry()
            geometry["basis"] = basis
            self.assertIs(transport.estimate_flash_geometry(geometry)["basis_verified"], False)

    def test_disabled_and_qualitative_geometry_do_not_make_numeric_estimates(self):
        self.assertEqual(transport.estimate_flash_geometry({"enabled": False}),
                         {"enabled": False, "calculation_performed": False})
        qualitative = {"enabled": True, "estimation": "qualitative_only", "basis": "user_provided",
                       "basis_note": "User only knows relative positions.", "receiver": "wall",
                       "distance_relations": ["Wall is behind subject; distance unknown."]}
        before = copy.deepcopy(qualitative)
        result = transport.estimate_flash_geometry(qualitative)
        self.assertEqual(qualitative, before)
        self.assertIs(result["calculation_performed"], False)
        self.assertEqual(result["distance_relations"], qualitative["distance_relations"])
        for key in ("source_to_receiver_m", "flash_irradiance_ratio_bg_to_subject",
                    "shadow_linear_magnification", "penumbra_width_m"):
            self.assertNotIn(key, result)
        for field, value in (("distance_relations", []), ("distance_relations", [""]),
                             ("source_to_subject_m", [2, 3])):
            invalid = {**qualitative, field: value}
            with self.assertRaises(ValueError):
                transport.estimate_flash_geometry(invalid)

    def test_large_angular_source_or_non_direct_irradiance_model_requires_qualitative_mode(self):
        for key, value in (("emitter_diameter_m", [.01, .3]), ("irradiance_model", "diffuse_bounce")):
            with self.subTest(key=key):
                invalid = self.geometry()
                invalid[key] = value
                with self.assertRaisesRegex(ValueError, "qualitative_only"):
                    transport.estimate_flash_geometry(invalid)
        boundary = self.geometry()
        boundary["emitter_diameter_m"] = [.01, .2]
        boundary["irradiance_model"] = "point_like_direct"
        result = transport.estimate_flash_geometry(boundary)
        self.assertAlmostEqual(result["max_emitter_size_to_subject_distance_ratio"], .1)
        self.assertIs(result["calculation_performed"], True)

    def test_cli_writes_new_json_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="light-transport-test-") as temporary:
            root = Path(temporary)
            source, output = root/"input.json", root/"estimate.json"
            source.write_text(json.dumps(self.geometry()), encoding="utf-8")
            original = source.read_bytes()
            command = [sys.executable, "-B", str(SCRIPT), "--input", str(source), "--output", str(output)]
            first = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertIs(json.loads(first.stdout)["model_called"], False)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), transport.estimate_flash_geometry(self.geometry()))
            saved = output.read_bytes()
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(output.read_bytes(), saved)
            self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
