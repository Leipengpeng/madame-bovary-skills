"""Synthetic-image tests for declared scale ranges, placement and cropping."""
import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1]/"scripts"/"plan_scene_scale.py"
SPEC = importlib.util.spec_from_file_location("scene_scale_under_test", SCRIPT)
scale = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scale)


class SceneScaleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="scene-scale-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.image = self.root/"environment.png"
        Image.new("RGB", (400, 800), (80, 90, 100)).save(self.image)
        self.contract = {"schema_version": 1, "mode": "bounded_projection", "environment_image": str(self.image),
                         "observations": ["A complete vertical reference is visible."],
                         "unknowns": ["Depth is a declared range, not measured."], "background_locked": True,
                         "projection_assumptions": {"shared_vertical_projection": True, "subject_upright": True,
                             "reference_fully_visible": True, "endpoint_depth_variation_negligible": True,
                             "depth_ratio_uses_optical_axis": True, "basis_note": "Hypothetical shared pinhole projection.",
                             "roll_and_distortion_treatment": "Assume negligible roll/distortion; not calibrated."},
                         "reference": {"id": "R1", "object": "real frame", "source_domain": "real_scene",
                             "observed_height_fraction": [.2, .3], "observation_note": "Declared visible span.",
                             "physical_height_m": [1.4, 2], "physical_size_basis": "creative_hypothesis",
                             "physical_size_note": "Size range chosen for this experiment."},
                         "subject": {"height_m": [1.7, 1.9], "height_basis": "user_provided", "height_note": "User range.",
                             "pose": "standing", "pose_width_over_full_height": [.18, .24]},
                         "depth": {"reference_over_subject": [1, 1.2], "relation": "subject_in_front",
                             "basis": "creative_hypothesis", "basis_note": "Optical-axis depth ratio, not flash distance."},
                         "selection": {"full_height_fraction": .35, "pose_width_over_height": .2, "center_x": .5,
                             "anchor": "feet", "anchor_y": 1, "rationale": "Keep complete body in this trial."}}

    def plan(self, contract=None):
        return scale.plan_scene_scale(self.contract if contract is None else contract)

    def test_interval_extrema_match_declared_projection_formula(self):
        before = copy.deepcopy(self.contract)
        result = self.plan()
        derived = result["derived_projection"]
        reference, subject, depth = [self.contract[k] for k in ("reference", "subject", "depth")]
        values = [span*(person/ref)*ratio for span,person,ref,ratio in itertools.product(
            reference["observed_height_fraction"], subject["height_m"], reference["physical_height_m"],
            depth["reference_over_subject"])]
        self.assertAlmostEqual(derived["full_projected_height_fraction"][0], min(values))
        self.assertAlmostEqual(derived["full_projected_height_fraction"][1], max(values))
        for pixel, fraction in zip(derived["projected_full_height_px_range"], derived["full_projected_height_fraction"]):
            self.assertAlmostEqual(pixel, fraction*800)
        self.assertEqual(self.contract, before)
        self.assertIs(result["ground_contact_verified"], False)
        self.assertIs(result["provenance"]["depth_measured"], False)

    def test_canvas_aspect_ratio_converts_body_pixel_aspect_correctly(self):
        first = self.plan()["derived_projection"]
        self.assertAlmostEqual(first["width_fraction"], .35*(800/400)*.2)
        self.assertAlmostEqual(first["width_fraction"]*400, .35*800*.2)
        landscape = self.root/"landscape.png"
        Image.new("RGB", (800, 400), "gray").save(landscape)
        contract = copy.deepcopy(self.contract)
        contract["environment_image"] = str(landscape)
        second = self.plan(contract)["derived_projection"]
        self.assertAlmostEqual(second["width_fraction"], .35*(400/800)*.2)
        self.assertAlmostEqual(first["width_fraction"], second["width_fraction"]*4)

    def test_exif_transpose_sets_actual_canvas_dimensions(self):
        path = self.root/"rotated.jpg"
        image = Image.new("RGB", (80, 40), "gray")
        exif = image.getexif()
        exif[274] = 6
        image.save(path, exif=exif)
        contract = copy.deepcopy(self.contract)
        contract["environment_image"] = str(path)
        before = path.read_bytes()
        result = self.plan(contract)
        self.assertEqual((result["canvas"]["width_px"], result["canvas"]["height_px"]), (40, 80))
        self.assertEqual(path.read_bytes(), before)

    def test_cropping_preserves_complete_standing_height_and_rejects_fully_off_frame(self):
        contract = copy.deepcopy(self.contract)
        contract["reference"].update(observed_height_fraction=[.7, .7], physical_height_m=[1, 1])
        contract["subject"]["height_m"] = [2, 2]
        contract["depth"].update(reference_over_subject=[1, 1], relation="same_depth")
        contract["selection"]["full_height_fraction"] = 1.4
        feet = self.plan(contract)["derived_projection"]
        self.assertEqual(feet["full_projected_height_fraction"], [1.4, 1.4])
        self.assertTrue(feet["cropped_top"])
        self.assertFalse(feet["cropped_bottom"])
        self.assertAlmostEqual(feet["normalized_full_bbox"][1], -.4)
        self.assertAlmostEqual(feet["visible_height_fraction"], 1)
        self.assertAlmostEqual(feet["visible_fraction_of_full_height"], 1/1.4)
        invalid = copy.deepcopy(contract)
        invalid["selection"]["full_height_fraction"] = feet["visible_height_fraction"]
        with self.assertRaises(ValueError):
            self.plan(invalid)
        contract["selection"].update(anchor="head_top", anchor_y=0)
        head = self.plan(contract)["derived_projection"]
        self.assertFalse(head["cropped_top"])
        self.assertTrue(head["cropped_bottom"])
        contract["selection"]["anchor_y"] = 2
        with self.assertRaises(ValueError):
            self.plan(contract)

    def test_invalid_domains_bases_depth_relations_and_projection_assumptions_are_rejected(self):
        failures = [("reference", "source_domain", "painting_content"),
                    ("reference", "physical_size_basis", "photo_guessed_fact"),
                    ("subject", "height_basis", "inferred_as_fact"), ("depth", "basis", "estimated"),
                    ("depth", "reference_over_subject", [.8, .9]), ("depth", "relation", "same_depth"),
                    ("depth", "relation", "subject_behind"), ("depth", "reference_over_subject", [0, 1]),
                    ("reference", "observed_height_fraction", [.2, 1.1]),
                    ("reference", "physical_height_m", [2, 1]), ("subject", "height_m", [True, 2]),
                    ("subject", "pose", "sitting"), ("selection", "full_height_fraction", float("nan")),
                    ("selection", "pose_width_over_height", .8), ("selection", "center_x", 1.1),
                    ("selection", "anchor_y", float("inf"))]
        failures += [("projection_assumptions", key, False) for key in (
            "shared_vertical_projection", "subject_upright", "reference_fully_visible",
            "endpoint_depth_variation_negligible", "depth_ratio_uses_optical_axis")]
        failures.append(("projection_assumptions", "roll_and_distortion_treatment", ""))
        for section, key, value in failures:
            with self.subTest(section=section, key=key, value=value):
                contract = copy.deepcopy(self.contract)
                contract[section][key] = value
                with self.assertRaises(ValueError):
                    self.plan(contract)
        # Unknown depth ordering is allowed, but q must still be an explicitly declared positive interval.
        contract = copy.deepcopy(self.contract)
        contract["depth"].update(relation="unknown", reference_over_subject=[.9, 1.1])
        self.assertTrue(self.plan(contract)["calculation_performed"])

    def test_qualitative_mode_keeps_relations_without_projection_numbers(self):
        contract = {key: copy.deepcopy(self.contract[key]) for key in (
            "schema_version", "environment_image", "observations", "unknowns", "background_locked")}
        contract.update(mode="qualitative_only", distance_and_scale_relations=["Person is in front of the frame; depth unknown."],
                        placement_guidance=["Place complete standing body without enlarging the background."])
        result = self.plan(contract)
        self.assertIs(result["calculation_performed"], False)
        self.assertNotIn("derived_projection", result)
        prompt = scale.render_prompt(result)
        for value in contract["distance_and_scale_relations"]+contract["placement_guidance"]:
            self.assertIn(value, prompt)
        for key in ("full_projected_height_fraction", "normalized_full_bbox", "width_fraction"):
            self.assertNotIn(key, prompt)

    def test_cli_writes_contract_and_cropping_then_refuses_overwrite_without_touching_inputs(self):
        source, output = self.root/"input.json", self.root/"plan"
        source.write_text(json.dumps(self.contract), encoding="utf-8")
        original_input, original_image = source.read_bytes(), self.image.read_bytes()
        command = [sys.executable, "-B", str(SCRIPT), "--input", str(source), "--out-dir", str(output)]
        first = subprocess.run(command, capture_output=True, text=True, check=True)
        self.assertIs(json.loads(first.stdout)["model_called"], False)
        saved = json.loads((output/"scale_plan.json").read_text(encoding="utf-8"))
        prompt = (output/"image_prompt.txt").read_text(encoding="utf-8")
        self.assertEqual(saved["contract"], self.contract)
        self.assertEqual(saved["provenance"]["input_sha256"], hashlib.sha256(original_input).hexdigest())
        self.assertEqual(saved["provenance"]["environment_sha256"], hashlib.sha256(original_image).hexdigest())
        for text in (str(self.image), "FOV", "cropped_top", "visible_height_fraction",
                     self.contract["projection_assumptions"]["roll_and_distortion_treatment"]):
            self.assertIn(text, prompt)
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        second = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, before)
        self.assertEqual(source.read_bytes(), original_input)
        self.assertEqual(self.image.read_bytes(), original_image)


if __name__ == "__main__":
    unittest.main()
