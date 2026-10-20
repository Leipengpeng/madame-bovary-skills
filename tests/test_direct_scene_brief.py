"""Default direct workflow checks; no user photographs or model calls."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("direct_brief_under_test", ROOT/"scripts"/"compile_scene_brief.py")
compiler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compiler)


class DirectSceneBriefTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="direct-brief-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.brief = json.loads((ROOT/"examples"/"direct_scene_brief.template.json").read_text(encoding="utf-8"))
        for role in ("environment", "identity_reference"):
            path = self.root/(role+".png")
            path.write_bytes(b"synthetic file-presence fixture; no pixel analysis claimed")
            self.brief["inputs"][role] = str(path)

    def compile(self, brief=None, output="compiled"):
        path = self.root/"brief.json"
        path.write_text(json.dumps(self.brief if brief is None else brief), encoding="utf-8")
        return compiler.compile_brief(path, self.root/output)

    def test_default_compiles_without_grade_and_pins_exact_assigned_references(self):
        result = self.compile()
        self.assertFalse(result["provenance"]["model_called"])
        refs = result["provenance"]["reference_order"]
        self.assertEqual([ref["role"] for ref in refs], ["environment", "identity_reference"])
        for ref in refs:
            self.assertEqual(ref["sha256"], hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest())
        prompt = (self.root/"compiled/image_prompt.txt").read_text(encoding="utf-8")
        self.assertIn("POSTPROCESS: none", prompt)
        self.assertIn("Nose and philtrum", prompt)
        self.assertIn("source-surface-camera", prompt)
        self.assertIn("foreground", prompt)
        self.assertFalse(result["derived_light_transport"]["calculation_performed"])

    def test_rejects_extra_identity_or_reused_input_roles(self):
        extra = copy.deepcopy(self.brief)
        extra["inputs"]["identity_references"] = [extra["inputs"]["identity_reference"]]
        with self.assertRaisesRegex(ValueError, "one assigned identity"):
            self.compile(extra)
        duplicate = copy.deepcopy(self.brief)
        duplicate["inputs"]["identity_reference"] = duplicate["inputs"]["environment"]
        with self.assertRaisesRegex(ValueError, "distinct roles"):
            self.compile(duplicate)

    def test_optional_outfit_has_explicit_secondary_authority(self):
        outfit = self.root/"outfit.png"
        outfit.write_bytes(b"synthetic outfit fixture")
        self.brief["inputs"]["outfit_reference"] = str(outfit)
        result = self.compile()
        self.assertEqual(result["provenance"]["reference_order"][2]["role"], "outfit_reference")
        prompt = (self.root/"compiled/image_prompt.txt").read_text(encoding="utf-8")
        self.assertIn("never a replacement identity", prompt)

    def test_multi_reference_is_opt_in_and_keeps_ordered_same_person_authority(self):
        second = self.root/"second_identity.png"
        second.write_bytes(b"second synthetic same-person reference fixture")
        first = self.brief["inputs"].pop("identity_reference")
        self.brief["inputs"]["identity_references"] = [first, str(second)]
        with self.assertRaisesRegex(ValueError, "explicit multi_reference"):
            self.compile()
        self.brief["identity"]["reference_mode"] = "multi_reference"
        result = self.compile()
        self.assertEqual([item["path"] for item in result["provenance"]["reference_order"]][1:],
                         [first, str(second)])
        prompt = (self.root/"compiled/image_prompt.txt").read_text(encoding="utf-8")
        self.assertIn("All assigned identity references depict the same person", prompt)

    def test_rejects_missing_self_shadow_plan_and_unknown_reflection_light(self):
        for field in ("facial_shadow_checks", "specular_surfaces"):
            invalid = copy.deepcopy(self.brief)
            if field == "facial_shadow_checks":
                invalid["flash_detail_plan"][field] = []
            else:
                invalid["flash_detail_plan"][field][0]["light_id"] = "UNDECLARED"
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.compile(invalid)

    def test_fov_change_or_unknown_postprocess_is_rejected(self):
        unlocked = copy.deepcopy(self.brief)
        unlocked["composition"]["environment_fov_locked"] = False
        with self.assertRaisesRegex(ValueError, "FOV"):
            self.compile(unlocked)
        invalid = copy.deepcopy(self.brief)
        invalid["postprocess"]["mode"] = "automatic_regrade"
        with self.assertRaisesRegex(ValueError, "Postprocess"):
            self.compile(invalid)

    def test_depicted_light_cannot_be_declared_real_scene_evidence(self):
        invalid = copy.deepcopy(self.brief)
        invalid["analysis"]["scene_evidence"][0]["region"] = "inside_painting sun"
        with self.assertRaisesRegex(ValueError, "real scene"):
            self.compile(invalid)

    def test_ambient_mode_compiles_without_positive_flash_instructions(self):
        ambient = copy.deepcopy(self.brief)
        ambient["lighting_mode"] = "ambient_shadow_match"
        ambient["synthetic_lights"] = []
        ambient["background_light_exceptions"] = []
        ambient["flash_geometry"] = {"enabled": False}
        ambient.pop("flash_detail_plan")
        track = ambient["shadow_tracking"]["tracks"][0]
        track["light_id"] = "L1"
        track["direction"] = "Follow the observed environmental light direction"
        track["softness"] = "Match broad environmental-source penumbra"
        ambient["shadow_tracking"]["unknowns"] = ["Actual environmental light direction remains uncertain"]
        ambient["acceptance"] = ["Observed environmental lighting only; no additional light"]
        result = self.compile(ambient)
        prompt = (self.root/"compiled/image_prompt.txt").read_text(encoding="utf-8")
        self.assertIn("AMBIENT LIGHT ONLY", prompt)
        self.assertIn("Do not add a phone flash", prompt)
        self.assertNotIn("Flash is a local positive contribution", prompt)
        self.assertNotIn("A near-axis body shadow may be mostly hidden", prompt)
        self.assertNotIn('"light_id": "F1"', prompt)
        self.assertFalse(result["derived_light_transport"]["enabled"])

    def test_existing_artifacts_and_inputs_remain_unchanged(self):
        before = {path: Path(path).read_bytes() for path in self.brief["inputs"].values()}
        self.compile()
        prompt = self.root/"compiled/image_prompt.txt"
        existing = prompt.read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.compile()
        self.assertEqual(existing, prompt.read_bytes())
        self.assertTrue(all(Path(path).read_bytes() == data for path, data in before.items()))


if __name__ == "__main__":
    unittest.main()
