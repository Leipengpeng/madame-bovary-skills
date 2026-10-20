"""Candidate check-mask behavior; JSON and in-memory arrays only, no photos.

Run with ``python -m unittest discover -s tests -p test_frozen_grade_regions.py -v``.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = (Path(__file__).resolve().parents[1]
          / "scripts" / "apply_frozen_grade.py")
sys.path.insert(0, str(SCRIPT.parent))
try:
    SPEC = importlib.util.spec_from_file_location("frozen_grade_regions_under_test", SCRIPT)
    frozen = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(frozen)
finally:
    sys.path.pop(0)


def original_config():
    return {"id": "03", "input": "unchanged-raw.png",
            "exclude_boxes": [[.1, .1, .3, .3]],
            "skin_boxes": [[.1, .1, .2, .2]],
            "audit_regions": {"old_background": [.3, .1, .4, .2]}}


def moved_regions():
    return {"exclude_boxes": [[.6, .6, .9, .9]],
            "skin_boxes": [[.7, .7, .8, .8]],
            "audit_regions": {"new_background": [.4, .4, .5, .5]}}


class FrozenGradeRegionTests(unittest.TestCase):
    def test_regions_reject_invalid_or_disabled_protection_boxes(self):
        good = moved_regions()
        self.assertEqual(frozen.validate_candidate_regions(good), good)
        invalid = [None, [], {}, {**good, "exclude_boxes": []},
                   {**good, "skin_boxes": []}, {**good, "audit_regions": {}},
                   {**good, "audit_regions": []},
                   {**good, "audit_regions": {" ": [0, 0, 1, 1]}}]
        invalid_boxes = [[True, 0, 1, 1], [0, False, 1, 1],
                         [float("nan"), 0, 1, 1], [0, 0, float("inf"), 1],
                         [-.1, 0, 1, 1], [0, 0, 1.1, 1],
                         [.5, 0, .5, 1], [0, 1, 1, 0], [0, 0, 1],
                         ["0", 0, 1, 1], [0, 0, 10**1000, 1]]
        for name in ("exclude_boxes", "skin_boxes", "audit_regions"):
            for box in invalid_boxes:
                document = deepcopy(good)
                document[name] = {"region": box} if name == "audit_regions" else [box]
                invalid.append(document)
        for index, document in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                frozen.validate_candidate_regions(document)

    def test_union_ignores_old_and_new_person_and_skin_guard_follows_new_location(self):
        original = original_config()
        unchanged = deepcopy(original)
        moved = moved_regions()
        with tempfile.TemporaryDirectory(prefix="frozen-regions-test-") as temporary:
            path = Path(temporary) / "candidate.json"
            path.write_text(json.dumps(moved), encoding="utf-8")
            candidate, boxes, _ = frozen.candidate_region_plan(
                SimpleNamespace(candidate_regions=str(path)), original)
        self.assertEqual(original, unchanged)
        self.assertEqual(candidate["skin_boxes"], moved["skin_boxes"])
        self.assertEqual(candidate["audit_regions"], moved["audit_regions"])
        mask = frozen.tone.boxes_mask((100, 100, 3), boxes)
        self.assertFalse(mask[15, 15])
        self.assertFalse(mask[75, 75])
        self.assertTrue(mask[45, 45])

        src = np.full((100, 100, 3), 100, dtype=np.uint8)
        source_lab = np.zeros(src.shape, dtype=np.float32)
        changed_lab = source_lab.copy()
        changed_lab[70:80, 70:80, 0] = 8
        with patch.object(frozen.tone, "lab", side_effect=[source_lab, changed_lab]):
            new_guards = frozen.check_guards(src, src.copy(), src, candidate)
        with patch.object(frozen.tone, "lab", side_effect=[source_lab, changed_lab]):
            old_guards = frozen.check_guards(src, src.copy(), src, original)
        self.assertIn("skin_change_limit", new_guards["rejected"])
        self.assertNotIn("skin_change_limit", old_guards["rejected"])
        self.assertEqual(set(new_guards["audit_regions"]), {"new_background"})
        self.assertEqual(new_guards["thresholds"], old_guards["thresholds"])

    def test_programmatic_args_keep_original_masks_and_thresholds(self):
        original = original_config()
        for args in (SimpleNamespace(), SimpleNamespace(candidate_regions=None)):
            with self.subTest(args=vars(args)):
                candidate, boxes, metadata = frozen.candidate_region_plan(args, original)
                self.assertIs(candidate, original)
                self.assertEqual(boxes, original["exclude_boxes"])
                self.assertEqual(metadata["skin_boxes"], original["skin_boxes"])
                self.assertEqual(metadata["audit_regions"], original["audit_regions"])
                self.assertFalse(metadata["provided"])
                self.assertIsNone(metadata["path"])
                self.assertIsNone(metadata["sha256"])

    def test_metadata_pins_exact_json_and_records_candidate_only_nonrefit_scope(self):
        original = original_config()
        moved = moved_regions()
        with tempfile.TemporaryDirectory(prefix="frozen-regions-evidence-") as temporary:
            path = Path(temporary) / "candidate.json"
            raw = (json.dumps(moved, indent=2) + "\n").encode("utf-8")
            path.write_bytes(raw)
            _, boxes, metadata = frozen.candidate_region_plan(
                SimpleNamespace(candidate_regions=path), original)
            self.assertEqual(metadata["path"], str(path.resolve()))
            self.assertEqual(metadata["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertTrue(metadata["provided"])
            self.assertEqual(metadata["scope"], "new_candidate_checks_only")
            self.assertTrue(metadata["nonrefit"])
            self.assertTrue(metadata["baseline_reproduction_unchanged"])
            self.assertTrue(metadata["protection_thresholds_unchanged"])
            self.assertEqual(metadata["exclude_boxes"], moved["exclude_boxes"])
            self.assertEqual(metadata["skin_boxes"], moved["skin_boxes"])
            self.assertEqual(metadata["audit_regions"], moved["audit_regions"])
            self.assertEqual(metadata["geometry_exclude_boxes_union"], boxes)
            with self.assertRaisesRegex(ValueError, "absolute"):
                frozen.candidate_region_plan(SimpleNamespace(candidate_regions="candidate.json"), original)


if __name__ == "__main__":
    unittest.main()
