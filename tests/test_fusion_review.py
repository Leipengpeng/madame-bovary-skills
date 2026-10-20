"""Scoring behavior only: no model and no actual image review."""
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = (Path(__file__).resolve().parents[1]
          /"scripts"/"score_fusion_review.py")
SPEC = importlib.util.spec_from_file_location("fusion_scorer_under_test", SCRIPT)
scorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scorer)


class FusionReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="fusion-review-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j7uoAAAAASUVORK5CYII=")
        for name in ("reference.png", "candidate.png"):
            (self.root/name).write_bytes(png)
        self.review = {"schema_version": 1, "reference": str(self.root/"reference.png"),
                       "candidate": str(self.root/"candidate.png"), "hard_flags": [],
                       "checks": {name: {"status": "observed", "rating": 5,
                                         "evidence": "Synthetic reviewer evidence for arithmetic tests.",
                                         "confidence": "medium"} for name in scorer.WEIGHTS}}

    def run_review(self, review, suffix="one"):
        source = self.root/(suffix+".json")
        source.write_text(json.dumps(review), encoding="utf-8")
        return scorer.score_review(source, self.root/(suffix+"-score.json"))

    def test_missing_visible_evidence_is_rejected(self):
        for bad in (None, "", "   "):
            with self.subTest(evidence=bad):
                review = copy.deepcopy(self.review)
                if bad is None:
                    del review["checks"]["lighting_shadow"]["evidence"]
                else:
                    review["checks"]["lighting_shadow"]["evidence"] = bad
                with self.assertRaises(ValueError):
                    self.run_review(review)
                self.assertFalse((self.root/"one-score.json").exists())

    def test_evidenced_hard_failure_caps_otherwise_high_score(self):
        review = copy.deepcopy(self.review)
        review["hard_flags"] = [{"code": "contradictory_shadow",
                                 "evidence": "Synthetic shadow contradiction test fixture."}]
        result = self.run_review(review)
        self.assertEqual(result["raw_observed_score"], 100)
        self.assertEqual(result["overall_visual_fusion_score"], 59)
        self.assertEqual(result["hard_flags"], review["hard_flags"])
        self.assertEqual(result["uncertainty_score_interval"], [59, 59])

    def test_uncertain_reduces_coverage_and_na_changes_denominator(self):
        uncertain, na = copy.deepcopy(self.review), copy.deepcopy(self.review)
        for name in ("spatial_contact", "lighting_shadow"):
            for review, status in ((uncertain, "uncertain"), (na, "not_applicable")):
                review["checks"][name] = {"status": status, "confidence": "low",
                                          "evidence": "Synthetic insufficient-view or non-applicability reason."}
        unknown_score = self.run_review(uncertain, "uncertain")
        na_score = self.run_review(na, "na")
        self.assertEqual(unknown_score["observed_weight_coverage"], .45)
        self.assertEqual(unknown_score["overall_visual_fusion_score"], 69)
        self.assertEqual(unknown_score["uncertainty_score_interval"], [45, 69])
        self.assertEqual(na_score["applicable_weight"], 45)
        self.assertEqual(na_score["observed_weight_coverage"], 1)
        self.assertEqual(na_score["overall_visual_fusion_score"], 100)
        self.assertEqual(na_score["uncertainty_score_interval"], [100, 100])
        for result in (unknown_score, na_score):
            self.assertNotIn("rating", result["dimensions"]["lighting_shadow"])
            provenance = result["provenance"]
            self.assertIs(provenance["model_called"], False)
            self.assertEqual(provenance["review_sha256"],
                             hashlib.sha256(Path(provenance["review_path"]).read_bytes()).hexdigest())
            for image in provenance["images"]:
                self.assertEqual(image["sha256"], hashlib.sha256(Path(image["path"]).read_bytes()).hexdigest())
        uncertain["checks"]["lighting_shadow"]["rating"] = 5
        with self.assertRaises(ValueError):
            self.run_review(uncertain, "invented")

    def test_existing_output_is_never_overwritten(self):
        source, output = self.root/"review.json", self.root/"existing.json"
        source.write_text(json.dumps(self.review), encoding="utf-8")
        output.write_bytes(b"frozen output")
        before = output.read_bytes()
        with self.assertRaises(ValueError):
            scorer.score_review(source, output)
        cli = subprocess.run([sys.executable, "-B", str(SCRIPT), "--review", str(source),
                              "--output", str(output)], capture_output=True, text=True)
        self.assertNotEqual(cli.returncode, 0)
        self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
