"""Behavior checks for deterministic grading, using synthetic images only.

Run with ``python -m unittest discover -s tests -p test_scene_tone.py -v``.
All generated images live in TemporaryDirectory, never in the sample folders.
"""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image


SCRIPT = (Path(__file__).resolve().parents[1]
          / "scripts" / "match_scene_tone.py")
SPEC = importlib.util.spec_from_file_location("scene_tone_under_test", SCRIPT)
tone = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tone)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def textured_scene(seed=731):
    """Colorful, spatially distributed texture with repeatable feature points."""
    rng = np.random.default_rng(seed)
    height, width = 384, 512
    coarse = rng.integers(45, 210, (16, 22, 3), dtype=np.uint8)
    image = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_LINEAR)
    image = np.clip(image.astype(float) + rng.normal(0, 5, image.shape),
                    20, 235).astype(np.uint8)
    for _ in range(180):
        center = tuple(int(n) for n in rng.integers([8, 8], [width-8, height-8]))
        color = tuple(int(n) for n in rng.integers(30, 225, 3))
        cv2.circle(image, center, int(rng.integers(3, 12)), color,
                   int(rng.integers(1, 3)), lineType=cv2.LINE_AA)
    return image


def rgb_from_lab(values):
    converted = cv2.cvtColor(values.astype(np.float32), cv2.COLOR_LAB2RGB)
    return np.rint(np.clip(converted, 0, 1) * 255).astype(np.uint8)


class SceneToneTests(unittest.TestCase):
    def test_curve_is_bounded_and_monotone_even_with_adversarial_samples(self):
        rng = np.random.default_rng(17)
        x = np.linspace(1, 99, 6000)
        targets = {
            "reversed": 100-x,
            "noisy": x + rng.normal(0, 18, x.shape),
        }
        for name, y in targets.items():
            with self.subTest(samples=name):
                xx, yy = tone.fit_curve(x, y)
                self.assertEqual(len(xx), len(yy))
                self.assertTrue(np.isfinite(xx).all() and np.isfinite(yy).all())
                self.assertTrue(np.all(np.diff(xx) > 0))
                np.testing.assert_allclose([xx[0], xx[-1], yy[0], yy[-1]],
                                           [0, 100, 0, 100], atol=1e-6)
                evaluated = np.interp(np.linspace(0, 100, 1001), xx, yy)
                self.assertTrue(np.all(np.diff(evaluated) >= -1e-7))
                self.assertGreaterEqual(float(evaluated.min()), 0)
                self.assertLessEqual(float(evaluated.max()), 100)

    def test_identity_rgb_lab_roundtrip_preserves_channel_values(self):
        rng = np.random.default_rng(91)
        rgb = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        # Include neutral endpoints and saturated colors, not just midtones.
        rgb[0, :8] = np.array([[r, g, b] for r in (0, 255)
                               for g in (0, 255) for b in (0, 255)])
        gray = np.arange(256, dtype=np.uint8)
        rgb[1:5] = np.repeat(gray[:, None], 3, axis=1).reshape(4, 64, 3)
        restored = rgb_from_lab(tone.lab(rgb))
        error = np.abs(restored.astype(np.int16)-rgb.astype(np.int16))
        self.assertLessEqual(int(error.max()), 3)

    def test_d65_conversion_preserves_rgb_and_compresses_chroma_not_lightness(self):
        rng = np.random.default_rng(103)
        # Interior sRGB colors must survive the new conversion, not just OpenCV's.
        rgb = rng.integers(4, 252, (64, 64, 3), dtype=np.uint8)
        restored, _ = tone.to_rgb(tone.lab(rgb))
        self.assertEqual(restored.shape, rgb.shape)
        error = np.abs(restored.astype(np.int16)-rgb.astype(np.int16))
        self.assertLessEqual(int(error.max()), 3)

        # Every L* here has an attainable neutral. Chroma is deliberately outside
        # sRGB, spanning hue directions rather than one easy red-only example.
        lightness, angle = np.meshgrid(np.linspace(30, 75, 12),
                                      np.linspace(0, 2*np.pi, 24, endpoint=False),
                                      indexing="ij")
        radius = rng.uniform(150, 190, lightness.shape)
        vivid = np.stack([lightness, radius*np.cos(angle), radius*np.sin(angle)],
                         axis=-1).astype(np.float32)
        compressed, fraction = tone.to_rgb(vivid)
        self.assertGreater(fraction, .95)
        self.assertEqual(compressed.shape, vivid.shape)
        endpoint_pixels = np.any((compressed == 0) | (compressed == 255), axis=-1)
        self.assertLessEqual(float(endpoint_pixels.mean()), .01)
        recovered_lightness = tone.lab(compressed)[..., 0]
        self.assertLess(float(np.max(np.abs(recovered_lightness-lightness))), 1.)

    def test_complete_lut_and_interpolation_keep_lightness_monotone(self):
        rng = np.random.default_rng(211)
        grid = np.stack(np.meshgrid(np.linspace(0, 100, 33),
                                    np.linspace(-128, 128, 33),
                                    np.linspace(-128, 128, 33), indexing="ij"),
                        axis=-1)
        residual = rng.uniform(-6, 6, grid.shape)
        # Confirm the stimulus contains reversals before the LUT regularization.
        self.assertTrue(np.any(np.diff((grid+residual)[..., 0], axis=0) < 0))
        identity_curve = (np.array([0., 100.]), np.array([0., 100.]))
        lut = tone.complete_lut(grid, residual, identity_curve,
                                np.array([1., 0., 0.]), 1., 1.)
        self.assertEqual(lut.shape, (33, 33, 33, 3))
        self.assertTrue(np.isfinite(lut).all())
        self.assertTrue(np.all(np.diff(lut[..., 0], axis=0) >= -1e-6))
        np.testing.assert_allclose(lut[0, ..., 0], 0, atol=1e-6)
        np.testing.assert_allclose(lut[-1, ..., 0], 100, atol=1e-6)

        ab = rng.uniform(-128, 128, (24, 2))
        ab = np.vstack([ab, [[0, 0], [-128, -128], [128, 128],
                                [-128, 128], [128, -128]]])
        samples = np.empty((len(ab), 401, 3), dtype=np.float32)
        samples[..., 0] = np.linspace(0, 100, samples.shape[1])
        samples[..., 1:] = ab[:, None, :]
        interpolated = tone.lut_lookup(samples, lut)
        self.assertEqual(interpolated.shape, samples.shape)
        self.assertTrue(np.isfinite(interpolated).all())
        self.assertTrue(np.all(np.diff(interpolated[..., 0], axis=1) >= -1e-5))

    def test_geometry_check_rejects_a_translated_background(self):
        reference = textured_scene()
        moved = np.roll(reference, 24, axis=1)
        mask = np.ones(reference.shape[:2], dtype=bool)
        cv2.setRNGSeed(31)
        # Many real feature matches remain; they are at the wrong coordinates.
        with self.assertRaises(ValueError):
            tone.geometry_check(reference, moved, mask)

    def test_scene_recovers_tone_without_mutation_or_overwrite(self):
        reference = textured_scene()
        shifted = tone.lab(reference)
        shifted[..., 0] += 5
        shifted[..., 1:] = shifted[..., 1:] * 1.06 + np.array([1., -1.])
        source = rgb_from_lab(shifted)
        height, width = source.shape[:2]
        # An unmatched foreground is deliberately excluded from background fit.
        box = [.42, .23, .64, .84]
        x0, y0, x1, y1 = (int(box[0]*width), int(box[1]*height),
                          int(np.ceil(box[2]*width)), int(np.ceil(box[3]*height)))
        source[y0:y1, x0:x1] = [165, 120, 90]
        background = np.ones((height, width), dtype=bool)
        background[y0:y1, x0:x1] = False

        with tempfile.TemporaryDirectory(prefix="scene-tone-test-") as temporary:
            folder = Path(temporary)
            ref_path, src_path = folder/"reference.png", folder/"input.png"
            Image.fromarray(reference).save(ref_path)
            Image.fromarray(source).save(src_path)
            originals = {p: digest(p) for p in (ref_path, src_path)}
            scene = {
                "id": "synthetic-textured-scene",
                "reference": str(ref_path), "input": str(src_path),
                "output": "corrected.png", "exclude_boxes": [box],
                "skin_boxes": [box],
            }
            cv2.setRNGSeed(31)
            result = tone.process_scene(scene, folder)
            output = folder/scene["output"]
            self.assertTrue(output.is_file())
            with Image.open(output) as saved:
                self.assertEqual(saved.size, (width, height))
                corrected = np.array(saved.convert("RGB"))
            self.assertEqual(corrected.shape, source.shape)
            self.assertIsInstance(result, dict)
            for p, checksum in originals.items():
                self.assertEqual(digest(p), checksum)

            ref_lab = tone.lab(reference)[background]
            before = tone.lab(source)[background]-ref_lab
            after = tone.lab(corrected)[background]-ref_lab
            self.assertLess(float(np.median(np.linalg.norm(after, axis=1))),
                            float(np.median(np.linalg.norm(before, axis=1))) * .5)
            self.assertLess(float(np.mean(np.abs(after[:, 0]))),
                            float(np.mean(np.abs(before[:, 0]))) * .5)

            accepted_hash = digest(output)
            with self.assertRaises(ValueError):
                tone.process_scene(scene, folder)
            self.assertEqual(digest(output), accepted_hash)
            for p, checksum in originals.items():
                self.assertEqual(digest(p), checksum)


if __name__ == "__main__":
    unittest.main()
