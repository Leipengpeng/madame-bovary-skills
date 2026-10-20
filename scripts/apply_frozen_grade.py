"""Apply a report's selected Lab LUT or parametric grade once, without fitting.

The adjacent config.json supplies the original reviewed masks. Every invocation
first replays the recorded raw input and requires exact decoded RGB agreement
with the recorded graded output. Any protection failure stops; no new grade is
fitted and no weaker grade or raw-image fallback is substituted.
An optional candidate-regions file changes only the new candidate's check masks.
"""
import argparse
from copy import deepcopy
import hashlib
import inspect
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import PIL
from PIL import Image, PngImagePlugin

import match_scene_tone as tone
import compile_lighting_contract as contract_validation


MARKER = "codex_frozen_grade"


def one_scene(document, scene_id):
    matches = [s for s in document.get("scenes", []) if str(s["id"]) == scene_id]
    if len(matches) != 1:
        raise ValueError("Scene ID must identify exactly one scene.")
    return matches[0]


def local_path(value, parent):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else parent / path).resolve(strict=True)


def decoded_rgb(path):
    # The recorded output is encoded in sRGB; compare stored sample values directly.
    with Image.open(path) as image:
        return np.array(image.convert("RGB")), image.info.copy()


def pixel_hash(rgb):
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def validate_candidate_regions(document):
    """Require explicit, non-empty candidate check regions without fitting."""
    if type(document) is not dict:
        raise ValueError("Candidate regions must be a JSON object.")

    def box(value, label):
        if type(value) is not list or len(value) != 4:
            raise ValueError(label + " must contain four normalized coordinates.")
        try:
            finite = all(type(v) in (int, float) and math.isfinite(v) for v in value)
        except OverflowError:
            finite = False
        if not finite or not (0 <= value[0] < value[2] <= 1 and
                              0 <= value[1] < value[3] <= 1):
            raise ValueError(label + " must be finite, non-boolean and satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1.")
        return [float(v) for v in value]

    result = {}
    for name in ("exclude_boxes", "skin_boxes"):
        values = document.get(name)
        if type(values) is not list or not values:
            raise ValueError(name + " must be a non-empty list of candidate boxes.")
        result[name] = [box(value, f"{name}[{index}]")
                        for index, value in enumerate(values)]
    regions = document.get("audit_regions")
    if type(regions) is not dict or not regions:
        raise ValueError("audit_regions must be a non-empty object of candidate boxes.")
    result["audit_regions"] = {}
    for name, value in regions.items():
        if type(name) is not str or not name.strip():
            raise ValueError("Candidate audit region names must be non-empty strings.")
        result["audit_regions"][name] = box(value, "audit_regions." + name)
    return result


def candidate_region_plan(args, scene_config):
    """Keep baseline configuration intact; replace new skin/audit checks only."""
    requested = getattr(args, "candidate_regions", None)
    path, checksum, overrides = None, None, None
    if requested is not None:
        path = Path(requested).expanduser()
        if not path.is_absolute():
            raise ValueError("Candidate regions path must be absolute.")
        path = path.resolve(strict=True)
        raw = path.read_bytes()
        overrides = validate_candidate_regions(json.loads(raw))
        checksum = hashlib.sha256(raw).hexdigest()
    candidate_config = scene_config if overrides is None else {
        **scene_config, **overrides}
    geometry_boxes = deepcopy(scene_config["exclude_boxes"])
    if overrides is not None:
        # boxes_mask excludes the union; both old and moved people are ignored.
        geometry_boxes.extend(deepcopy(overrides["exclude_boxes"]))
    metadata = {
        "provided": overrides is not None,
        "path": str(path) if path is not None else None,
        "sha256": checksum,
        "scope": "new_candidate_checks_only",
        "nonrefit": True,
        "baseline_reproduction_unchanged": True,
        "protection_thresholds_unchanged": True,
        "exclude_boxes": deepcopy(candidate_config["exclude_boxes"]),
        "skin_boxes": deepcopy(candidate_config.get("skin_boxes", [])),
        "audit_regions": deepcopy(candidate_config.get("audit_regions", {})),
        "geometry_exclude_boxes_union": deepcopy(geometry_boxes),
    }
    return candidate_config, geometry_boxes, metadata


def check_guards(src, graded, reference, scene_config):
    full_lab = tone.lab(src)
    actual_lab = tone.lab(graded)
    skin = tone.boxes_mask(src.shape, scene_config.get("skin_boxes", []), inside=True)
    skin_change = None
    rejected = []
    if skin.any():
        delta = actual_lab[skin] - full_lab[skin]
        skin_change = {
            "median_abs_delta_L": float(np.median(np.abs(delta[:, 0]))),
            "p95_abs_delta_L": float(np.percentile(np.abs(delta[:, 0]), 95)),
            "p95_chroma_vector_change": float(np.percentile(np.linalg.norm(delta[:, 1:], axis=1), 95)),
        }
        if skin_change["p95_abs_delta_L"] > 6 or skin_change["p95_chroma_vector_change"] > 5:
            rejected.append("skin_change_limit")
    else:
        raise ValueError("Reviewed skin boxes are required for this portrait experiment.")

    original_clip, graded_clip = tone.clip_stats(src), tone.clip_stats(graded)
    for key in ("all_channels_near_black_fraction", "all_channels_near_white_fraction"):
        if graded_clip[key] > original_clip[key] + .001:
            rejected.append("new_neutral_clipping")
    h, w = src.shape[:2]
    ref_at_size = cv2.resize(reference, (w, h), interpolation=cv2.INTER_AREA)
    regions = {}
    for name, box in scene_config.get("audit_regions", {}).items():
        mask = tone.boxes_mask(src.shape, [box], inside=True)
        limit = max(tone.clip_stats(src[mask])["any_channel_endpoint_fraction"],
                    tone.clip_stats(ref_at_size[mask])["any_channel_endpoint_fraction"]) + .003
        actual = tone.clip_stats(graded[mask])["any_channel_endpoint_fraction"]
        regions[name] = {"endpoint_fraction": actual, "endpoint_limit": limit}
        if actual > limit:
            rejected.append("new_channel_endpoint_" + name)
    return {"skin_change_not_accuracy": skin_change, "clip": graded_clip,
            "audit_regions": regions, "rejected": rejected,
            "thresholds": {"skin_p95_abs_L": 6, "skin_p95_chroma": 5,
                           "neutral_clip_increase": .001, "ROI_endpoint_margin": .003}}


def run(args):
    input_path = Path(args.input).expanduser().resolve(strict=True)
    report_path = Path(args.report).expanduser().resolve(strict=True)
    config_path = report_path.parent / "config.json"
    output_path = Path(args.output).expanduser().resolve()
    evidence_path = output_path.with_suffix(".frozen-grade.json")
    if output_path.suffix.lower() != ".png":
        raise ValueError("Use a .png output to preserve exact RGB sample values.")
    if output_path.exists() or evidence_path.exists():
        raise ValueError("Refusing to overwrite an image or its evidence file.")
    if not output_path.parent.is_dir():
        raise ValueError("Output parent directory must already exist.")

    report_raw = report_path.read_bytes()
    report_hash = hashlib.sha256(report_raw).hexdigest()
    report_pin = getattr(args, "report_sha256", None)
    if report_pin is not None and report_pin != report_hash:
        raise ValueError("Report SHA256 does not match the frozen report pin.")
    report = json.loads(report_raw)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    scene = one_scene(report, args.scene_id)
    scene_config = one_scene(config, args.scene_id)
    candidate_config, geometry_boxes, candidate_regions = candidate_region_plan(args, scene_config)
    chosen = scene["selected"]
    parametric = chosen.get("color_mode") in ("offset_only", "shared_scale_offset")
    parameters = None
    if parametric:
        if report_pin is None:
            raise ValueError("Parametric grade requires --report-sha256 from the frozen contract.")
        parameters = contract_validation.validate_parametric_grade(scene)
    elif chosen.get("rejected") or not chosen.get("lut_file") or not chosen.get("lut_sha256"):
        raise ValueError("The selected scene must contain an accepted pinned LUT or parametric grade.")
    baseline_path = local_path(scene["input"], report_path.parent)
    recorded_path = local_path(scene["output"], report_path.parent)
    reference_path = local_path(scene["reference"], report_path.parent)
    lut_path = None if parametric else local_path(chosen["lut_file"], report_path.parent)
    if local_path(scene_config["input"], config_path.parent) != baseline_path:
        raise ValueError("Adjacent config and report refer to different raw baselines.")
    if local_path(scene_config["reference"], config_path.parent) != reference_path:
        raise ValueError("Adjacent config and report refer to different references.")
    protected = {input_path, report_path, config_path, baseline_path, recorded_path,
                 reference_path, Path(tone.__file__).resolve(), Path(__file__).resolve()}
    if parametric:
        protected.add(Path(contract_validation.__file__).resolve())
    else:
        protected.add(lut_path)
    if candidate_regions["provided"]:
        protected.add(Path(candidate_regions["path"]))
    if output_path in protected or evidence_path in protected:
        raise ValueError("Output cannot replace an input, baseline, LUT or tool.")
    tracked_hashes = {str(path): tone.sha256(path) for path in protected}
    if tracked_hashes[str(report_path)] != report_hash:
        raise ValueError("Report changed while being loaded.")
    if candidate_regions["provided"] and tracked_hashes[candidate_regions["path"]] != candidate_regions["sha256"]:
        raise ValueError("Candidate regions changed while being loaded.")
    if not parametric and tracked_hashes[str(lut_path)] != chosen["lut_sha256"]:
        raise ValueError("Selected LUT hash does not match the report.")
    if tracked_hashes[str(recorded_path)] != scene["output_sha256"]:
        raise ValueError("Recorded graded output has changed.")
    for path in (baseline_path, reference_path):
        expected = next((v for k, v in scene["input_hashes"].items()
                         if Path(k).resolve() == path), None)
        if not expected or tracked_hashes[str(path)] != expected:
            raise ValueError("Recorded raw input or scene reference has changed.")

    if parametric:
        curve = tuple(np.array(parameters["brightness_curve_Lstar"][key], dtype=np.float64)
                      for key in ("input", "target"))
        chroma = np.array(parameters["chroma_scale_offset"], dtype=np.float64)

        def transform(values):
            return tone.apply_transform(values, curve, chroma, parameters["strength"])
    else:
        with np.load(lut_path, allow_pickle=False) as archive:
            lut = archive["Lab_lut"].copy()
            if not np.array_equal(archive["domain_min"], [0, -128, -128]) or not np.array_equal(archive["domain_max"], [100, 128, 128]):
                raise ValueError("LUT domain differs from the lookup implementation.")
        if lut.shape != (33, 33, 33, 3) or not np.all(np.isfinite(lut)):
            raise ValueError("Expected a finite 33-cube Lab LUT.")
        if np.any(np.diff(lut[..., 0], axis=0) < -1e-6):
            raise ValueError("Selected LUT is not monotone along input L*.")

        def transform(values):
            return tone.lut_lookup(values, lut)

    baseline, baseline_profile = tone.read_srgb(baseline_path)
    recorded, recorded_info = decoded_rgb(recorded_path)
    if not recorded_info.get("icc_profile"):
        raise ValueError("Recorded graded output must supply the exact output sRGB ICC profile.")
    if recorded.shape != baseline.shape or [baseline.shape[1], baseline.shape[0]] != scene["size"]:
        raise ValueError("Recorded raw/recorded dimensions disagree.")
    replay, replay_compressed = tone.to_rgb(transform(tone.lab(baseline)))
    difference = np.abs(replay.astype(np.int16) - recorded.astype(np.int16))
    reproduction = {
        "baseline_raw": str(baseline_path), "recorded_output": str(recorded_path),
        "exact_pixels": bool(np.array_equal(replay, recorded)),
        "max_abs_channel_difference_8bit": int(difference.max()),
        "mean_abs_channel_difference_8bit": float(difference.mean()),
        "different_pixel_count": int(np.count_nonzero(np.any(difference, axis=-1))),
        "allowed_max_abs_channel_difference_8bit": 0,
        "replay_pixel_sha256": pixel_hash(replay), "recorded_pixel_sha256": pixel_hash(recorded),
        "gamut_compressed_fraction": replay_compressed,
    }
    if not reproduction["exact_pixels"]:
        raise ValueError("Frozen replay failed exact recorded-output reproduction: " + json.dumps(reproduction))

    with Image.open(input_path) as source_image:
        if source_image.info.get(MARKER):
            raise ValueError("Input is already marked as frozen-graded; refusing double grading.")
    # Check file hashes and decoded pixels, including re-encoded graded copies.
    if any(tracked_hashes[str(input_path)] == s.get("output_sha256") for s in report.get("scenes", [])):
        raise ValueError("Input matches an already graded report output.")
    src, input_profile = tone.read_srgb(input_path)
    if src.shape != baseline.shape:
        raise ValueError("Input size must exactly match the raw baseline; no resize or warp is allowed.")
    if np.array_equal(src, recorded):
        raise ValueError("Input contains the already graded output pixels.")

    h, w = src.shape[:2]
    size = (min(w, 960), round(h * min(w, 960) / w))
    small_base = cv2.resize(baseline, size, interpolation=cv2.INTER_AREA)
    small_src = cv2.resize(src, size, interpolation=cv2.INTER_AREA)
    valid = tone.boxes_mask(small_src.shape, geometry_boxes)
    valid = cv2.erode(valid.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
    geometry = tone.geometry_check(small_base, small_src, valid)
    graded, compressed = tone.to_rgb(transform(tone.lab(src)))
    reference, reference_profile = tone.read_srgb(reference_path)
    guards = check_guards(src, graded, reference, candidate_config)
    if guards["rejected"]:
        raise ValueError("Frozen grade failed protection; parameters were not changed: " + json.dumps(guards))

    evidence = {
        "method": ("one exact application of the report's selected parametric apply_transform and current to_rgb"
                   if parametric else "one application of the report's complete selected Lab LUT and current to_rgb"),
        "transform_type": "parametric_curve_chroma" if parametric else "lab_lut",
        "verified_report_sha256": report_hash,
        "no_fit_no_parameter_search_no_warp": True, "scene_id": args.scene_id,
        "input": str(input_path), "output": str(output_path), "report": str(report_path),
        "input_and_dependency_hashes": tracked_hashes, "size": [w, h],
        "frozen_parameters": {k: chosen.get(k) for k in ("color_mode", "strength", "residual_strength", "chroma_scale_offset", "skin_color_anchors", "lut_grid_size", "lut_sha256")},
        "lut_file": str(lut_path) if lut_path is not None else None,
        "frozen_LUT_contains_base_and_residual": not parametric,
        "runtime": {"python": sys.version, "numpy": np.__version__, "opencv": cv2.__version__, "pillow": PIL.__version__},
        "function_source_sha256": {name: hashlib.sha256(inspect.getsource(getattr(tone, name)).encode("utf-8")).hexdigest() for name in ("read_srgb", "lab", "apply_transform" if parametric else "lut_lookup", "linear_rgb", "to_rgb")},
        "color_management": {"input": input_profile, "baseline": baseline_profile, "reference": reference_profile,
                             "output": "exact ICC bytes copied from verified graded output", "output_ICC_sha256": hashlib.sha256(recorded_info["icc_profile"]).hexdigest()},
        "baseline_reproduction": reproduction, "geometry_check_against_raw_baseline": geometry,
        "candidate_regions": candidate_regions,
        "gamut_compressed_fraction": compressed, "protection": guards,
        "output_pixel_sha256": pixel_hash(graded),
        "limits": "Unknown prior grading cannot be detected if metadata and provenance were removed. Geometry checks cover visible background only. Skin change is not identity or physical lighting accuracy.",
    }
    if parametric:
        evidence["frozen_parameters"] = deepcopy(parameters)
        evidence["function_source_sha256"]["validate_parametric_grade"] = hashlib.sha256(
            inspect.getsource(contract_validation.validate_parametric_grade).encode("utf-8")).hexdigest()
    for path, expected in tracked_hashes.items():
        if tone.sha256(path) != expected:
            raise ValueError("An input or dependency changed during execution.")
    metadata = PngImagePlugin.PngInfo()
    grade_marker = {"scene_id": args.scene_id, "input_sha256": tracked_hashes[str(input_path)]}
    if parametric:
        grade_marker.update({"transform_type": "parametric_curve_chroma", "report_sha256": report_hash})
    else:
        grade_marker["lut_sha256"] = chosen["lut_sha256"]
    metadata.add_text(MARKER, json.dumps(grade_marker))
    with output_path.open("xb") as stream:
        Image.fromarray(graded).save(stream, format="PNG", icc_profile=recorded_info["icc_profile"], pnginfo=metadata)
    saved_rgb, saved_info = decoded_rgb(output_path)
    if not np.array_equal(saved_rgb, graded) or saved_info.get("icc_profile") != recorded_info["icc_profile"]:
        raise ValueError("Saved output failed exact RGB/ICC read-back verification.")
    evidence["saved_RGB_and_ICC_verified"] = True
    evidence["output_sha256"] = tone.sha256(output_path)
    with evidence_path.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2)
    return {"output": str(output_path), "evidence": str(evidence_path),
            "baseline_reproduction": reproduction, "protection_rejected": guards["rejected"],
            "output_sha256": evidence["output_sha256"]}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "report", "scene-id", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--candidate-regions", help="Absolute JSON path overriding only new-candidate check regions; never refits the selected transform.")
    parser.add_argument("--report-sha256", help="Frozen contract report hash; required for parametric curve/chroma replay.")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    main()
