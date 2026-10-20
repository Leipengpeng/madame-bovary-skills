"""Score reviewer-supplied visual observations; this script does not inspect images.

The result is a subjective editorial score, never a calibrated probability.
CLI: --review ABSOLUTE_REVIEW_JSON --output ABSOLUTE_OUTPUT_JSON
"""
import argparse
import hashlib
import json
from pathlib import Path


WEIGHTS = {"identity": 15, "spatial_contact": 25, "lighting_shadow": 30,
           "texture_edges": 15, "background_preservation": 15}
CONFIDENCES = {"low", "medium", "high"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def field(obj, name, kind):
    if type(obj) is not dict or name not in obj or type(obj[name]) is not kind:
        raise ValueError(f"Invalid or missing {name}: expected {kind.__name__}.")
    return obj[name]


def text(obj, name):
    value = field(obj, name, str)
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")
    return value


def existing_image(review, role):
    path = Path(text(review, role))
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"{role} must be an existing absolute image path.")
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".avif"}:
        raise ValueError(f"{role} must have a supported image filename extension.")
    return path.resolve()


def score_review(review_path, output_path):
    review_path, output_path = Path(review_path), Path(output_path)
    if not review_path.is_absolute() or not output_path.is_absolute():
        raise ValueError("Review and output paths must be absolute.")
    if output_path.exists():
        raise ValueError("Refusing to overwrite an existing output or input.")
    raw = review_path.read_bytes()
    review = json.loads(raw)
    if field(review, "schema_version", int) != 1:
        raise ValueError("Unsupported review schema.")
    images = [(role, existing_image(review, role)) for role in ("reference", "candidate")]
    if output_path.resolve() in [review_path.resolve(), *[p for _, p in images]]:
        raise ValueError("Output cannot replace a review or reference image.")
    checks = field(review, "checks", dict)
    if set(checks) != set(WEIGHTS):
        raise ValueError("All five weighted review dimensions must be supplied exactly once.")
    observed_weight = applicable_weight = 0
    earned = 0.
    dimensions, deductions, unknowns = {}, [], []
    for name, weight in WEIGHTS.items():
        item = checks[name]
        status, evidence = text(item, "status"), text(item, "evidence")
        confidence = text(item, "confidence")
        if confidence not in CONFIDENCES or status not in {"observed", "uncertain", "not_applicable"}:
            raise ValueError(f"Invalid status or confidence in {name}.")
        dimensions[name] = {"weight": weight, "status": status,
                            "evidence": evidence, "confidence": confidence}
        if status == "not_applicable":
            if "rating" in item:
                raise ValueError("Not-applicable dimensions cannot have invented ratings.")
            continue
        applicable_weight += weight
        if status == "uncertain":
            if "rating" in item:
                raise ValueError("Uncertain dimensions cannot have invented ratings.")
            unknowns.append(name)
            continue
        rating = field(item, "rating", int)
        if not 0 <= rating <= 5:
            raise ValueError("Observed ratings must be integers between zero and five.")
        dimensions[name]["rating"] = rating
        observed_weight += weight
        earned += weight*rating/5
    if not observed_weight:
        raise ValueError("At least one evidenced, observed dimension is needed to score.")
    flags = field(review, "hard_flags", list)
    for flag in flags:
        text(flag, "code")
        text(flag, "evidence")
        if "confidence" in flag and text(flag, "confidence") not in CONFIDENCES:
            raise ValueError("Invalid hard-flag confidence.")
    coverage = observed_weight/applicable_weight
    raw_score = earned/observed_weight*100
    cap, gates = 100, []
    if observed_weight*10 < applicable_weight*7:
        cap = 69
        gates.append({"reason": "observed_weight_coverage_below_0.7", "cap": 69})
    if flags:
        cap = min(cap, 59)
        gates.append({"reason": "evidenced_hard_flags", "cap": 59})
    for name, item in dimensions.items():
        if item["status"] == "observed" and item["rating"] < 5:
            deductions.append({"dimension": name, "rating": item["rating"],
                               "points_lost_on_observed_scale": round(item["weight"]*(5-item["rating"])/5/observed_weight*100, 2),
                               "evidence": item["evidence"], "confidence": item["confidence"]})
    deductions.sort(key=lambda item: item["points_lost_on_observed_scale"], reverse=True)
    lower = earned/applicable_weight*100
    upper = (earned+applicable_weight-observed_weight)/applicable_weight*100
    result = {
        "schema_version": 1, "overall_visual_fusion_score": round(min(raw_score, cap), 2),
        "score_definition": "主观编辑参考分，非校准概率；脚本不识图，评分依据由独立审图者提供。",
        "raw_observed_score": round(raw_score, 2), "observed_weight_coverage": round(coverage, 4),
        "observed_weight": observed_weight, "applicable_weight": applicable_weight,
        "uncertain_dimensions": unknowns, "uncertainty_score_interval": [round(min(lower, cap), 2), round(min(upper, cap), 2)],
        "interval_definition": "适用但未观察维度按0至5分给出的上下界，已应用总分上限；不是统计置信区间。",
        "dimensions": dimensions, "major_deductions": deductions[:3], "hard_flags": flags, "score_caps": gates,
        "provenance": {"scorer": "deterministic_editorial_rubric", "model_called": False,
                       "review_path": str(review_path.resolve()), "review_sha256": hashlib.sha256(raw).hexdigest(),
                       "scorer_sha256": sha256(__file__),
                       "images": [{"role": role, "path": str(path), "sha256": sha256(path)} for role, path in images]},
        "limits": "仅验证图片路径存在和扩展名，不验证像素内容、审图证据真实性或reviewer置信度的校准。"
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = score_review(args.review, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Review scoring failed: {exc}\n")
    print(json.dumps({"output": str(args.output.resolve()), "score": result["overall_visual_fusion_score"]}))


if __name__ == "__main__":
    main()
