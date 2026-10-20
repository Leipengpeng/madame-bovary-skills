"""Estimate explicitly supplied near-axis flash geometry; never measure a photo."""
import argparse
import json
import math
from pathlib import Path


def _text(obj, key):
    value = obj.get(key)
    if type(value) is not str or not value.strip():
        raise ValueError(f"{key} must be a non-empty string.")
    return value


def _interval(obj, key, positive):
    values = obj.get(key)
    if type(values) is not list or len(values) != 2:
        raise ValueError(f"{key} must be a two-number interval.")
    try:
        valid = all(type(v) in (int, float) and math.isfinite(v) for v in values)
    except OverflowError:
        valid = False
    if not valid or values[0] > values[1] or (values[0] <= 0 if positive else values[0] < 0):
        raise ValueError(f"Invalid finite ordered interval: {key}.")
    return [float(v) for v in values]


def estimate_flash_geometry(geometry):
    """Return endpoint bounds for parallel planes and a collinear near-axis source.

    Distances and their basis are upstream declarations, not verified measurements.
    Irradiance ratios use a small-angular-size direct point-source approximation,
    equal incidence angle, no occlusion and equal source intensity. They do not
    predict camera exposure or material/pixel brightness.
    """
    if type(geometry) is not dict or type(geometry.get("enabled")) is not bool:
        raise ValueError("flash_geometry.enabled must be a boolean.")
    if not geometry["enabled"]:
        return {"enabled": False, "calculation_performed": False}
    basis = _text(geometry, "basis")
    if basis not in ("creative_hypothesis", "user_provided", "measured"):
        raise ValueError("Unsupported geometry basis.")
    basis_note = _text(geometry, "basis_note")
    receiver = _text(geometry, "receiver")
    estimation = geometry.get("estimation", "bounded_geometry")
    if estimation not in ("bounded_geometry", "qualitative_only"):
        raise ValueError("Unsupported estimation mode.")
    common = {"enabled": True, "estimation": estimation, "basis": basis,
              "basis_note": basis_note, "basis_verified": False, "receiver": receiver,
              "limitations": ["输入距离与basis由上游声明，脚本未从照片测量或验证，包括标为measured的输入。",
                              "不是三维渲染器，不预测相机曝光、材质反射或像素亮度。",
                              "M为墙面投影的物理线性尺寸倍率，不可乘照片中人物的像素尺寸。",
                              "半影宽度是阴影过渡范围，不是Gaussian sigma；完全近轴的投影可隐藏在主体后。"]}
    if estimation == "qualitative_only":
        relations = geometry.get("distance_relations")
        if type(relations) is not list or not relations or any(type(v) is not str or not v.strip() for v in relations):
            raise ValueError("Qualitative geometry needs non-empty distance_relations.")
        if any(key in geometry for key in ("source_to_subject_m", "subject_to_receiver_m", "emitter_diameter_m")):
            raise ValueError("Qualitative-only mode must not supply numeric geometry intervals.")
        return {**common, "calculation_performed": False, "distance_relations": relations,
                "assumptions": ["仅记录距离关系与未知量；未给定距离区间，不计算数值照度比、投影倍率或半影。"]}
    if _text(geometry, "arrangement") != "collinear_parallel_planes":
        raise ValueError("Only collinear_parallel_planes is supported.")
    a = _interval(geometry, "source_to_subject_m", True)
    b = _interval(geometry, "subject_to_receiver_m", False)
    diameter = _interval(geometry, "emitter_diameter_m", True)
    irradiance_model = geometry.get("irradiance_model", "point_like_direct")
    if irradiance_model != "point_like_direct":
        raise ValueError("Only point_like_direct supports numeric irradiance bounds; use estimation='qualitative_only'.")
    max_size_ratio = diameter[1]/a[0]
    if max_size_ratio > .1:
        raise ValueError("Emitter Dmax/a_min exceeds this tool's conservative 0.1 applicability threshold, not a physical law; use estimation='qualitative_only'.")
    bounds = {
        "source_to_receiver_m": [a[0]+b[0], a[1]+b[1]],
        "flash_irradiance_ratio_bg_to_subject": [(a[0]/(a[0]+b[1]))**2, (a[1]/(a[1]+b[0]))**2],
        "shadow_linear_magnification": [1+b[0]/a[1], 1+b[1]/a[0]],
        "penumbra_width_m": [diameter[0]*(b[0]/a[1]), diameter[1]*(b[1]/a[0])],
    }
    if not all(math.isfinite(v) for pair in bounds.values() for v in pair):
        raise ValueError("Declared geometry exceeds finite calculation bounds.")
    return {
        **common, "calculation_performed": True, "arrangement": "collinear_parallel_planes",
        "irradiance_model": irradiance_model,
        "max_emitter_size_to_subject_distance_ratio": max_size_ratio,
        "source_to_subject_m": a, "subject_to_receiver_m": b, "emitter_diameter_m": diameter,
        **bounds,
        "assumptions": ["近轴光源与主体、接收面沿同轴排列，主体和接收面为平行平面。",
                        "照度模型仅为小角尺寸直达点源近似（point_like_direct），不适用于漫反射反弹光（bounce）。",
                        "照度比仅比较等入射角、未遮挡、同光强条件下的直达闪光贡献，不含环境光。",
                        "Dmax/a_min≤0.1仅为本工具保守适用门槛，不是物理定律；非小光源不声称1/r²适用。",
                        "范围是声明输入端点的简化几何界限，不是统计置信区间。"],
        "formulas": {"source_to_receiver_m": "a+b",
                     "flash_irradiance_ratio_bg_to_subject": "(a/(a+b))^2",
                     "shadow_linear_magnification": "1+b/a", "penumbra_width_m": "D*b/a"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if not args.input.is_absolute() or not args.output.is_absolute():
            raise ValueError("Input and output paths must be absolute.")
        if args.output.exists():
            raise ValueError("Refusing to overwrite a transport estimate.")
        result = estimate_flash_geometry(json.loads(args.input.read_bytes()))
        content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Light transport planning failed: {exc}\n")
    print(json.dumps({"output": str(args.output.resolve()), "model_called": False}))


if __name__ == "__main__":
    main()
