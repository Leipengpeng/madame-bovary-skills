"""Plan declared subject scale/placement using image dimensions, never depth sensing."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageOps


BASES = ("creative_hypothesis", "user_provided", "measured")


def field(obj, key, kind):
    if type(obj) is not dict or type(obj.get(key)) is not kind:
        raise ValueError(f"Invalid or missing {key}: expected {kind.__name__}.")
    return obj[key]


def text(obj, key):
    value = field(obj, key, str)
    if not value.strip():
        raise ValueError(f"{key} must be non-empty.")
    return value


def texts(obj, key):
    values = field(obj, key, list)
    if not values or any(type(v) is not str or not v.strip() for v in values):
        raise ValueError(f"{key} must contain non-empty strings.")
    return values


def finite(value, name):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite, not boolean.")
    return float(value)


def interval(obj, key, maximum=None):
    values = field(obj, key, list)
    if len(values) != 2:
        raise ValueError(f"{key} needs two endpoints.")
    low, high = [finite(v, key) for v in values]
    if not 0 < low <= high or (maximum is not None and high > maximum):
        raise ValueError(f"Invalid positive ordered interval: {key}.")
    return [low, high]


def declared_basis(obj, key, note_key):
    if text(obj, key) not in BASES:
        raise ValueError(f"Unsupported declaration basis: {key}.")
    text(obj, note_key)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bounded_projection(contract, width, height):
    assumptions = field(contract, "projection_assumptions", dict)
    for key in ("shared_vertical_projection", "subject_upright", "reference_fully_visible",
                "endpoint_depth_variation_negligible", "depth_ratio_uses_optical_axis"):
        if field(assumptions, key, bool) is not True:
            raise ValueError(f"Numeric projection requires declared {key}=true; otherwise use qualitative_only.")
    text(assumptions, "basis_note")
    text(assumptions, "roll_and_distortion_treatment")
    reference, subject, depth = [field(contract, key, dict) for key in ("reference", "subject", "depth")]
    text(reference, "id")
    text(reference, "object")
    text(reference, "observation_note")
    if text(reference, "source_domain") != "real_scene":
        raise ValueError("The reference must be an actual real-scene object, not depicted content.")
    span = interval(reference, "observed_height_fraction", 1)
    ref_height = interval(reference, "physical_height_m")
    declared_basis(reference, "physical_size_basis", "physical_size_note")
    subject_height = interval(subject, "height_m")
    declared_basis(subject, "height_basis", "height_note")
    if text(subject, "pose") != "standing":
        raise ValueError("This full-height projection requires a standing subject.")
    aspect = interval(subject, "pose_width_over_full_height")
    ratio = interval(depth, "reference_over_subject")
    declared_basis(depth, "basis", "basis_note")
    relation = text(depth, "relation")
    if relation not in ("subject_in_front", "same_depth", "subject_behind", "unknown"):
        raise ValueError("Unsupported depth relation.")
    if ((relation == "subject_in_front" and ratio[0] < 1) or
        (relation == "subject_behind" and ratio[1] > 1) or
        (relation == "same_depth" and ratio != [1, 1])):
        raise ValueError("Declared optical-axis depth relation contradicts its q interval.")
    projected = [span[0]*(subject_height[0]/ref_height[1])*ratio[0],
                 span[1]*(subject_height[1]/ref_height[0])*ratio[1]]
    if any(not math.isfinite(v) or v <= 0 for v in projected):
        raise ValueError("Projected height is outside finite positive calculation bounds.")
    selection = field(contract, "selection", dict)
    chosen = finite(selection.get("full_height_fraction"), "full_height_fraction")
    chosen_aspect = finite(selection.get("pose_width_over_height"), "pose_width_over_height")
    if not projected[0] <= chosen <= projected[1] or not aspect[0] <= chosen_aspect <= aspect[1]:
        raise ValueError("Selected complete standing height and pose ratio must lie within their declared ranges.")
    center = finite(selection.get("center_x"), "center_x")
    anchor_y = finite(selection.get("anchor_y"), "anchor_y")
    anchor = text(selection, "anchor")
    text(selection, "rationale")
    if not 0 <= center <= 1 or anchor not in ("head_top", "feet"):
        raise ValueError("Invalid center or placement anchor.")
    body_width = chosen*(height/width)*chosen_aspect
    top = anchor_y if anchor == "head_top" else anchor_y-chosen
    full = [center-body_width/2, top, center+body_width/2, top+chosen]
    pixel_range = [v*height for v in projected]
    if (body_width <= 0 or any(not math.isfinite(v) for v in full+pixel_range) or
        full[2] <= full[0] or full[3] <= full[1]):
        raise ValueError("Placement exceeds finite nonempty geometry bounds.")
    clipped = [max(0., full[0]), max(0., full[1]), min(1., full[2]), min(1., full[3])]
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        raise ValueError("The selected subject is entirely outside the canvas.")
    visible_height = clipped[3]-clipped[1]
    return {"full_projected_height_fraction": projected, "projected_full_height_px_range": pixel_range,
            "selected_full_height_fraction": chosen, "selected_pose_width_over_height": chosen_aspect,
            "width_fraction": body_width, "normalized_full_bbox": full, "clipped_visible_bbox": clipped,
            "cropped_top": full[1] < 0, "cropped_bottom": full[3] > 1,
            "visible_height_fraction": visible_height, "visible_fraction_of_full_height": visible_height/chosen,
            "visible_height_fraction_basis": "fraction of canvas height, not complete standing height",
            "formula": "reference_span * (subject_height_m / reference_height_m) * (Z_reference / Z_subject)",
            "width_formula": "selected_full_height_fraction * (canvas_height / canvas_width) * pose_width_over_height"}


def plan_scene_scale(contract):
    if field(contract, "schema_version", int) != 1:
        raise ValueError("Unsupported scale schema.")
    mode = text(contract, "mode")
    if mode not in ("qualitative_only", "bounded_projection"):
        raise ValueError("Unsupported scale planning mode.")
    if field(contract, "background_locked", bool) is not True:
        raise ValueError("The reference background must remain locked.")
    texts(contract, "observations")
    texts(contract, "unknowns")
    path = Path(text(contract, "environment_image"))
    if not path.is_absolute() or not path.is_file():
        raise ValueError("environment_image must be an existing absolute image path.")
    with Image.open(path) as image:
        width, height = ImageOps.exif_transpose(image).size
    result = {"schema_version": 1, "mode": mode, "environment_image": str(path.resolve()),
              "canvas": {"width_px": width, "height_px": height, "exif_transposed": True},
              "contract": contract, "calculation_performed": mode == "bounded_projection",
              "ground_contact_verified": False,
              "limitations": ["这里只读取实际画布尺寸；场景观察、物理尺寸、深度关系及投影条件均为上游声明，未经校准验证。",
                              "不能从单张照片反推出唯一的画作尺寸、人物真实距离或真实深度；bbox不是已标定三维测量。",
                              "q为光轴深度Z_reference/Z_subject，不是欧氏距离比，也不是闪光灯距a。",
                              "明显俯仰、人物前倾、端点深度差或畸变需先校正或使用qualitative_only，不能仅靠bbox认定已校准。",
                              "完整投影高度指完整站立身高，不是可见躯干；裁切不能被当作人物变矮。",
                              "head_top/feet锚点仅为构图声明，不验证脚与地面接触或地面位置。"],
              "provenance": {"planner": "declared_projection_and_image_dimensions_only", "model_called": False,
                             "depth_measured": False, "physical_dimensions_verified": False,
                             "environment_sha256": sha256(path), "script_sha256": sha256(__file__)}}
    if mode == "qualitative_only":
        texts(contract, "distance_and_scale_relations")
        texts(contract, "placement_guidance")
    else:
        result["derived_projection"] = bounded_projection(contract, width, height)
    return result


def render_prompt(plan):
    sections = ["人物尺度与站位指导（未执行合成、测深或相机标定）：",
                "真实环境图："+plan["environment_image"],
                "实际画布（按EXIF转正）："+json.dumps(plan["canvas"], ensure_ascii=False),
                "固定原环境FOV、机位、投影和背景内容。在放置阶段允许整个人物整体等比缩放、平移，保持身体比例；不要拉伸身体或改变背景来迁就人物。",
                "上游观察、所有来源假设、数值范围与构图选择（声明不是现场实测证明）：",
                json.dumps(plan["contract"], ensure_ascii=False, indent=2),
                "限制与未知：", *plan["limitations"]]
    if plan["calculation_performed"]:
        sections += ["完整站立高度、宽度换算和裁切（归一化bbox顺序为left,top,right,bottom）：",
                     json.dumps(plan["derived_projection"], ensure_ascii=False, indent=2),
                     "完整站立高度与可见高度分别记录；超出画布可正常裁切，不能把可见躯干高度重新当作完整身高。"]
    else:
        sections += ["定性模式：只执行所列远近、尺度和放置关系，不虚构人物投影高度、深度数值或bbox。"]
    sections += ["ground_contact_verified:false；脚位锚点不构成地面接触验证。"]
    return "\n\n".join(sections)+"\n"


def compile_scale_plan(input_path, out_dir):
    input_path, out_dir = Path(input_path), Path(out_dir)
    if not input_path.is_absolute() or not out_dir.is_absolute():
        raise ValueError("Input and output directory must be absolute.")
    if out_dir.exists():
        raise ValueError("Refusing to overwrite or reuse an existing output directory.")
    raw = input_path.read_bytes()
    plan = plan_scene_scale(json.loads(raw))
    plan["provenance"].update(input_path=str(input_path.resolve()), input_sha256=hashlib.sha256(raw).hexdigest())
    contents = {"scale_plan.json": json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                "image_prompt.txt": render_prompt(plan)}
    out_dir.mkdir(parents=True, exist_ok=False)
    created = []
    try:
        for name, content in contents.items():
            path = out_dir/name
            with path.open("x", encoding="utf-8") as stream:
                created.append(path)
                stream.write(content)
    except OSError:
        for path in created:
            path.unlink(missing_ok=True)
        out_dir.rmdir()
        raise
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        compile_scale_plan(args.input, args.out_dir)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Scene scale planning failed: {exc}\n")
    print(json.dumps({"output_directory": str(args.out_dir.resolve()), "model_called": False}))


if __name__ == "__main__":
    main()
