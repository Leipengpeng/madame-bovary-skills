"""Validate and compile a lighting contract offline, without calling a model."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def field(obj, key, kind):
    if type(obj) is not dict or key not in obj or type(obj[key]) is not kind:
        raise ValueError(f"Invalid or missing {key}: expected {kind.__name__}.")
    return obj[key]


def texts(obj, key):
    values = field(obj, key, list)
    if any(type(v) is not str or not v.strip() for v in values):
        raise ValueError(f"{key} must contain non-empty strings.")
    return values


def file_path(value):
    if type(value) is not str or not Path(value).is_absolute() or not Path(value).is_file():
        raise ValueError("A required input must be an existing absolute file path.")
    return Path(value).resolve()


def confidence(obj):
    value = field(obj, "confidence", str)
    if value not in ("low", "medium", "high"):
        raise ValueError("Invalid confidence.")
    return value


def shadow_and_flash(contract, evidence_ids, environment_ids=()):
    """Validate explicit light/caster/receiver chains; no geometry is inferred here."""
    mode=field(contract,'lighting_mode',str)
    if mode not in ('ambient_shadow_match','direct_phone_flash'):
        raise ValueError('Unsupported lighting mode.')
    lights=field(contract,'synthetic_lights',list)
    exceptions=field(contract,'background_light_exceptions',list)
    ids=set()
    lines=['灯光模式：'+mode]
    for light in lights:
        lid=field(light,'id',str)
        if not lid or lid in ids or field(light,'kind',str)!='phone_flash':
            raise ValueError('Synthetic light must have a unique phone_flash ID.')
        if field(light,'authority',str)!='user_requested_synthetic':
            raise ValueError('Added light must be identified as user-requested synthetic light.')
        ids.add(lid)
        lines.append('新增光源 '+lid+'（用户指定创作设置，不是原图实测光源）：'+
                     '; '.join(field(light,k,str) for k in ('position','size','relative_strength','falloff','color_response')))
    if mode=='ambient_shadow_match' and (lights or exceptions):
        raise ValueError('Ambient mode cannot add a light or arbitrary background relighting.')
    if mode=='direct_phone_flash' and len(lights)!=1:
        raise ValueError('This mode requires exactly one camera-near phone flash.')
    if mode=='direct_phone_flash' and not exceptions:
        raise ValueError('Flash needs explicit bounded receiver illumination exceptions.')
    for ex in exceptions:
        if field(ex,'light_id',str) not in ids:
            raise ValueError('Background exception refers to an unknown added light.')
        box=field(ex,'box',list)
        if len(box)!=4 or any(type(v) not in (int,float) for v in box) or not (0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1):
            raise ValueError('Receiver exceptions need normalized nonempty boxes.')
        lines.append('背景受光例外 '+field(ex,'region',str)+' '+str(box)+': '+field(ex,'allowed_change',str))
    tracking=field(contract,'shadow_tracking',dict)
    if field(tracking,'enabled',bool) is not True:
        raise ValueError('Shadow tracking must be enabled.')
    lines += ['阴影未知项（不得当作事实）：'+s for s in texts(tracking,'unknowns')]
    tracks=field(tracking,'tracks',list)
    if not tracks:
        raise ValueError('At least one shadow relationship is required.')
    track_ids=set()
    for track in tracks:
        tid=field(track,'id',str); lid=field(track,'light_id',str)
        if not tid or tid in track_ids:
            raise ValueError('Shadow IDs must be unique.')
        track_ids.add(tid)
        if lid not in ids and lid not in environment_ids and not (lid.startswith('ambient:') and lid[8:] in evidence_ids):
            raise ValueError('Shadow source must resolve to added light or scene evidence.')
        receiver=field(track,'receiver',dict)
        if field(receiver,'source_domain',str) not in ('real_scene','person_surface'):
            raise ValueError('A shadow receiver must be a real surface.')
        visibility=field(track,'visibility',str)
        if visibility not in ('visible','mostly_occluded','off_frame','uncertain'):
            raise ValueError('Invalid shadow visibility.')
        for eid in texts(track,'evidence_ids'):
            if eid not in evidence_ids:
                raise ValueError('Unknown shadow evidence.')
        lines.append('阴影链 '+tid+': '+lid+' → '+field(track,'occluder',str)+' → '+field(receiver,'region',str)+
                     '; '+field(track,'direction',str)+'; '+field(track,'softness',str)+'; visibility='+visibility+
                     '; '+field(track,'attachment',str)+'; confidence='+confidence(track)+
                     '; evidence_ids='+','.join(texts(track,'evidence_ids')))
    if mode=='direct_phone_flash' and not any(t['light_id'] in ids for t in tracks):
        raise ValueError('Flash mode must track the flash-produced shadow, even if mostly hidden.')
    if mode=='direct_phone_flash':
        lines.append('固定环境曝光下，新增闪光是原环境光上的正向贡献。被遮挡的闪光接收区保留原环境照明；近轴投影通常藏在主体后，不为证明有阴影而画大片黑影。背景仅允许上述光照例外，物体/画作内容和几何保持。')
    return mode,lines


def nonempty(obj, key):
    value = field(obj, key, str)
    if not value.strip():
        raise ValueError(f"{key} must be non-empty.")
    return value


def environment_lighting(contract, evidence_ids):
    sources = field(field(contract, "environment_lighting", dict), "sources", list)
    if not sources:
        raise ValueError("At least one environmental light source is required.")
    ids, lines = set(), ["环境光源模型（几何类型与传输方式分别记录；不是实测重建）。这些仅解释原图已经记录的环境照明，用来让人物适应该光场；不新增灯具、不对原背景再次叠加环境光，不因分析出射灯而增加墙面亮斑："]
    enums = {"geometry_type": ("point_like", "area", "directional", "unknown"),
             "transport": ("direct", "diffuse_bounce", "mixed", "unknown"),
             "role": ("key", "fill", "ambient", "unknown")}
    for source in sources:
        lid = nonempty(source, "id")
        if lid in ids:
            raise ValueError("Environmental light IDs must be unique.")
        ids.add(lid)
        for key, allowed in enums.items():
            if field(source, key, str) not in allowed:
                raise ValueError(f"Invalid environmental light {key}.")
        direction, color = field(source, "direction", dict), field(source, "color", dict)
        for obj, keys in ((direction, ("image_space", "depth")), (color, ("tendency", "basis"))):
            for key in keys:
                nonempty(obj, key)
        nonempty(source, "apparent_size")
        confidence(source)
        refs = texts(source, "evidence_ids")
        if not refs or any(eid not in evidence_ids for eid in refs):
            raise ValueError("Environmental lights need non-empty real-scene evidence references.")
        # Explicit keys retain every independent physical/observational field in the prompt.
        lines.append("环境光源 "+lid+"："+json.dumps(source, ensure_ascii=False))
    return ids, lines


def flash_transport(contract, mode):
    geometry = field(contract, "flash_geometry", dict)
    if field(geometry, "enabled", bool) is not (mode == "direct_phone_flash"):
        raise ValueError("Flash geometry must be enabled only for direct_phone_flash mode.")
    spec = importlib.util.spec_from_file_location("_light_transport", Path(__file__).with_name("plan_light_transport.py"))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    result = helper.estimate_flash_geometry(geometry)
    return result, ["闪光几何输入/推导（上游声明，非照片测距；仅近轴平行面简化）：",
                    json.dumps(result, ensure_ascii=False, allow_nan=False)]


def flash_details(contract, mode, evidence_ids, light_ids):
    """Validate authored depth, reflection and facial-shadow plans, not scene physics."""
    if mode != "direct_phone_flash" and "flash_detail_plan" not in contract:
        return []
    detail = field(contract, "flash_detail_plan", dict)

    def evidence_refs(item):
        refs = texts(item, "evidence_ids")
        if any(eid not in evidence_ids for eid in refs):
            raise ValueError("Flash detail plan refers to unknown scene evidence.")
        # Empty refs deliberately mean authored creative instructions, not observations.

    def declared_light(item):
        if nonempty(item, "light_id") not in light_ids:
            raise ValueError("Flash detail plan needs a declared environmental/synthetic light_id.")

    layers = field(detail, "depth_layers", list)
    if not layers:
        raise ValueError("Flash detail plan needs non-empty depth_layers.")
    layer_ids = set()
    for layer in layers:
        lid = nonempty(layer, "id")
        if lid in layer_ids:
            raise ValueError("Flash depth layer IDs must be unique.")
        layer_ids.add(lid)
        if field(layer, "layer", str) not in ("foreground", "subject", "background"):
            raise ValueError("Invalid flash depth layer category.")
        for key in ("region", "relative_distance", "flash_response"):
            nonempty(layer, key)
        confidence(layer)
        evidence_refs(layer)

    links = field(detail, "shadow_links", list)
    if not links:
        raise ValueError("Flash detail plan needs non-empty shadow_links.")
    known_tracks = {track["id"] for track in contract["shadow_tracking"]["tracks"]}
    linked_tracks = set()
    for link in links:
        tid = nonempty(link, "track_id")
        if tid not in known_tracks or tid in linked_tracks:
            raise ValueError("Flash shadow_links need unique existing track_id references.")
        linked_tracks.add(tid)
        for key in ("occluder_layer_id", "receiver_layer_id"):
            if nonempty(link, key) not in layer_ids:
                raise ValueError("Flash shadow_links refer to an unknown depth layer.")
        nonempty(link, "visibility_reason")

    surfaces = field(detail, "specular_surfaces", list)
    if not surfaces:
        nonempty(detail, "specular_absence_reason")
    surface_ids = set()
    for surface in surfaces:
        sid = nonempty(surface, "id")
        if sid in surface_ids:
            raise ValueError("Specular surface IDs must be unique.")
        surface_ids.add(sid)
        for key in ("region", "material", "orientation", "reflection_path", "highlight_action"):
            nonempty(surface, key)
        box = field(surface, "box", list)
        if (len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
                or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)):
            raise ValueError("Specular surfaces need finite normalized non-empty boxes.")
        if field(surface, "expected_visibility", str) not in ("visible", "conditional", "not_visible", "uncertain"):
            raise ValueError("Invalid specular expected_visibility.")
        declared_light(surface)
        confidence(surface)
        evidence_refs(surface)

    facial = field(detail, "facial_shadow_checks", list)
    if not facial:
        raise ValueError("Flash detail plan needs non-empty facial_shadow_checks.")
    for check in facial:
        for key in ("region", "occluder", "receiver", "direction", "action"):
            nonempty(check, key)
        if field(check, "visibility", str) not in ("visible", "mostly_occluded", "off_frame", "uncertain"):
            raise ValueError("Invalid facial shadow visibility.")
        declared_light(check)
        confidence(check)

    return [
        "闪光细节计划（上游人工编写的创作假设，不是自动测距、材质法线或物理真值重建；"
        "evidence_ids 为空表示无实测依据的创作指令，非空仅引用原图观察）：",
        json.dumps(detail, ensure_ascii=False, allow_nan=False),
        "按 depth_layers 的前景、人物与后景关系执行距离衰减，并逐条遵循 shadow_links 的遮挡者与接收面。"
        "阴影可见性依既定光源、接收面和遮挡决定；不为加强效果把近轴隐藏投影强行外移，不压黑原有环境照明。",
        "镜面亮点必须满足光源→该材质表面→相机的反射路径；朝向不支持时应弱化或不出现。"
        "不得在所有玻璃、金属或镜头上复制同一亮点，不添加无关轮廓光或逆光。"
        "按 facial_shadow_checks 保留鼻、下唇、下巴等明确遮挡产生的局部阴影；直闪也不能把面部抹成平面，"
        "同时不为显示阴影而虚构侧光。",
    ]


def validate_parametric_grade(scene):
    """Validate a selected tone fitter result without fitting or sampling a LUT."""
    chosen = field(scene, "selected", dict)
    mode = field(chosen, "color_mode", str)
    if (mode not in ("offset_only", "shared_scale_offset")
            or field(chosen, "status", str) != "selected_by_heldout_metrics_pending_visual_review"
            or field(chosen, "rejected", list)):
        raise ValueError("Parametric grade must be an accepted selected offset_only/shared_scale_offset result.")
    if any(key in chosen for key in ("lut_file", "lut_sha256", "residual_strength", "lut_grid_size")):
        raise ValueError("Parametric grade cannot contain a LUT or residual transform.")

    def number(value):
        try:
            return type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            return False

    strength = chosen.get("strength")
    if not number(strength) or not 0 <= strength <= 1:
        raise ValueError("Selected parametric strength must be finite and in [0, 1].")
    chroma = field(chosen, "chroma_scale_offset", list)
    if (len(chroma) != 3 or not all(number(v) for v in chroma)
            or not .8 <= chroma[0] <= 1.15 or any(abs(v) > 3 for v in chroma[1:])
            or (mode == "offset_only" and chroma[0] != 1)):
        raise ValueError("Selected chroma must be a bounded finite [shared_scale, a_offset, b_offset].")
    curve = field(scene, "brightness_curve_Lstar", dict)
    x, y = field(curve, "input", list), field(curve, "target", list)
    if (len(x) < 2 or len(x) != len(y) or not all(number(v) and 0 <= v <= 100 for v in x+y)
            or x[0] != 0 or y[0] != 0 or x[-1] != 100 or y[-1] != 100
            or any(a >= b for a, b in zip(x, x[1:]))
            or any(a > b for a, b in zip(y, y[1:]))):
        raise ValueError("Selected Lstar curve needs finite matched monotone knots with fixed 0/100 endpoints.")
    return {"color_mode": mode, "strength": strength, "chroma_scale_offset": chroma,
            "brightness_curve_Lstar": {"input": x, "target": y}}


def compile_contract(contract_path, out_dir):
    contract_path, out_dir = Path(contract_path), Path(out_dir)
    if not contract_path.is_absolute() or not out_dir.is_absolute():
        raise ValueError("Contract and output directory must be absolute.")
    outputs = [out_dir/"image_prompt.txt", out_dir/"compiled_contract.json"]
    if any(p.exists() for p in outputs):
        raise ValueError("Refusing to overwrite compiled artifacts.")
    raw = contract_path.read_bytes()
    contract = json.loads(raw)
    version=field(contract, "schema_version", int)
    if version not in (1,2,3,4) or field(contract, "mode", str) != "subject_lighting_edit":
        raise ValueError("Unsupported contract schema or mode.")
    inputs = field(contract, "inputs", dict)
    paths = [("edit_target", file_path(field(inputs, "edit_target", str))),
             ("environment", file_path(field(inputs, "environment", str)))]
    paths += [("identity_reference", file_path(p)) for p in texts(inputs, "identity_references")]
    if len(paths) < 3:
        raise ValueError("At least one identity reference is required.")
    analysis = field(contract, "analysis", dict)
    evidence, observations = {}, []
    for item in field(analysis, "scene_evidence", list):
        eid, region = field(item, "id", str), field(item, "region", str)
        if not eid.strip() or eid in evidence:
            raise ValueError("Scene evidence IDs must be non-empty and unique.")
        domain = field(item, "source_domain", str)
        forbidden = ("画内", "画中", "画作内部", "painting_content", "artwork_content", "depicted_scene", "inside_painting")
        if domain != "real_scene" or any(tag in region.lower() for tag in forbidden):
            raise ValueError("Depicted artwork cannot establish real-scene lighting evidence.")
        observation, inference = field(item, "observation", str), field(item, "inference", str)
        observations.append(f"证据 {eid}（真实场景区域：{region}；信心：{confidence(item)}）\n观察：{observation}\n推断（非实测事实）：{inference}")
        evidence[eid] = item
    if not evidence:
        raise ValueError("At least one real-scene evidence item is required.")
    source_ids, environment_sections = environment_lighting(contract, evidence) if version >= 3 else (set(), [])
    lighting_mode,extra_sections=shadow_and_flash(contract,evidence,source_ids) if version>=2 else ('subject_relight_v1',[])
    derived = None
    if version >= 3:
        synthetic_ids = {s["id"] for s in contract["synthetic_lights"]}
        if source_ids & synthetic_ids:
            raise ValueError("Environmental and synthetic light IDs must not collide.")
        source_ids |= synthetic_ids
        derived, transport_sections = flash_transport(contract, lighting_mode)
        extra_sections += environment_sections + transport_sections
        if version >= 4:
            extra_sections += flash_details(contract, lighting_mode, evidence, source_ids)

    def references(obj):
        ids = texts(obj, "evidence_ids")
        if any(eid not in evidence for eid in ids):
            raise ValueError("Unresolved scene evidence reference.")
        return ", ".join(ids)

    hypothesis = field(contract, "selected_hypothesis", dict)
    hid, statement = field(hypothesis, "id", str), field(hypothesis, "statement", str)
    uncertainty = field(hypothesis, "uncertainty", str)
    hypothesis_text = (f"工作假设 {hid}（尚未证实；信心：{confidence(hypothesis)}；依据：{references(hypothesis)}）：{statement}\n"
                       f"不确定性：{uncertainty}")
    constraints = field(contract, "constraints", dict)
    required = {"geometry_locked": True, "identity_locked": True, "pose_locked": True,
                "background_light_locked": True, "subject_light_edit_allowed": True,
                "atmospheric_effects_allowed": False}
    if any(field(constraints, key, bool) is not value for key, value in required.items()):
        raise ValueError("Geometry, identity, pose and existing ambient illumination must stay locked; declared lighting edits remain allowed.")
    plan = []
    for item in field(contract, "subject_plan", list):
        region, prompt = field(item, "region", str), field(item, "prompt", str)
        basis = "依赖工作假设，非已证实光源" if field(item, "hypothesis_based", bool) else "依据观察的操作计划"
        entry = f"- {region}：{prompt}（{basis}；依据：{references(item)}）"
        if version >= 3:
            refs = texts(item, "source_ids")
            if not refs or any(lid not in source_ids for lid in refs):
                raise ValueError("Each v3/v4 subject region needs known environmental/synthetic source_ids.")
            entry += ("\n  source_ids="+", ".join(refs)+"；material_response="+nonempty(item, "material_response")+
                      "；shadow_response="+nonempty(item, "shadow_response"))
        plan.append(entry)
    if not plan:
        raise ValueError("A subject lighting plan is required.")
    frozen = field(contract, "frozen_grade", dict)
    if field(frozen, "refit", bool) is not False:
        raise ValueError("Frozen grade must not be refitted.")
    report_path = file_path(field(frozen, "report", str))
    report = json.loads(report_path.read_bytes())
    scene_id = field(frozen, "scene_id", str)
    scenes = [s for s in field(report, "scenes", list) if field(s, "id", str) == scene_id]
    if len(scenes) != 1:
        raise ValueError("Frozen grade scene must resolve uniquely.")
    selected = field(scenes[0], "selected", dict)
    report_hash = sha256(report_path)
    parametric = frozen.get("transform_type") == "parametric_curve_chroma"
    lut_hash = None
    if parametric:
        if version < 4:
            raise ValueError("Parametric frozen-grade contracts require schema v4.")
        if field(frozen, "report_sha256", str) != report_hash:
            raise ValueError("Frozen parametric report hash does not match the pinned report.")
        parameters = validate_parametric_grade(scenes[0])
        frozen_text = (f"冻结追色：沿用场景 {scene_id} 已选参数曲线/色度变换，一次应用原始 "
                       f"apply_transform + to_rgb；报告 SHA256 {report_hash}；禁止重新拟合、转为近似 LUT 或叠加第二次。\n"+
                       json.dumps(parameters, ensure_ascii=False, allow_nan=False))
    else:
        if "transform_type" in frozen and frozen["transform_type"] != "lab_lut":
            raise ValueError("Unsupported frozen transform_type.")
        lut_hash = field(frozen, "lut_sha256", str)
        lut_path = file_path(field(selected, "lut_file", str))
        if field(selected, "lut_sha256", str) != lut_hash or sha256(lut_path) != lut_hash:
            raise ValueError("Frozen LUT hash does not match its selected report and file.")
        frozen_text = f"冻结追色：沿用场景 {scene_id} 已选 LUT {lut_hash}；禁止重新拟合。"
    sections = [field(contract, "base_prompt", str), "引用顺序（不得交换角色）："]
    sections += [f"{index}. {role}: {path}" for index, (role, path) in enumerate(paths, 1)]
    boundary=('锁定身份、姿态、画面几何和既有环境光；允许人物受光、所列闪光及其有限接收面变化；禁止新增大气光效。'
              if lighting_mode=='direct_phone_flash' else '允许主体受光和所列人物投影修订；锁定身份、姿态、画面几何与既有环境光源/原物件投影方向，禁止新增大气光效。')
    sections += ["场景观察与推断：", *observations, hypothesis_text,boundary,*extra_sections]
    for key, label in [("person_observation", "主体原图观察"), ("lighting_gap", "受光差异"),
                       ("unknowns", "未知，不得当作事实"), ("exclusions", "排除与禁止")]:
        sections += [label + "：", *["- " + s for s in texts(analysis, key)]]
    sections += ["主体调整计划：", *plan, "验收条件：", *["- " + s for s in texts(contract, "acceptance")],
                 frozen_text]
    compiled = {"schema_version": version, "contract": contract,
                "provenance": {"compiler": "stdlib_offline_script", "model_called": False,
                               "contract_path": str(contract_path.resolve()), "contract_sha256": hashlib.sha256(raw).hexdigest(),
                               "reference_order": [{"index": i, "role": role, "path": str(p), "sha256": sha256(p)}
                                                   for i, (role, p) in enumerate(paths, 1)],
                               "grade_report_sha256": report_hash}}
    if parametric:
        compiled["provenance"]["verified_transform_type"] = "parametric_curve_chroma"
        compiled["provenance"]["verified_parametric_parameters"] = parameters
    else:
        compiled["provenance"]["verified_lut_sha256"] = lut_hash
    if version >= 3:
        compiled["derived_light_transport"] = derived
    contents = ["\n\n".join(sections) + "\n", json.dumps(compiled, ensure_ascii=False, indent=2) + "\n"]
    out_dir.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for path, content in zip(outputs, contents):
            with path.open("x", encoding="utf-8") as stream:
                created.append(path)
                stream.write(content)
    except OSError:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return compiled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        compile_contract(args.contract, args.out_dir)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Contract compilation failed: {exc}\n")
    print(json.dumps({"output_directory": str(args.out_dir.resolve()), "model_called": False}))


if __name__ == "__main__":
    main()
