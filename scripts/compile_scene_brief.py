"""Compile a direct portrait-in-scene brief; never call an image model.

Default path: one assigned identity photograph plus one environment photograph,
direct generation and no color postprocessing. All observations remain authored
inputs, not measurements inferred by this validator.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "_scene_lighting_validation", Path(__file__).with_name("compile_lighting_contract.py"))
lighting = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lighting)


def compile_brief(brief_path, out_dir):
    brief_path, out_dir = Path(brief_path), Path(out_dir)
    if not brief_path.is_absolute() or not out_dir.is_absolute():
        raise ValueError("Brief and output directory must be absolute paths.")
    outputs = [out_dir/"image_prompt.txt", out_dir/"compiled_brief.json"]
    if any(path.exists() for path in outputs):
        raise ValueError("Refusing to overwrite compiled artifacts.")
    raw = brief_path.read_bytes()
    brief = json.loads(raw)
    field, texts, nonempty = lighting.field, lighting.texts, lighting.nonempty
    if field(brief, "schema_version", int) != 1 or nonempty(brief, "mode") != "direct_scene_generation":
        raise ValueError("Unsupported direct scene brief schema or mode.")
    inputs = field(brief, "inputs", dict)
    if set(inputs) - {"environment", "identity_reference", "identity_references", "outfit_reference"}:
        raise ValueError("Direct mode accepts only environment, one assigned identity reference set, and optional outfit reference.")
    identity = field(brief, "identity", dict)
    reference_mode = identity.get("reference_mode", "single_reference")
    if ("identity_reference" in inputs) == ("identity_references" in inputs):
        raise ValueError("Supply exactly one assigned identity input field: identity_reference OR identity_references.")
    paths = [("environment", lighting.file_path(nonempty(inputs, "environment")))]
    if "identity_reference" in inputs:
        if reference_mode != "single_reference":
            raise ValueError("identity_reference requires single_reference mode.")
        paths.append(("identity_reference", lighting.file_path(nonempty(inputs, "identity_reference"))))
    else:
        references = texts(inputs, "identity_references")
        if reference_mode != "multi_reference" or len(references) < 2:
            raise ValueError("A list requires explicit multi_reference mode and at least two photographs of the same assigned person.")
        paths += [(f"identity_reference_{index}", lighting.file_path(path))
                  for index, path in enumerate(references, 1)]
    if "outfit_reference" in inputs:
        paths.append(("outfit_reference", lighting.file_path(nonempty(inputs, "outfit_reference"))))
    if len({p for _, p in paths}) != len(paths):
        raise ValueError("Environment, assigned identity and outfit inputs must have distinct roles/files.")
    for name in ("preserve", "ignore_source_conditions"):
        if not texts(identity, name):
            raise ValueError("Identity preservation and source-condition exclusions must be explicit.")
    composition = field(brief, "composition", dict)
    if field(composition, "environment_fov_locked", bool) is not True:
        raise ValueError("Direct mode requires the environment camera/FOV to remain locked.")
    for key in ("anchors", "placement", "pose", "outfit", "occlusion_and_crop"):
        if key == "anchors":
            if not texts(composition, key):
                raise ValueError("At least one visible background anchor is required.")
        else:
            nonempty(composition, key)
    analysis = field(brief, "analysis", dict)
    evidence = {}
    for item in field(analysis, "scene_evidence", list):
        eid = nonempty(item, "id")
        forbidden = ("画内", "画中", "画作内部", "painting_content", "artwork_content", "depicted_scene", "inside_painting")
        if (eid in evidence or nonempty(item, "source_domain") != "real_scene"
                or any(tag in nonempty(item, "region").lower() for tag in forbidden)):
            raise ValueError("Unique evidence IDs must describe real scene surfaces.")
        for key in ("region", "observation", "inference"):
            nonempty(item, key)
        lighting.confidence(item)
        evidence[eid] = item
    if not evidence:
        raise ValueError("At least one authored real-scene observation is required.")
    texts(analysis, "unknowns")
    source_ids, environment_sections = lighting.environment_lighting(brief, evidence)
    mode, shadow_sections = lighting.shadow_and_flash(brief, evidence, source_ids)
    synthetic_ids = {source["id"] for source in brief["synthetic_lights"]}
    if source_ids & synthetic_ids:
        raise ValueError("Environmental and added light IDs must not collide.")
    derived, transport_sections = lighting.flash_transport(brief, mode)
    detail_sections = lighting.flash_details(brief, mode, evidence, source_ids | synthetic_ids)
    postprocess = field(brief, "postprocess", dict)
    if nonempty(postprocess, "mode") not in ("none", "frozen_grade_requested"):
        raise ValueError("Postprocess mode must be none or explicitly frozen_grade_requested.")
    acceptance = texts(brief, "acceptance")
    if not acceptance:
        raise ValueError("At least one acceptance criterion is required.")
    light_execution = (
        "Keep the original ambient exposure and white balance. Flash is a local positive "
        "contribution; nearby reachable surfaces react by distance, orientation and occlusion, "
        "while distant sky/city/river remain unchanged. Do not invent an opaque receiving wall "
        "or paint a cast shadow on distant scenery. A near-axis body shadow may be mostly hidden. "
        "Keep attached hand/prop/contact shadows and anatomical nose/lip/chin/glasses shadows. "
        "Preserve eyes through glasses. Specular glints require a plausible source-surface-camera "
        "path; do not stamp identical white dots on every reflective surface. No unrelated rim "
        "light, beauty softbox, starburst, flare or bloom."
        if mode == "direct_phone_flash" else
        "AMBIENT LIGHT ONLY: use the observed environmental light sources to illuminate the "
        "inserted person. Do not add a phone flash or any other synthetic light. Preserve "
        "the original background ambient exposure, white balance and recorded illumination. "
        "Match the declared environmental light directions, color and softness on face, neck, "
        "hands, clothing and props. Keep attached contact shadows and anatomical shading "
        "consistent with those same observed sources; do not impose near-axis flash shadows. "
        "Preserve eyes through glasses and retain only reflections supported by the existing "
        "environment-source/surface/camera path; do not add flash glints, unrelated rim light, "
        "beauty softbox, starburst, flare or bloom."
    )
    sections = [
        "Generate ONE natural smartphone photograph from these assigned references. "
        "Use the environment for canvas, camera/FOV, layout and original ambient exposure; "
        "use the assigned identity reference set only for this ONE photographed person's identity. "
        "Do not borrow a face from earlier outputs or other photographs. An optional outfit "
        "reference supplies clothing/body cues only, never a replacement identity.",
        "REFERENCE ORDER (submit the images in exactly this order):",
        *[f"{index}. {role}: {path}" for index, (role, path) in enumerate(paths, 1)],
        "IDENTITY REFERENCE MODE: " + reference_mode + ". " +
        ("Use only the single assigned photograph; do not fill missing identity details from another reference or prior output."
         if reference_mode == "single_reference" else
         "All assigned identity references depict the same person. Reconcile pose and illumination differences; do not average them into a generic face or import their backgrounds/props."),
        "IDENTITY TO PRESERVE: " + "; ".join(identity["preserve"]),
        "SOURCE CONDITIONS/OBJECTS TO IGNORE: " + "; ".join(identity["ignore_source_conditions"]),
        "COMPOSITION: " + json.dumps(composition, ensure_ascii=False),
        "SCENE OBSERVATIONS / INFERENCES (not physical ground truth): " + json.dumps(analysis, ensure_ascii=False),
        *environment_sections, *shadow_sections, *transport_sections, *detail_sections,
        light_execution,
        "Preserve natural face fullness, skin albedo and texture; no face slimming, whitening "
        "or beauty smoothing. Reproject the pose without preserving source-photo lens distortion "
        "as anatomy. No background zoom/crop/reframe, synthetic grain, HDR or cinematic recoloring. "
        "Do not add text, panels or a watermark.",
        "ACCEPTANCE: " + "; ".join(acceptance),
        "Return only one completed photograph. Any optional postprocessing is a separate tool "
        "operation after generation; do not bake a second color grade into the image.",
    ]
    if postprocess["mode"] == "none":
        sections.append("POSTPROCESS: none. Keep the direct generation for visual review.")
    compiled = {"schema_version": 1, "brief": brief, "derived_light_transport": derived,
                "provenance": {"compiler": "offline_direct_brief", "model_called": False,
                    "brief_sha256": hashlib.sha256(raw).hexdigest(),
                    "reference_order": [{"index": i, "role": role, "path": str(path),
                        "sha256": lighting.sha256(path)} for i, (role, path) in enumerate(paths, 1)]}}
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
    parser.add_argument("--brief", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compile_brief(args.brief, args.out_dir)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Brief compilation failed: {exc}\n")
    print(json.dumps({"output_directory": str(args.out_dir.resolve()), "model_called": False,
                      "postprocess": result["brief"]["postprocess"]["mode"]}))


if __name__ == "__main__":
    main()
