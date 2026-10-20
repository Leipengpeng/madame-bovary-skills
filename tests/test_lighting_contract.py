"""Small offline contract tests; fixtures never contain user images or models."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1]/"scripts"/"compile_lighting_contract.py"
SPEC = importlib.util.spec_from_file_location("lighting_compiler_under_test", SCRIPT)
compiler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compiler)


class LightingContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lighting-contract-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("target.png", "environment.png", "identity.png"):
            (self.root/name).write_bytes(name.encode())
        lut = self.root/"frozen.npz"
        lut.write_bytes(b"temporary LUT fixture; hashing only, no NumPy needed")
        self.lut_hash = hashlib.sha256(lut.read_bytes()).hexdigest()
        report = self.root/"grade.json"
        report.write_text(json.dumps({"scenes": [{"id": "03", "selected": {
            "lut_file": str(lut), "lut_sha256": self.lut_hash}}]}), encoding="utf-8")
        self.contract = {
            "schema_version": 1, "mode": "subject_lighting_edit",
            "inputs": {"edit_target": str(self.root/"target.png"),
                       "environment": str(self.root/"environment.png"),
                       "identity_references": [str(self.root/"identity.png")]},
            "analysis": {"scene_evidence": [{"id": "E1", "region": "展厅真实墙面",
                         "observation": "左侧墙面有较亮的反射", "inference": "可能存在左侧漫射补光",
                         "confidence": "medium", "source_domain": "real_scene"}],
                         "person_observation": ["面部右侧偏亮"], "lighting_gap": ["需要协调受光方向"],
                         "unknowns": ["灯具不在画面内"], "exclusions": ["画作中的太阳不能证明展厅主光"]},
            "selected_hypothesis": {"id": "H1", "statement": "主光可能来自左上方",
                                    "evidence_ids": ["E1"], "confidence": "low",
                                    "uncertainty": "仅凭墙面反射不能确定灯具位置"},
            "constraints": {"geometry_locked": True, "identity_locked": True, "pose_locked": True,
                            "background_light_locked": True, "subject_light_edit_allowed": True,
                            "atmospheric_effects_allowed": False},
            "subject_plan": [{"region": "面部", "prompt": "轻调面部受光以匹配环境",
                              "evidence_ids": ["E1"], "hypothesis_based": True}],
            "acceptance": ["背景、姿态与身份保持"],
            "frozen_grade": {"report": str(report), "scene_id": "03",
                             "lut_sha256": self.lut_hash, "refit": False},
            "base_prompt": "只调整主体受光，不引入额外的美颜灯。",
        }

    def compile(self, contract, output="compiled"):
        path = self.root/"contract.json"
        path.write_text(json.dumps(contract, ensure_ascii=False), encoding="utf-8")
        return compiler.compile_contract(path, self.root/output)

    def v2_contract(self, mode):
        contract = copy.deepcopy(self.contract)
        contract.update({"schema_version": 2, "lighting_mode": mode,
                         "synthetic_lights": [], "background_light_exceptions": [],
                         "base_prompt": "保持身份、姿态与几何；按明确列出的光源修订受光。"})
        track = {"id": "S1", "light_id": "ambient:E1", "occluder": "人物头肩",
                 "receiver": {"region": "人物后方真实墙面", "source_domain": "real_scene"},
                 "direction": "沿工作假设主光经过头肩的方向落向后方墙面",
                 "softness": "随漫射环境光保留柔和半影", "visibility": "mostly_occluded",
                 "attachment": "投影起点与遮挡者的投影位置相连，不悬浮",
                 "confidence": "medium", "evidence_ids": ["E1"]}
        if mode == "direct_phone_flash":
            contract["synthetic_lights"] = [{
                "id": "F1", "kind": "phone_flash", "authority": "user_requested_synthetic",
                "position": "镜头附近的近轴手机闪光", "size": "小面积灯面",
                "relative_strength": "固定环境曝光下增加中等强度前景照明",
                "falloff": "近处更亮，随距离增加衰减", "color_response": "中性闪光叠加现有环境色"}]
            contract["background_light_exceptions"] = [{
                "light_id": "F1", "region": "人物身后真实墙面接收区", "box": [.3, .25, .8, .9],
                "allowed_change": "仅允许新增闪光及其投影改变此区受光，保留已有环境光"}]
            track.update({"light_id": "F1", "direction": "近轴投影位于人物背后，远离镜头",
                          "softness": "小光源投影较硬，随物面间距形成有限半影"})
            contract["subject_plan"][0]["prompt"] = "新增已授权的近轴手机闪光，使主体获得正向照明贡献。"
        contract["shadow_tracking"] = {"enabled": True, "tracks": [track],
                                       "unknowns": ["仅为合同关系验证，未由脚本测量真实三维距离"]}
        return contract

    def v3_contract(self, mode):
        contract = self.v2_contract(mode)
        contract["schema_version"] = 3
        contract["environment_lighting"] = {"sources": [{
            "id": "L1", "geometry_type": "area", "transport": "diffuse_bounce", "role": "fill",
            "direction": {"image_space": "画面左侧", "depth": "斜向镜头，深度未知"},
            "color": {"tendency": "轻微暖色", "basis": "真实墙面可见反射颜色，非标定色温"},
            "apparent_size": "反射面较宽，实际光源尺寸未知", "evidence_ids": ["E1"], "confidence": "medium"}]}
        contract["subject_plan"][0].update({
            "source_ids": ["L1"], "material_response": "皮肤保留微弱暖漫反射，眼镜仅有材质高光",
            "shadow_response": "鼻下和衣领阴影匹配这组漫射光，不抹掉遮蔽关系"})
        contract["flash_geometry"] = {"enabled": False}
        if mode == "direct_phone_flash":
            contract["subject_plan"][0]["source_ids"].append("F1")
            contract["flash_geometry"] = {
                "enabled": True, "estimation": "bounded_geometry", "basis": "creative_hypothesis",
                "basis_note": "为创作试验主动设定的距离区间，不是照片测距。",
                "arrangement": "collinear_parallel_planes", "receiver": "人物后方真实墙面",
                "source_to_subject_m": [2, 3], "subject_to_receiver_m": [0, 2],
                "emitter_diameter_m": [.004, .008]}
        else:
            # v3 may bind the chain directly to its explicit environmental source model.
            contract["shadow_tracking"]["tracks"][0]["light_id"] = "L1"
        return contract

    def v4_contract(self, mode="direct_phone_flash"):
        contract = self.v3_contract(mode)
        contract["schema_version"] = 4
        if mode != "direct_phone_flash":
            return contract
        contract["flash_detail_plan"] = {
            "depth_layers": [
                {"id": "person", "region": "前景人物", "layer": "subject",
                 "relative_distance": "近于后墙，实际灯距未知",
                 "flash_response": "新增直接照明，保留环境光和遮蔽",
                 "confidence": "medium", "evidence_ids": []},
                {"id": "wall", "region": "人物后方真实墙面", "layer": "background",
                 "relative_distance": "位于人物之后，间距为创作假设",
                 "flash_response": "受光弱于前景，遮挡区只保留环境光",
                 "confidence": "low", "evidence_ids": ["E1"]},
            ],
            "shadow_links": [{"track_id": "S1", "occluder_layer_id": "person",
                              "receiver_layer_id": "wall",
                              "visibility_reason": "近轴投影大多被头肩遮住，不能任意画外移黑边"}],
            "specular_surfaces": [
                {"id": "nose_ring", "region": "鼻部金属饰品", "material": "小面积抛光金属",
                 "box": [.45, .3, .48, .34], "orientation": "弯曲表面局部朝向未知",
                 "reflection_path": "仅局部法线允许闪光反射回镜头时出现小亮点",
                 "expected_visibility": "conditional", "highlight_action": "允许微小高光，不画星芒",
                 "light_id": "F1", "confidence": "low", "evidence_ids": []},
                {"id": "table_glass", "region": "桌面玻璃杯", "material": "透明弯曲玻璃",
                 "box": [.15, .7, .25, .9], "orientation": "杯壁曲率可见但法线未测量",
                 "reflection_path": "闪光经杯壁适当朝向的局部表面返回相机",
                 "expected_visibility": "uncertain", "highlight_action": "保留透明度，只在支持方向处形成小反射",
                 "light_id": "F1", "confidence": "low", "evidence_ids": ["E1"]},
            ],
            "facial_shadow_checks": [
                {"region": "鼻底与人中", "occluder": "鼻尖及鼻翼", "receiver": "上唇及人中皮肤",
                 "direction": "随稍高于镜头的近轴光落向下方，不能改成强侧光",
                 "visibility": "visible", "action": "保留短而有限的鼻底阴影和鼻翼层次",
                 "light_id": "F1", "confidence": "medium"},
                {"region": "下巴与颈部", "occluder": "下巴", "receiver": "上颈部皮肤",
                 "direction": "短投影沿下巴下方延续", "visibility": "mostly_occluded",
                 "action": "保留遮蔽并随姿态弱化，不绘制漂浮边缘",
                 "light_id": "F1", "confidence": "medium"},
            ],
        }
        return contract

    def parametric_contract(self, color_mode="shared_scale_offset"):
        contract = self.v4_contract("ambient_shadow_match")
        report_path = self.root/"parametric-grade.json"
        report_path.write_text(json.dumps({"scenes": [{"id": "05",
            "brightness_curve_Lstar": {"input": [0, 50, 100], "target": [0, 51, 100]},
            "selected": {"status": "selected_by_heldout_metrics_pending_visual_review",
                         "rejected": [], "color_mode": color_mode, "strength": .5,
                         "chroma_scale_offset": [1 if color_mode == "offset_only" else 1.05, .5, -.5]}}]}),
                               encoding="utf-8")
        contract["frozen_grade"] = {"report": str(report_path), "scene_id": "05", "refit": False,
            "transform_type": "parametric_curve_chroma", "report_sha256": compiler.sha256(report_path)}
        return contract

    def test_valid_contract_preserves_hypothesis_and_provenance(self):
        compiled = self.compile(self.contract)
        prompt = (self.root/"compiled"/"image_prompt.txt").read_text(encoding="utf-8")
        self.assertIn(self.contract["selected_hypothesis"]["statement"], prompt)
        self.assertIn("工作假设", prompt)
        self.assertIn("尚未证实", prompt)
        self.assertIn(self.contract["selected_hypothesis"]["uncertainty"], prompt)
        self.assertIn(self.contract["analysis"]["scene_evidence"][0]["inference"], prompt)
        provenance = compiled["provenance"]
        self.assertIs(provenance["model_called"], False)
        self.assertEqual([r["role"] for r in provenance["reference_order"]],
                         ["edit_target", "environment", "identity_reference"])
        self.assertEqual(provenance["verified_lut_sha256"], self.lut_hash)
        for reference in provenance["reference_order"]:
            self.assertEqual(reference["sha256"],
                             hashlib.sha256(Path(reference["path"]).read_bytes()).hexdigest())
        self.assertEqual(provenance["contract_sha256"],
                         hashlib.sha256((self.root/"contract.json").read_bytes()).hexdigest())
        saved = json.loads((self.root/"compiled"/"compiled_contract.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, compiled)
        before = (self.root/"compiled"/"image_prompt.txt").read_bytes()
        with self.assertRaises(ValueError):
            self.compile(self.contract)
        self.assertEqual((self.root/"compiled"/"image_prompt.txt").read_bytes(), before)
        cli = subprocess.run([sys.executable, "-B", str(SCRIPT), "--contract",
                              str(self.root/"contract.json"), "--out-dir", str(self.root/"cli")],
                             capture_output=True, text=True, check=True)
        self.assertIs(json.loads(cli.stdout)["model_called"], False)
        self.assertTrue((self.root/"cli"/"image_prompt.txt").is_file())

    def test_missing_or_depicted_evidence_is_rejected(self):
        for failure in ("missing", "domain", "region"):
            with self.subTest(failure=failure):
                contract = copy.deepcopy(self.contract)
                if failure == "missing":
                    contract["subject_plan"][0]["evidence_ids"] = ["missing-evidence"]
                elif failure == "domain":
                    contract["analysis"]["scene_evidence"][0]["source_domain"] = "painting_content"
                else:
                    contract["analysis"]["scene_evidence"][0]["region"] = "画内太阳"
                with self.assertRaises(ValueError):
                    self.compile(contract, "invalid-" + failure)
                self.assertFalse((self.root/("invalid-" + failure)).exists())

    def test_unlocked_boundaries_refit_or_changed_lut_are_rejected(self):
        for key in ("background_light_locked", "identity_locked", "pose_locked", "geometry_locked",
                    "refit", "lut_hash", "lut_bytes"):
            with self.subTest(field=key):
                contract = copy.deepcopy(self.contract)
                if key == "refit":
                    contract["frozen_grade"][key] = True
                elif key == "lut_hash":
                    contract["frozen_grade"]["lut_sha256"] = "0" * 64
                elif key == "lut_bytes":
                    (self.root/"frozen.npz").write_bytes(b"changed after selection")
                else:
                    contract["constraints"][key] = False
                with self.assertRaises(ValueError):
                    self.compile(contract, "invalid-" + key)

    def test_v2_ambient_and_phone_flash_preserve_shadow_and_added_light_plan(self):
        for mode in ("ambient_shadow_match", "direct_phone_flash"):
            with self.subTest(mode=mode):
                contract = self.v2_contract(mode)
                output = "valid-v2-" + mode
                compiled = self.compile(contract, output)
                prompt = (self.root/output/"image_prompt.txt").read_text(encoding="utf-8")
                self.assertEqual(compiled["schema_version"], 2)
                self.assertEqual(compiled["contract"], contract)
                self.assertIn(mode, prompt)
                self.assertIn(contract["subject_plan"][0]["prompt"], prompt)
                track = contract["shadow_tracking"]["tracks"][0]
                for key in ("light_id", "occluder", "direction", "softness", "attachment"):
                    self.assertIn(track[key], prompt)
                self.assertIn(track["receiver"]["region"], prompt)
                if mode == "direct_phone_flash":
                    light = contract["synthetic_lights"][0]
                    for key in ("id", "position", "size", "relative_strength", "falloff", "color_response"):
                        self.assertIn(light[key], prompt)
                    self.assertIn(contract["background_light_exceptions"][0]["allowed_change"], prompt)
                    self.assertIn("不是原图实测光源", prompt)
                else:
                    self.assertEqual(compiled["contract"]["synthetic_lights"], [])
                    self.assertEqual(compiled["contract"]["background_light_exceptions"], [])

    def test_v2_rejects_unknown_light_sources_and_depicted_receivers(self):
        for failure in ("unknown_ambient", "unknown_flash", "unknown_exception", "depicted_receiver"):
            with self.subTest(failure=failure):
                mode = "ambient_shadow_match" if failure == "unknown_ambient" else "direct_phone_flash"
                contract = self.v2_contract(mode)
                track = contract["shadow_tracking"]["tracks"][0]
                if failure == "unknown_ambient":
                    track["light_id"] = "ambient:unobserved-source"
                elif failure == "unknown_flash":
                    track["light_id"] = "undeclared-flash"
                elif failure == "unknown_exception":
                    contract["background_light_exceptions"][0]["light_id"] = "undeclared-flash"
                else:
                    track["receiver"]["source_domain"] = "painting_content"
                with self.assertRaises(ValueError):
                    self.compile(contract, "invalid-v2-" + failure)
                self.assertFalse((self.root/("invalid-v2-" + failure)).exists())

    def test_v2_flash_needs_receiver_exception_and_ambient_forbids_added_light(self):
        for failure in ("no_receiver_exception", "flash_disguised_as_ambient"):
            with self.subTest(failure=failure):
                contract = self.v2_contract("direct_phone_flash")
                if failure == "no_receiver_exception":
                    contract["background_light_exceptions"] = []
                else:
                    # This was an otherwise valid flash contract: only mode changes.
                    contract["lighting_mode"] = "ambient_shadow_match"
                with self.assertRaises(ValueError):
                    self.compile(contract, "invalid-v2-" + failure)
                self.assertFalse((self.root/("invalid-v2-" + failure)).exists())

    def test_v3_compiles_all_light_fields_region_mapping_and_declared_geometry(self):
        for mode in ("ambient_shadow_match", "direct_phone_flash"):
            with self.subTest(mode=mode):
                contract = self.v3_contract(mode)
                output = "valid-v3-"+mode
                compiled = self.compile(contract, output)
                self.assertEqual(compiled["schema_version"], 3)
                self.assertEqual(compiled["contract"], contract)
                prompt = (self.root/output/"image_prompt.txt").read_text(encoding="utf-8")
                light = contract["environment_lighting"]["sources"][0]
                for value in (light["id"], light["geometry_type"], light["transport"], light["role"],
                              *light["direction"].values(), *light["color"].values(), light["apparent_size"],
                              light["confidence"], *light["evidence_ids"]):
                    self.assertIn(value, prompt)
                plan = contract["subject_plan"][0]
                for key in ("material_response", "shadow_response"):
                    self.assertIn(plan[key], prompt)
                for source_id in plan["source_ids"]:
                    self.assertIn(source_id, prompt)
                derived = compiled["derived_light_transport"]
                self.assertIs(derived["calculation_performed"], mode == "direct_phone_flash")
                if mode == "direct_phone_flash":
                    self.assertEqual(derived["source_to_receiver_m"], [2, 5])
                    self.assertEqual(derived["flash_irradiance_ratio_bg_to_subject"], [.25, 1])
                    self.assertEqual(derived["shadow_linear_magnification"], [1, 2])
                    self.assertEqual(derived["penumbra_width_m"], [0, .008])
                    self.assertIs(derived["basis_verified"], False)
                    self.assertIn(contract["flash_geometry"]["basis_note"], prompt)
                    for key in ("source_to_subject_m", "subject_to_receiver_m", "emitter_diameter_m",
                                "source_to_receiver_m", "flash_irradiance_ratio_bg_to_subject",
                                "shadow_linear_magnification", "penumbra_width_m"):
                        self.assertIn(key, prompt)
                    for condition in derived["assumptions"]+derived["limitations"]:
                        self.assertIn(condition, prompt)
                else:
                    self.assertNotIn("source_to_receiver_m", derived)

    def test_v3_rejects_unresolved_sources_empty_evidence_and_incompatible_geometry(self):
        failures = ("plan_source", "empty_plan_sources", "source_evidence", "empty_source_evidence",
                    "duplicate_source", "source_collision", "empty_sources", "geometry_type", "transport",
                    "material_response", "ambient_geometry_enabled", "flash_geometry_disabled")
        for failure in failures:
            with self.subTest(failure=failure):
                contract = self.v3_contract("direct_phone_flash" if failure == "flash_geometry_disabled" else "ambient_shadow_match")
                source = contract["environment_lighting"]["sources"][0]
                if failure == "plan_source":
                    contract["subject_plan"][0]["source_ids"] = ["ambient:E1"]
                elif failure == "empty_plan_sources":
                    contract["subject_plan"][0]["source_ids"] = []
                elif failure == "source_evidence":
                    source["evidence_ids"] = ["unobserved"]
                elif failure == "empty_source_evidence":
                    source["evidence_ids"] = []
                elif failure == "duplicate_source":
                    contract["environment_lighting"]["sources"].append(copy.deepcopy(source))
                elif failure == "source_collision":
                    contract = self.v3_contract("direct_phone_flash")
                    contract["environment_lighting"]["sources"][0]["id"] = "F1"
                elif failure == "empty_sources":
                    contract["environment_lighting"]["sources"] = []
                elif failure in ("geometry_type", "transport"):
                    source[failure] = "diffuse_bounce" if failure == "geometry_type" else "area"
                elif failure == "material_response":
                    contract["subject_plan"][0]["material_response"] = "  "
                elif failure == "ambient_geometry_enabled":
                    contract["flash_geometry"] = self.v3_contract("direct_phone_flash")["flash_geometry"]
                else:
                    contract["flash_geometry"]["enabled"] = False
                with self.assertRaises(ValueError):
                    self.compile(contract, "invalid-v3-"+failure)
                self.assertFalse((self.root/("invalid-v3-"+failure)).exists())

    def test_v3_qualitative_unknown_geometry_is_retained_without_numeric_estimates(self):
        contract = self.v3_contract("direct_phone_flash")
        geometry = {"enabled": True, "estimation": "qualitative_only", "basis": "creative_hypothesis",
                    "basis_note": "灯距和人与墙距未知，不从图中假测。", "receiver": "后方真实墙面",
                    "distance_relations": ["墙面位于主体后方，间距未知", "灯源靠近镜头，灯面尺寸未知"]}
        contract["flash_geometry"] = geometry
        compiled = self.compile(contract)
        derived = compiled["derived_light_transport"]
        self.assertIs(derived["calculation_performed"], False)
        prompt = (self.root/"compiled"/"image_prompt.txt").read_text(encoding="utf-8")
        for text in [geometry["basis_note"], *geometry["distance_relations"]]:
            self.assertIn(text, prompt)
        for key in ("source_to_receiver_m", "flash_irradiance_ratio_bg_to_subject",
                    "shadow_linear_magnification", "penumbra_width_m"):
            self.assertNotIn(key, derived)
            self.assertNotIn(key, prompt)

    def test_v4_compiles_complete_detail_plan_and_preserves_shared_provenance(self):
        contract = self.v4_contract()
        compiled = self.compile(contract)
        prompt = (self.root/"compiled"/"image_prompt.txt").read_text(encoding="utf-8")
        self.assertEqual(compiled["schema_version"], 4)
        self.assertEqual(compiled["contract"], contract)
        self.assertIn(json.dumps(contract["flash_detail_plan"], ensure_ascii=False), prompt)
        for marker in ("不是自动测距", "光源→该材质表面→相机", "不得在所有玻璃",
                       "不添加无关轮廓光或逆光", "直闪也不能把面部抹成平面"):
            self.assertIn(marker, prompt)
        self.assertEqual(compiled["derived_light_transport"]["source_to_receiver_m"], [2, 5])
        self.assertIs(compiled["derived_light_transport"]["basis_verified"], False)
        self.assertEqual(compiled["provenance"]["verified_lut_sha256"], self.lut_hash)
        self.assertIs(compiled["provenance"]["model_called"], False)
        self.assertEqual(compiled["provenance"]["contract_sha256"],
                         hashlib.sha256((self.root/"contract.json").read_bytes()).hexdigest())

    def test_v4_ambient_can_omit_flash_details_but_preserves_v3_validation(self):
        contract = self.v4_contract("ambient_shadow_match")
        compiled = self.compile(contract)
        self.assertNotIn("flash_detail_plan", compiled["contract"])
        self.assertIs(compiled["derived_light_transport"]["calculation_performed"], False)
        prompt = (self.root/"compiled"/"image_prompt.txt").read_text(encoding="utf-8")
        self.assertIn(contract["environment_lighting"]["sources"][0]["apparent_size"], prompt)
        for failure in ("unknown_source", "unlocked", "refit", "changed_lut"):
            with self.subTest(failure=failure):
                invalid = copy.deepcopy(contract)
                if failure == "unknown_source":
                    invalid["subject_plan"][0]["source_ids"] = ["unknown"]
                elif failure == "unlocked":
                    invalid["constraints"]["geometry_locked"] = False
                elif failure == "refit":
                    invalid["frozen_grade"]["refit"] = True
                else:
                    invalid["frozen_grade"]["lut_sha256"] = "0" * 64
                with self.assertRaises(ValueError):
                    self.compile(invalid, "invalid-v4-shared-"+failure)

    def test_v4_flash_rejects_missing_or_empty_detail_groups(self):
        for failure in ("missing_plan", "depth_layers", "shadow_links", "facial_shadow_checks"):
            with self.subTest(failure=failure):
                contract = self.v4_contract()
                if failure == "missing_plan":
                    del contract["flash_detail_plan"]
                else:
                    contract["flash_detail_plan"][failure] = []
                with self.assertRaises(ValueError):
                    self.compile(contract, "invalid-v4-empty-"+failure)
                self.assertFalse((self.root/("invalid-v4-empty-"+failure)).exists())

    def test_v4_rejects_unresolved_references_and_duplicate_inventory_ids(self):
        failures = ("depth_evidence", "surface_evidence", "track", "occluder_layer", "receiver_layer",
                    "surface_light", "face_light", "duplicate_layer", "duplicate_surface", "duplicate_track_link")
        for failure in failures:
            with self.subTest(failure=failure):
                contract = self.v4_contract()
                plan = contract["flash_detail_plan"]
                if failure == "depth_evidence":
                    plan["depth_layers"][0]["evidence_ids"] = ["missing"]
                elif failure == "surface_evidence":
                    plan["specular_surfaces"][0]["evidence_ids"] = ["missing"]
                elif failure == "track":
                    plan["shadow_links"][0]["track_id"] = "missing"
                elif failure in ("occluder_layer", "receiver_layer"):
                    plan["shadow_links"][0][failure+"_id"] = "missing"
                elif failure == "surface_light":
                    plan["specular_surfaces"][0]["light_id"] = "ambient:E1"
                elif failure == "face_light":
                    plan["facial_shadow_checks"][0]["light_id"] = "missing"
                else:
                    group = {"duplicate_layer": "depth_layers", "duplicate_surface": "specular_surfaces",
                             "duplicate_track_link": "shadow_links"}[failure]
                    plan[group].append(copy.deepcopy(plan[group][0]))
                with self.assertRaises(ValueError):
                    self.compile(contract, "invalid-v4-ref-"+failure)
                self.assertFalse((self.root/("invalid-v4-ref-"+failure)).exists())

    def test_v4_specular_boxes_must_be_finite_normalized_and_nonempty(self):
        boxes = ([0, 0, float("nan"), 1], [0, 0, 1, float("inf")],
                 [-float("inf"), 0, 1, 1], [0, 0, True, 1],
                 [0, 0, 1.1, 1], [.5, 0, .5, 1], [0, 1, 1, 0], [0, 0, 1])
        for index, box in enumerate(boxes):
            with self.subTest(box=box):
                contract = self.v4_contract()
                contract["flash_detail_plan"]["specular_surfaces"][0]["box"] = box
                with self.assertRaises(ValueError):
                    self.compile(contract, "invalid-v4-box-"+str(index))
                self.assertFalse((self.root/("invalid-v4-box-"+str(index))).exists())

    def test_v4_empty_reflective_inventory_requires_explicit_absence_reason(self):
        contract = self.v4_contract()
        contract["flash_detail_plan"]["specular_surfaces"] = []
        for index, reason in enumerate((None, "  ", 0)):
            with self.subTest(reason=reason):
                invalid = copy.deepcopy(contract)
                if reason is not None:
                    invalid["flash_detail_plan"]["specular_absence_reason"] = reason
                with self.assertRaises(ValueError):
                    self.compile(invalid, "invalid-v4-absence-"+str(index))
        reason = "逐区查看后未见需要建模的玻璃或金属，不能为制造闪光感凭空添加。"
        contract["flash_detail_plan"]["specular_absence_reason"] = reason
        self.compile(contract)
        self.assertIn(reason, (self.root/"compiled"/"image_prompt.txt").read_text(encoding="utf-8"))

    def test_v4_rejects_empty_instruction_strings_and_invalid_detail_enums(self):
        fields = {
            "depth_layers": ("id", "region", "relative_distance", "flash_response"),
            "shadow_links": ("track_id", "occluder_layer_id", "receiver_layer_id", "visibility_reason"),
            "specular_surfaces": ("id", "region", "material", "orientation", "reflection_path", "highlight_action"),
            "facial_shadow_checks": ("region", "occluder", "receiver", "direction", "action"),
        }
        for group, keys in fields.items():
            for key in keys:
                with self.subTest(group=group, key=key):
                    contract = self.v4_contract()
                    contract["flash_detail_plan"][group][0][key] = "  "
                    with self.assertRaises(ValueError):
                        self.compile(contract, "invalid-v4-text-"+group+"-"+key)
        for group, key in (("depth_layers", "layer"), ("depth_layers", "confidence"),
                           ("specular_surfaces", "expected_visibility"), ("specular_surfaces", "confidence"),
                           ("facial_shadow_checks", "visibility"), ("facial_shadow_checks", "confidence")):
            with self.subTest(group=group, key=key):
                contract = self.v4_contract()
                contract["flash_detail_plan"][group][0][key] = "unsupported"
                with self.assertRaises(ValueError):
                    self.compile(contract, "invalid-v4-enum-"+group+"-"+key)

    def test_v4_parametric_grade_pins_report_and_compiles_exact_parameters(self):
        for mode in ("offset_only", "shared_scale_offset"):
            with self.subTest(mode=mode):
                contract = self.parametric_contract(mode)
                compiled = self.compile(contract, "parametric-"+mode)
                prompt = (self.root/("parametric-"+mode)/"image_prompt.txt").read_text(encoding="utf-8")
                self.assertIn("已选参数曲线/色度变换", prompt)
                self.assertIn("apply_transform + to_rgb", prompt)
                self.assertNotIn("已选 LUT", prompt)
                self.assertIn(contract["frozen_grade"]["report_sha256"], prompt)
                parameters = compiled["provenance"]["verified_parametric_parameters"]
                self.assertIn(json.dumps(parameters, ensure_ascii=False), prompt)
                self.assertEqual(parameters["color_mode"], mode)
                self.assertEqual(compiled["provenance"]["verified_transform_type"], "parametric_curve_chroma")
                self.assertNotIn("verified_lut_sha256", compiled["provenance"])

    def test_parametric_contract_rejects_unpinned_report_old_schema_and_invalid_curve(self):
        for failure in ("hash", "changed_report", "v3", "curve", "rejected"):
            with self.subTest(failure=failure):
                contract = self.parametric_contract()
                report = Path(contract["frozen_grade"]["report"])
                if failure == "hash":
                    contract["frozen_grade"]["report_sha256"] = "0" * 64
                elif failure == "v3":
                    contract["schema_version"] = 3
                else:
                    document = json.loads(report.read_text(encoding="utf-8"))
                    scene = document["scenes"][0]
                    if failure == "changed_report":
                        scene["selected"]["strength"] = .75
                    elif failure == "curve":
                        scene["brightness_curve_Lstar"]["input"] = [0, 50, 50]
                    else:
                        scene["selected"]["rejected"] = ["skin_change_limit"]
                    report.write_text(json.dumps(document), encoding="utf-8")
                    if failure != "changed_report":
                        contract["frozen_grade"]["report_sha256"] = compiler.sha256(report)
                with self.assertRaises(ValueError):
                    self.compile(contract, "invalid-parametric-"+failure)


if __name__ == "__main__":
    unittest.main()
