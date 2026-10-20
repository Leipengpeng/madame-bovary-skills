# 文件结构与命令：能跑的才叫流程

所有命令从Skill目录运行；输入及输出参数使用当前系统的绝对路径。下文 `/workspace/...` 是中性占位，Windows请改成自己的绝对路径。新建输出文件/目录，避免覆盖。模板中的场景判断和坐标是示例，须先看实际图片再重写。

这里有数种独立JSON类型，各自拥有 `schema_version`，数字相同不代表可以互换。脚本只校验声明、引用和条件计算，不识图、不推断真实物理参数、不调用图像模型。

## 直接生成简报

默认入口为 `scripts/compile_scene_brief.py`，类型为 `schema_version:1`、`mode:"direct_scene_generation"`。完整可填模板是 [../examples/direct_scene_brief.template.json](../examples/direct_scene_brief.template.json)。

```text
python scripts/compile_scene_brief.py --brief /workspace/run/scene-brief.json --out-dir /workspace/run/compiled-new
```

输出 `image_prompt.txt` 与 `compiled_brief.json`。核对后将提示词和输出清单中的图片，按顺序传给内建图像工具；编译成功本身还没有出图。

| 顶层字段 | 约束与作用 |
|---|---|
| `inputs` | `environment`，以及单张 `identity_reference` 或多张 `identity_references`；两者互斥。可选 `outfit_reference`。所有文件存在且角色文件不重复 |
| `identity` | 非空 `preserve`、`ignore_source_conditions` 字符串列表。单图默认 `reference_mode:"single_reference"`；多图显式 `reference_mode:"multi_reference"`，列表至少两张同一人照片 |
| `composition` | `environment_fov_locked:true`；非空 `anchors` 列表；`placement`、`pose`、`outfit`、`occlusion_and_crop` 描述 |
| `analysis` | 非空 `scene_evidence` 列表，各项 `id` 唯一、`source_domain:"real_scene"`、`region`、`observation`、`inference`、`confidence`；另有 `unknowns` 字符串列表 |
| `environment_lighting` | 非空 `sources`，形状见下文“通用灯光字段” |
| `lighting_mode` | `ambient_shadow_match` 或 `direct_phone_flash`，不能任意拼接名称 |
| `synthetic_lights`、`background_light_exceptions`、`shadow_tracking` | 形状及模式约束见下文 |
| `flash_geometry` | 无闪光 `{ "enabled":false }`；直闪则为下文距离计划对象 |
| `flash_detail_plan` | 直闪必需；包含深度层、投影关联、反光面与面部短阴影 |
| `postprocess` | `mode:"none"` 为默认；用户明确指定已有冻结追色时可写 `mode:"frozen_grade_requested"`。该字段不执行调色，也不自动验证外部追色报告 |
| `acceptance` | 非空验收条件字符串列表 |

简报没有独立调用尺度规划器；必须先将尺度/站位结论写入 `composition`，再编译。多图身份参考按列表顺序提交；单参考稳定性测试严格用单图字段，不能借多图稳住结果后仍标单图。

## 通用灯光字段

以下字段同时用于直接生成简报和受光修订合同。

`environment_lighting.sources` 每项：

```json
{
  "id": "L1",
  "geometry_type": "area",
  "transport": "diffuse_bounce",
  "role": "fill",
  "direction": {"image_space": "图像右侧宽范围填充", "depth": "前后位置未知"},
  "color": {"tendency": "相对偏暖", "basis": "依据真实场景观察E1，材质和白平衡仍有混淆"},
  "apparent_size": "反射区域较宽，实际角尺寸未测",
  "confidence": "low",
  "evidence_ids": ["E1"]
}
```

`geometry_type` 为 `point_like / area / directional / unknown`；`transport` 为 `direct / diffuse_bounce / mixed / unknown`；`role` 为 `key / fill / ambient / unknown`。证据ID必须非空且引用实际 `scene_evidence`；`confidence` 统一为 `low / medium / high`。字段只是上游分析，不认证光源真值。

直闪模式要求恰好一项 `synthetic_lights`：`id`、`kind:"phone_flash"`、`authority:"user_requested_synthetic"`、`position`、`size`、`relative_strength`、`falloff`、`color_response`。新增光与环境光ID不能碰撞。至少一项 `background_light_exceptions` 指明新增光ID、真实接收区域 `region`、归一化 `box` 与 `allowed_change`；框是限定范围，不是矩形光束。

环境光模式不添加人造灯，`synthetic_lights` 和 `background_light_exceptions` 均为空列表。

`shadow_tracking` 形状为 `enabled:true`、`unknowns` 字符串列表，以及非空 `tracks`。每条含：

| 字段 | 内容 |
|---|---|
| `id`、`light_id` | 阴影链唯一ID；光源引用已声明环境/新增光，或 `ambient:E1` 形式的真实证据 |
| `occluder` | 实际遮挡物或人物部位 |
| `receiver` | `{ "region":"实际接收面", "source_domain":"real_scene" }`；人物皮肤/衣物等使用 `person_surface` |
| `direction`、`softness`、`attachment` | 方向、软硬条件、与接触/遮挡的贴合关系 |
| `visibility` | `visible / mostly_occluded / off_frame / uncertain` |
| `confidence`、`evidence_ids` | 观察把握；证据可为空表示创作假设，非空须能解析 |

直闪至少有一条链引用新增闪光，即使其投影主要隐藏也应记录原因。

## 闪光细节计划

`flash_detail_plan` 结构如下；详见 [lighting-and-materials.md](lighting-and-materials.md)。

| 字段 | 每项要求 |
|---|---|
| `depth_layers` | 非空；`id`唯一，`layer`为 `foreground / subject / background`，以及 `region`、`relative_distance`、`flash_response`、`confidence`、`evidence_ids` |
| `shadow_links` | 非空；`track_id` 引用已存在阴影链且不可重复，`occluder_layer_id`、`receiver_layer_id` 引用深度层，`visibility_reason` 解释可见性 |
| `specular_surfaces` | `id`唯一；`region`、`material`、`box`、`orientation`、`reflection_path`、`highlight_action`、`light_id`、`confidence`、`evidence_ids`；`expected_visibility`为 `visible / conditional / not_visible / uncertain` |
| `facial_shadow_checks` | 非空；`region`、`occluder`、`receiver`、`direction`、`action`、`light_id`、`confidence`；`visibility`为 `visible / mostly_occluded / off_frame / uncertain` |

反光物列表可以为空，此时额外给非空 `specular_absence_reason`。所有反光面框须有限、非空且在0–1范围内。灯ID必须已声明，证据引用必须能解析；空证据列表不等于已测量。校验器没有验证材质法线或文字的物理正确性。

## 尺度计划

`scripts/plan_scene_scale.py` 的定性最小完整输入：

```json
{
  "schema_version": 1,
  "mode": "qualitative_only",
  "environment_image": "/workspace/inputs/environment.png",
  "background_locked": true,
  "observations": ["替换为实际可见家具和遮挡关系"],
  "unknowns": ["现场尺寸和相机参数未测"],
  "distance_and_scale_relations": ["依据实际图片描述人物与参照物的相对关系"],
  "placement_guidance": ["依据实际图片描述站位、身体占比和裁切"]
}
```

```text
python scripts/plan_scene_scale.py --input /workspace/run/scale.json --out-dir /workspace/run/scale-plan-new
```

输出 `scale_plan.json` 和 `image_prompt.txt`；始终记录 `ground_contact_verified:false`。数值模式 `bounded_projection` 还要求：

- `projection_assumptions` 中 `shared_vertical_projection`、`subject_upright`、`reference_fully_visible`、`endpoint_depth_variation_negligible`、`depth_ratio_uses_optical_axis` 均为true，并写 `basis_note`、`roll_and_distortion_treatment`。
- `reference` 有 `id`、`object`、`observation_note`、`source_domain:"real_scene"`、`observed_height_fraction` 正区间且不大于1、`physical_height_m` 正区间、`physical_size_basis`、`physical_size_note`。
- `subject` 有 `pose:"standing"`、`height_m` 正区间、`height_basis`、`height_note`、`pose_width_over_full_height` 正区间。
- `depth` 有正区间 `reference_over_subject`（光轴深度 `Z参照/Z人物`）、`basis`、`basis_note`、`relation`。关系为 `subject_in_front / same_depth / subject_behind / unknown`；在前要求比值下限不小于1，在后要求上限不大于1，同深度须为 `[1,1]`。
- `selection` 有在范围内的 `full_height_fraction`、`pose_width_over_height`，`center_x`，`anchor:"head_top"` 或 `"feet"`、`anchor_y`、`rationale`。

来源基准使用 `creative_hypothesis / user_provided / measured`，脚本均不认证真实性。默认不需要为了填表编造尺寸；定性模式能清楚说明构图时就用定性模式。

## 闪光距离计划

`scripts/plan_light_transport.py` 的输入是 `flash_geometry` 对象本身，不是整份简报。默认定性输入：

```json
{
  "enabled": true,
  "estimation": "qualitative_only",
  "basis": "creative_hypothesis",
  "basis_note": "根据画面选择的距离次序，非测距",
  "receiver": "实际存在的近处桌面及实心背景",
  "distance_relations": ["近处可达桌面与人物有闪光贡献，远景保持环境光"]
}
```

```text
python scripts/plan_light_transport.py --input /workspace/run/flash-geometry.json --output /workspace/run/flash-estimate-new.json
```

定性模式不得附数字距离。数值模式用 `estimation:"bounded_geometry"`、`arrangement:"collinear_parallel_planes"`、`irradiance_model:"point_like_direct"`，并给米制区间 `source_to_subject_m`、`subject_to_receiver_m`、`emitter_diameter_m`；距离须有限且依序，人物到接收面允许0，其余严格正值。还需相同的 `basis`、`basis_note`、`receiver`。模式条件见 [lighting-and-materials.md](lighting-and-materials.md#距离只在有依据时算数)。

## 可选受光修订合同

`scripts/compile_lighting_contract.py` 用于对已有未调色成图作受光修订，且**要求已有同场景冻结追色报告**。它不是默认直接生成入口。

```text
python scripts/compile_lighting_contract.py --contract /workspace/run/lighting-contract.json --out-dir /workspace/run/lighting-compiled-new
```

输入 `mode:"subject_lighting_edit"`；编译器兼容 `schema_version` 1–4，新直闪细节任务用4：

| 版本 | 字段层次 |
|---|---|
| 1 | 公共输入、场景证据、工作假设、锁定条件、人物操作计划、验收和冻结LUT |
| 2 | 增加 `lighting_mode`、新增光、背景受光例外与阴影链 |
| 3 | 增加结构化 `environment_lighting`、`flash_geometry`；每个主体操作绑定 `source_ids`、`material_response`、`shadow_response` |
| 4 | 增加闪光细节计划；支持核对精确参数曲线/色度变换的冻结分支 |

公共字段要求：

- `inputs.edit_target`、`inputs.environment` 及非空 `inputs.identity_references`，均为存在的绝对文件路径。这里顺序为 **编辑目标 → 原环境 → 身份参考**，与直接生成简报的顺序不同。
- `analysis.scene_evidence` 的结构同直接简报；另含 `person_observation`、`lighting_gap`、`unknowns`、`exclusions` 字符串列表。
- `selected_hypothesis` 含 `id`、`statement`、`uncertainty`、`confidence`、`evidence_ids`。这是待验证的工作假设。
- `constraints`：`geometry_locked`、`identity_locked`、`pose_locked`、`background_light_locked`、`subject_light_edit_allowed` 均为true，`atmospheric_effects_allowed:false`。尺度和站位应在进入本分支前确定。
- 非空 `subject_plan`：每项 `region`、`prompt`、`hypothesis_based` 布尔值、`evidence_ids`；版本3/4再含非空 `source_ids` 及 `material_response`、`shadow_response`。
- `base_prompt` 字符串、`acceptance` 字符串列表。
- `frozen_grade`：`refit:false`、`report`、`scene_id`。LUT分支使用 `transform_type:"lab_lut"` 与 `lut_sha256`；精确参数分支仅版本4支持，使用 `transform_type:"parametric_curve_chroma"` 和 `report_sha256`。

报告必须唯一解析该场景且哈希匹配。模板或别的场景报告不能凑数。编译输出 `image_prompt.txt`、`compiled_contract.json`；它只验证并组织提示，既不编辑图片，也不应用颜色变换。

## 审图记录

`scripts/score_fusion_review.py` 输入 `schema_version:1`，`reference` 为原环境绝对路径，`candidate` 为实际交付图绝对路径；两者必须存在。`checks` 必须且仅包含五项：`identity`、`spatial_contact`、`lighting_shadow`、`texture_edges`、`background_preservation`。

每项为 `{ "status":"observed", "rating":3, "confidence":"medium", "evidence":"实际观察位置和证据" }`。`uncertain`、`not_applicable` 不得包含rating。非空证据与把握是必需的。`hard_flags` 是列表，每项含非空 `code`、`evidence`，可附 `confidence`；无硬问题时为空列表。

```text
python scripts/score_fusion_review.py --review /workspace/run/review.json --output /workspace/run/score-new.json
```

可附 `identity_references`、审图方式、光照目标实现状态和限制，评分脚本不会把它们变成新维度。评分定义、权重和结束条件见 [review-and-stability.md](review-and-stability.md)。

## 颜色配置与冻结区域

追色配置、命令及一次冻结的完整要求见 [color-workflow.md](color-workflow.md)。初次配置保存为新基线目录内的 `config.json`，与脚本输出的 `grading_report.json` 相邻；冻结脚本固定读取此文件并核对原输入/参照路径。配置不会由追色脚本自动复制。参数曲线/色度分支执行冻结时必须带 `--report-sha256`，值取选定基线或冻结合同已记录的原报告哈希；LUT分支也可带此检查。冻结输出为新PNG，父目录必须已存在。`candidate-regions.json` 形状为：

```json
{
  "exclude_boxes": [[0.4, 0.2, 0.9, 1.0]],
  "skin_boxes": [[0.55, 0.3, 0.7, 0.45]],
  "audit_regions": {"stable_wall": [0.05, 0.05, 0.3, 0.2]}
}
```

这是格式示例，不是实际标注。三个容器均非空，框有限、非空、处于0–1内；区域名字可按实际场景取。检查区语义是否正确仍须看图，合法坐标不会自动变成合法皮肤。
