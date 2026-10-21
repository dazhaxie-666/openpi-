# 多物体视觉抓取模块合入 latest

## 主目录与合入范围

主项目：`D:\openpi\latest\openpi--main`。
来源：`D:\openpi\openpi--main\openpi--main\openarm_cloth`。
来源副本于 2026-10-21 完整归档到
`D:\openpi\_archive\cleanup_20261021\imported_working_copy`，含其独有未合入文件。
没有覆盖整个项目，也没有上传 GitHub；详见 [清理清单](../../docs/cleanup_20261021.md)。

合入颜色无关的 RGB-D 几何识别、视觉查询模型、物理抓取与图形选择器。
机器人、相机标定、夹爪执行器、物理步长仍沿用 latest 的 B 环境。
新入口独立，不改变默认视频动作复现和三维回放。

## 直接运行

在 PowerShell 执行：

```powershell
Set-Location -LiteralPath 'D:\openpi\latest\openpi--main'

# 随机布置一个受支持的测试物体，随后由相机识别
& .\run.ps1 -Mode tabletop

# 可选图形选择器；不需要先选物体形状/颜色/尺寸
& .\run.ps1 -Mode tabletop-ui
```

三维窗口可用鼠标转动/缩放。先摆放物体，松开鼠标再按空格，重新拍照并开始抓取。
R 仅在摆放时恢复物体，Q 或关闭窗口退出。青色是真实物理挡边，黄色线是初始
摆放外形边界，不是可达性保证；抓取开始后不裁剪、不托住、不传送物体。
抓取阶段的相机窗口是观察阶段快照，不是全程实时视频。

指定测试布置（参数不传给视觉识别器）：

```powershell
& .\run.ps1 -Mode tabletop -ScriptArgs '--shape','box','--color','green','--size','.070','.040','.055','--fixture-pose','.43','.10','30'

# 只验证识别，不执行抓取
& .\run.ps1 -Mode tabletop -ScriptArgs '--check','--perception-only'

# 黑色圆柱物理测试
& .\run.ps1 -Mode tabletop -ScriptArgs '--shape','cylinder','--color','black','--size','.045','.045','.075','--fixture-pose','.43','-.10','0','--check'

# 固定六例回归：默认新时间目录，也可传 --output 指定新的空目录
& .\run.ps1 -Mode tabletop-check
```

`--size` 为米，圆柱 X/Y 为相等直径，Z 为高度。`--fixture-pose` 是米、米、角度。
报告和诊断图另存 `openarm_cloth/outputs/generic_objects/`；默认时间目录避免覆盖。
已经存在的显式报告或非空 benchmark 目录会被拒绝，不改写历史证据。

原有入口保持：

```powershell
& .\run.ps1 -Mode fusion-replay  # 原视频与机械臂动作回放
& .\run.ps1 -Mode live           # 真人摄像头自由动作跟随
& .\run.ps1 -Mode grasp          # 旧红方块自主抓取
```

## 模块与接口

| 模块 | 职责 |
| --- | --- |
| `tabletop_perception.py` | 深度反投影、前景分离、几何分类、中心/尺寸/朝向、候选身份保持 |
| `tabletop_fixture.py` | 随机测试世界构建参数；识别/IK 不导入采样器 |
| `generic_object_scene.py` | 从冻结 XML 独立生成物体/挡边模型，摆放阶段辅助 |
| `generic_grasp_planner.py` | 仅使用相机测量生成私有查询几何，候选 IK 与采样碰撞检查 |
| `demo_tabletop_objects.py` | 胸部粗定位、腕部精定位、抓前复查、物理闭爪与抬升 |
| `launch_tabletop_grasp.pyw` | 使用当前解释器的图形选择器 |
| `benchmark_tabletop_objects.py` | 六个固定渲染/动力学验证样例 |

共享 `visual_grasp_planner.plan_face_grasp` 新增可选 `closing_axes`；不传时仍使用旧
红方块的两个相对面方向，原调用方式与默认行为保留。
没有用来源版本覆盖 `demo_grasp_viewer.py`、`bimanual_teleop.py`、动作修复模块或执行器。
`start_tabletop_grasp.ps1` 改为调用仓库根入口，不再寻找旧 `venv_member_ab_live`。
新增的六个测试文件自动进入根入口 `-Mode test` 的测试发现。

## 能力边界

- 支持标定水平桌面上彼此分离、竖直、不透明、顶面可见的方块/长方体/圆柱。
- 不依赖颜色阈值，也不需要预先提供物体尺寸；标定平面、相机内外参仍是必需输入。
- 识别可输出多个分离候选，但当前物理演示每次布置一个物体；没有完成多物体杂乱
  场景避障，也不是批量自动分拣。
- 圆柱是几何标签，不表示认识杯子/瓶子。未训练新识别模型或 OpenPI/BC 抓取策略。
- 未验收不规则、堆叠、倾倒、透明/镜面物体、真实深度噪声或实体机器人。
- 固定工况通过不代表任意尺寸/摆位成功或连续路径无碰撞保证。
- A 自由动作映射和 B 物体抓取现已共存于同一项目，但仍是独立模式；本次没有把
  人体自由摆臂变成抓取示范，也没有完成示范重定位、搬运释放和学习闭环。

## 本次验证

专项 42 项全部通过。根入口完整回归共 239 项：236 通过、3 项符号链接权限相关
跳过、0 失败（B 228 项，A 11 项）。旧动作参考重建与跟随回归包含在内。
运行环境检查确认新增模块均来自 latest，不从来源目录或归档导入。
结果目录：`outputs/generic_objects/merged_latest_20261020_validation/`。

| 固定样例 | 检查 | 定位误差约 | 抬升约 | 结果 |
| --- | --- | --- | --- | --- |
| 蓝方块 | 识别＋物理抓取 | 1.96 mm | 70.1 mm | PASS |
| 绿长方体 | 识别＋另一臂物理抓取 | 2.97 mm | 69.9 mm | PASS |
| 黑圆柱 | 识别＋物理抓取 | 2.29 mm | 67.6 mm | PASS |
| 棕方块 | 仅识别 | 0.65 mm | 未测试 | PERCEPTION_ONLY |
| 白长方体 | 仅识别 | 0.48 mm | 未测试 | PERCEPTION_ONLY |
| 黄圆柱 | 仅识别 | 0.54 mm | 未测试 | PERCEPTION_ONLY |

三项物理抓取均满足抬升超过 3 cm 且双指接触持续约 0.8 秒。每个报告保存当前实现的
源文件 SHA256 与诊断图；不是直接沿用来源目录的旧报告。
六个冻结配置字段（workspace、physics、cameras、task_definition、actuators、controller）
均与 `deliverables/member_AB_20261016/scene_manifest.json` 一致。
本轮验证包含离屏渲染与动力学，不把它声称为人工可见窗口交互验收或实体硬件验收。

补充兼容性复测：

- 不传形状/颜色/尺寸，仅执行根入口 `tabletop --check --perception-only`，自动识别
  到长方体并输出约 70.9×40.5×54.1 mm；此项未抓取。
  报告：`outputs/generic_objects/merged_latest_auto_recognition_20261020.json`。
- 原 `grasp` 在 `.43, 0, 30°` 工况重新物理执行 PASS，定位误差约 4.86 mm、
  抬升约 69.9 mm，未启用 weld。
  报告：`outputs/generic_objects/merged_latest_red_cube_20261020.json`。
- 默认 `outputs/a_motion/latest.json` 仍指向 `p2a_test1_repaired_kinematic`，没有被抓取
  测试切换；动作修复源文件未被此次合入覆盖。
- 最终运行环境检查：`outputs/generic_objects/merged_latest_runtime_final_20261020.json`。
