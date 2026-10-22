# OpenArm / Pose2Arm A+B 主项目

更新日期：2026-10-21。日常只使用 `D:\openpi\latest\openpi--main`。
基于上游 `8df4726` 加本地集成改动；代码发布目标为用户仓库，未训练新识别模型。

后续代码发布目标与数据排除范围见 [GitHub 发布说明](docs/github_publish_20261021.md)。
最新默认实时路径包含躯干参考标定、完整相对掌面手腕、个人捏合标定与独立物理线程。
操作、验证和未完成的独立验收见 [动作跟随 v2](openarm_cloth/docs/motion_v2_implementation_20261021_CN.md)。
2026-10-22 接入手部置信度参数，修复手腕位置模式的连续捏合接线与手臂受阻时的夹爪独立更新；见 [手部分支集成](openarm_cloth/docs/direct_hand_branch_integration_20261022_CN.md)。
后续区分完整手臂与手腕位置的观测要求，并明确显示个人标定是否通过；见 [坐站切换与标定反馈](openarm_cloth/docs/standing_observation_fix_20261022_CN.md)。
完整入镜仍漏检时，新增 R 暂停并重新创建视觉跟踪器，和 C 动作重标定分开；恢复后须空格继续，说明与本地诊断流程见上述文档。尚未完成这次真人站姿漏检的复现与效果验收。
骨架整体丢失/间歇落到背景时，可用 `diagnose_human_tracking.py` 单独采集摄像头，并在保存后对同图进行 VIDEO/IMAGE 模式对比，不加载机械臂场景。入口见 [运行说明](RUNNING.md)，这是定位工具而非已修复站姿识别的声明。
最新录像改善了坐站入镜范围；生产观测增加高置信倒置躯干拦截，避免错误骨架继续驱动机械臂。回测指标与限制见 [录像复核与调整](openarm_cloth/docs/standing_video_adjustment_20261022_CN.md)。
直接遥操现在只要求明确 START 与有效手腕即可开始，手指缺失保持夹爪，不再阻塞手腕启动；摄像头状态改为独立深色面板。见 [启动与文字显示修复](openarm_cloth/docs/direct_start_preview_fix_20261022_CN.md)。
需要分析卡顿/前后轴/手背识别时，增加 `--diagnostic-recording`，录制后台人体视频、分轴目标链路和各环节耗时，带65秒操作提示及自动保存。见 [诊断录制](openarm_cloth/docs/guided_diagnostic_recording_20261022_CN.md)。

手腕直接模式现已丢弃工作区外的多余位移，减轻越界后返向空走；预览默认限频10 Hz，未标定夹爪显示黄色提示。操作与验证边界见 [边界响应及预览修复](openarm_cloth/docs/wrist_boundary_preview_fix_20261022_CN.md)。
新克隆不包含本机视频、模型权重和 outputs；示范回放需另行准备对应数据。

## 快速开始

PowerShell：

```powershell
Set-Location -LiteralPath 'D:\openpi\latest\openpi--main'

# 查看整段视频动作回放，可旋转、缩放、暂停和逐帧
& .\run.ps1 -Mode fusion-replay

# 视觉识别并抓取蓝色方块；摆好物体后松开鼠标，空格开始
& .\run.ps1 -Mode tabletop -ScriptArgs '--shape','cube','--color','blue'

# 可选的几何抓取图形选择器
& .\run.ps1 -Mode tabletop-ui

# A 肩肘腕标定 + 完整相对掌面 → B 动力学双臂跟随
& .\run.ps1 -Mode live

# 同一条 A 映射参与双臂动作/右臂抓取任务，允许重试；空格开始
& .\run.ps1 -Mode human-task

# 同一个 B 环境中的资源检查、完整回归
& .\run.ps1 -Mode check
& .\run.ps1 -Mode test
```

回放窗口不能当作物体摆放窗口。`tabletop` 中双击物体选中，再 Ctrl＋鼠标右键拖动；
普通拖动用于调整视角，当前没有 WASD 摆物快捷键。相机图像窗口不能拖动三维物体。
抓取开始后不再进行摆放约束或复位。鼠标交互仍需专项使用验收，不能用离屏测试替代。

## 项目结构

| 路径 | 作用 |
| --- | --- |
| `openarm_cloth/` | B 机器人、相机标定、控制器、仿真、视觉抓取及 A→B 适配 |
| `pose2arm-main/` | A 人体/手部识别、动作参考、视频、模型及历史研究工具 |
| `openarm_cloth/outputs/` | 本地回放和验证证据；历史结果不随整理删除 |
| `docs/` | 本次工作区整理与归档说明 |
| `RUNNING.md` | 统一入口和完整命令 |
| `PROJECT_STATUS_20261021.md` | 当前实现、已有指标、真实失败情况和下一步 |

全部日常入口沿用 B 的 Python，配置保存在忽略提交的 `.runtime.local.json`。
旧依赖整包、独立 Notebook 和另一份工作副本已移到工作区外层 `_archive/`，
避免误用。源码、运行模型、标定、数据集、私人视频与有效输出均保留。

## 当前范围

人体动作复现与几何视觉抓取已共存，但仍是**独立运行模式**，不是已训练的完整任务策略。
`live` / `human-task` 默认复用 A 的肩肘腕映射、平滑与臂长标定，并增加躯干参考与 B 完整腕部适配，驱动 B 的左右臂。
独立 `human-task` 保留桌面/物块约束，右臂抓取评分，捏合闭爪/张手松爪，抓空允许继续调整；
`--control assisted` 保留原视觉对齐、自动抬升/放置。已验证明确标记的软件输入检查；
USB 真人操作验收仍待完成，不能把自动检查 PASS 计为真人示范成功。
用法与数据/回放说明见 [任务控制器](openarm_cloth/docs/human_task_20261021_CN.md)。
最新接线、标定操作及真实能力边界见 [原始 A 映射接入](openarm_cloth/docs/original_pose2arm_integration_20261021_CN.md)。
针对抖动新增执行端平滑、加速度限制，默认 A 夹爪改为连续开度；
实现与验证见 [实时平滑与连续夹爪](openarm_cloth/docs/live_smoothing_continuous_gripper_20261021_CN.md)。
最新真人失败记录已用于修复 A/B 限位精度导致的退出，并将默认人体模型换为 full，
新增当前手腕关联和观测一致性检查；见 [真人识别与退出修复](openarm_cloth/docs/live_observation_exit_fix_20261021_CN.md)。
默认 `fusion` 是纯动作运动学回放；`--execution dynamic` 才测试仿真执行器。
`tabletop` 使用 RGB-D 几何识别方块/长方体/竖直圆柱，当前每次物理演示一个目标。
没有承诺任意物体、任意摆位、多人/遮挡精度、全程安全、零抖动或实体硬件效果。

最近固定仿真验证六例通过（3 次物理抓取、3 次仅识别），但最新交互记录也有一次
“无足够对齐的抓取候选”失败；完整记录见状态文档，不把固定工况当作泛化成功率。
