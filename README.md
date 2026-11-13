# OpenArm / Pose2Arm A+B 主项目

2026-10-27 GitHub 更新：仓库内可用 `start_shared_grasp.ps1` 启动当前共享抓取。
新电脑配置、发布范围及验证见[本次更新说明](docs/github_publish_20261027.md)。

最新交互：遮挡上抬由人手位移控制，取消固定高度自动上升；自然张手靠近，进入范围直接捏合抓取，无需额外张手准备。
操作与证据见[手动上抬与抓取防误触](openarm_cloth/docs/manual_view_pinch_20261027_CN.md)。

最新放置修复：下降允许有限腕姿调整；不可达时恢复人工搬运并提示换位置，避免持续张手原地重试。
实测定位、代码与论文采用范围见[张手放置优化](openarm_cloth/docs/placement_release_research_20261027_CN.md)。

最新交互：进入辅助抓取范围显示中文绿色大字；抓牢后直接张手确认放下，无需再次捏合。
操作与证据见[抓取提示和放下修复](openarm_cloth/docs/grasp_prompt_release_20261027_CN.md)。

最新接触入口修复：按方块表面距离接受近处捏合，遮挡时允许明确向上退出恢复视野。
当前识别范围与证据见[接触入口和识别范围](openarm_cloth/docs/contact_entry_fix_20261027_CN.md)。

现场追加：修复搬运受限后持物暂停目标逐帧漂移导致的下沉，保留夹持并支持移回可达方向继续搬运。
见[持物保持修复](openarm_cloth/docs/carry_hold_fix_20261027_CN.md)。

当前现场追加：人手靠近/搬运跟随，7.5 cm 内捏合才接管精抓，持续张手确认取消，完整任务从人工靠近开始分段。
运行与验证见[人工跟随与完整任务采集](openarm_cloth/docs/shared_human_workflow_20261027_CN.md)。

2026-10-27：当前抓取入口改为共享流程——人手粗移动/捏合请求，B RGB-D 规划对准、闭合、确认、提起，
人手搬运并张手请求支撑释放。支持连续试次自动分段与异步 RGB-D 采集。
运行 `D:\openpi\启动共享抓取.ps1`；操作和验证见[共享抓取与连续采集](openarm_cloth/docs/shared_grasp_dataset_20261027_CN.md)。
现场接触后卡住已追加修复：捏合时若夹爪遮挡方块，先张爪向上退出并重新取得深度位姿，再自动抓取。

2026-10-26 现场追加修复：单臂默认支持手掌倾转控制夹爪桌面转角，
修复近物限速从编码器反复起步导致的迟缓，增加捏合时的有界视觉居中与先留开口后闭合。
当前操作和证据见[手腕与贴边抓取修复](openarm_cloth/docs/wrist_edge_grasp_fix_20261026_CN.md)。
正常入口已设为 30 Hz 控制与 20 Hz 三维显示；同录像窗口实测约 29.22/19.82 Hz，动作重放通过。

2026-10-26 自动抓取辅助：正常单臂入口现已自动渐进精细控制、确认夹持后保持、明确张手释放，
采集过程无需精细切换/锁定按键。已改善的人手识别与前后/左右观测源保持不变。
操作、验证与仅关闭辅助的回退方式见[自动精细遥操与稳定夹持](openarm_cloth/docs/automatic_grasp_assist_20261026_CN.md)。

2026-10-26 当前修复版：原 MediaPipe 识别，固定图像左右/上下尺度，掌部 IPPE 图像求姿前后输入；
默认不再运行额外肘点显示模型。正常入口仍为 `run.ps1 -Mode single-arm`。
最新现场冻结的复现、实际仿真执行与回退见[修复版交付说明](openarm_cloth/docs/single_arm_emergency_release_20261026_CN.md)。
下方此前同日修复记录保留为历史过程，当前运行配置以上述交付说明为准。

2026-10-26 更新：新现场记录证实 RTMPose 二维身体控制存在尺度和身份门控回归，
单臂控制已恢复 MediaPipe 上身输入，前后保留独立掌部投影。RTMPose 仅在后台修复显示肘点。
减少三维刷新拷贝并限制启动前空闲日志增长；内置相机实测 720p 约 30 Hz，1080p 请求仍返回 720p。
日常直接 `run.ps1 -Mode single-arm`，无需反复诊断。见[性能修复、相机实测与使用](openarm_cloth/docs/single_arm_regression_camera_20261026_CN.md)。

2026-10-26：50 秒真人记录定位到近镜头时的投影保持和三维刷新负载。
已增加有形状与时限约束的二维掌部尺度补偿，三维刷新默认 10 Hz；同观测回放有效数
594/682→633/682，仍非新的真人验收。见[复核与调整](openarm_cloth/docs/single_arm_scale_preview_20261026_CN.md)。

2026-10-26：已执行用户提供的 RGB 单目排障方案，增加四层对齐日志与报告，
前后 ±2 cm 仿真和直接关节微动检查通过。当前现场复测使用 50 秒 `depth-chain` 引导，见
[四层诊断和下一步操作](openarm_cloth/docs/single_arm_four_layer_diagnosis_20261026_CN.md)。

2026-10-26：现场记录复现了夹爪稳定条件冻结前后输入、短时闪断后归零吞掉回收动作。
已解除这两处耦合；原记录有效前后输入由 47/210 提高到 152/210，仍需新真人验收。
运行方式、视频回放结果与剩余碰撞/姿态限制见
[前后输入修复](openarm_cloth/docs/single_arm_depth_decoupling_20261026_CN.md)。

23:04 真人录像确认原人体模型的前后信号不可靠；单臂默认改为掌部二维/三维投影的相对前后输入，
左右上下保持原上身位置。新 `DEPTH` 状态、录像对齐验证及仍会保持的场景见
[掌部相对前后输入](openarm_cloth/docs/single_arm_palm_depth_20261025_CN.md)。

2026-10-25：当前优先实现单臂实时遥操。运行 `& .\run.ps1 -Mode single-arm`，
右手 XYZ 位移与连续捏合控制右臂；已接入今天的单臂分支并优化识别调度与物理时钟。
操作、测量结果和论文采用范围见 [单臂实时遥操说明](openarm_cloth/docs/single_arm_realtime_20261025_CN.md)。
后续卡顿反馈已增加独立摄像头显示、三维刷新节拍与识别后连续小位移，见
[显示与动作连续性修复](openarm_cloth/docs/single_arm_preview_continuity_20261025_CN.md)。
前后轴混合及远距离手部丢失反馈，增加单臂上身位置参考、局部恢复搜索与边界提示，见
[前后轴与远距离手部说明](openarm_cloth/docs/single_arm_depth_distance_20261025_CN.md)。
闭合末段已修正连续收拢的重复确认与个人闭合终点，前后轴支持独立增益；
整臂前送、后收采用连续位置优先与肩肘腕共同限速，解析 IK 减少每帧求解开销。
论文/社区依据、离线深度候选与验证范围见
[闭合末段及前后平移研究调整](openarm_cloth/docs/single_arm_closure_depth_research_20261025_CN.md)。

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
