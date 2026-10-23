# A/B 统一运行入口

最近整理：2026-10-21。当前能力、交互失败记录与下一步见
[项目状态](PROJECT_STATUS_20261021.md)，归档与恢复见 [清理记录](docs/cleanup_20261021.md)。
旧副本与依赖整包已移出工作位置，以下主项目路径和命令保持不变。

本地代码基于 `https://github.com/sishui667/openpi-` 的 `8df4726`。
当前工作目录为 `D:\openpi\latest\openpi--main`。Windows MuJoCo 的 XML、mesh
路径必须为纯英文/ASCII；统一入口会在启动前检查。

后续 A/B 融合固定沿用 B 的仿真、相机标定与执行器。
当前能力和剩余接入工作见 [统一方向说明](openarm_cloth/docs/ab_unified_direction_20261019.md)。

## 本机环境

统一入口使用 B 的 Python 3.12 / MuJoCo 环境，当前绑定
`D:\ProgramData\Anaconda3\envs\openpi\python.exe`。
本机选择保存在未提交的 `.runtime.local.json`，也可以通过 `-Python` 或
环境变量 `OPENPI_PYTHON` 显式指定。不会自动使用 WindowsApps 的 Python 存根，
也不再依赖不存在的 `venv_member_ab_live` 或其他开发者的 Miniconda 路径。

在 PowerShell 执行：

```powershell
Set-Location -LiteralPath 'D:\openpi\latest\openpi--main'
& .\run.ps1 -Mode check -ScriptArgs '--check-models'
```

## 实时双臂跟随

2026-10-22 的手部参数、连续夹爪与独立更新修复见 [手部分支集成说明](openarm_cloth/docs/direct_hand_branch_integration_20261022_CN.md)。
坐站切换、肘部遮挡与“10/10 不等于标定通过”的修复见 [站姿观测与标定反馈](openarm_cloth/docs/standing_observation_fix_20261022_CN.md)。

```powershell
& .\run.ps1 -Mode live
```

默认复用 **A 的 `OpenArmLiveController`** 处理根/肢体平滑、10 个有效帧臂长标定与肩肘腕 IK，
增加可靠躯干参考和相对前臂掌面旋转，经 B 三腕关节适配后由 B 原执行器负责动力学。
人体模型与手腕模式默认 full；auto 躯干在髋部不可见时使用正对相机的标定上身参考。
身体标定后，G 记录一次合拢、O 记录一次张开，每次保持至少 1.2 秒，交替至少 10 轮完成左右个人捏合标定。
默认仍是连续夹爪；C 重标定身体/腕部并保持夹爪，空格暂停，M 镜像预览，T 标记 15 秒开发片段，Q 保存退出。
完整说明与当前验收边界见 [动作跟随 v2](openarm_cloth/docs/motion_v2_implementation_20261021_CN.md)。
身体/手部位置不一致或手腕画外时会显示原因并保持，手部重新捕获需要两帧确认。
保留头部、双肩、肘腕在画面内；限位舍入差和误关联的修复见
[真人识别与退出修复](openarm_cloth/docs/live_observation_exit_fix_20261021_CN.md)。
默认 responsive 使用来源时间的 A 滤波及 B 加速度限制，去掉重复关节低通；夹爪按掌尺度归一化捏合连续改变开度，
预览显示左右开度百分比，半开也能保持。细节与录制对比见
[实时平滑与连续夹爪](openarm_cloth/docs/live_smoothing_continuous_gripper_20261021_CN.md)。
录像、数值记录保存在 `openarm_cloth\outputs\a_live`。关闭录像：

```powershell
& .\run.ps1 -Mode live -ScriptArgs '--no-save-video'
# 原 B split/legacy 姿态控制留作对照
& .\run.ps1 -Mode live -ScriptArgs '--mapping','b-legacy'
```

## A 原始映射参与抓取场景

```powershell
# 默认 A 原映射驱动双臂；无倒计时，右臂抓取评分，抓空继续调整
& .\run.ps1 -Mode human-task -ScriptArgs '--camera','0'
# 直接控制软件检查：20 秒空抓/张手/往返，结果 INCOMPLETE 正常，不冒充抓取成功
& .\run.ps1 -Mode human-task -ScriptArgs '--check','--scripted-check'
# 保留旧视觉辅助模式，捏合后自动对齐、抬升，绿色区张手后辅助放置
& .\run.ps1 -Mode human-task -ScriptArgs '--control','assisted','--timeout','60'
# 原辅助模式的合成输入完整抓放检查
& .\run.ps1 -Mode human-task -ScriptArgs '--control','assisted','--check','--scripted-check'
```

当前默认 `--mapping pose2arm --arms both`。先用 10 个有效观测标定各臂，右手张开后空格开始。
直接遥操模式按空格/START 且手腕有效即可进入 DIRECT，手指缺失不再阻塞启动，夹爪保持原开度。辅助抓取模式仍要求识别到手势。状态改为摄像头右侧深色面板，底部大字提示 READY、START REQUESTED、PAUSED 或 DIRECT。
定位卡顿、前后轴和手背捏合可加 `--diagnostic-recording`：后台保存本地人体录像，关闭逐帧PNG写盘，按屏幕提示分轴/手掌手背操作，START后65秒自动保存。完整命令与输出字段见 [引导式诊断录制](openarm_cloth/docs/guided_diagnostic_recording_20261022_CN.md)。
左手→左臂、右手→右臂，捏合闭爪/张手松爪，下降/抬升/搬运由人操作，不自动接管抓取。
B 保留桌面/物块接触检查，受阻侧限步或保持，另一侧及夹爪独立处理；评分仍只看右臂任务。
`--arms right` 仅控制右臂；`--mapping wrist` 保留此前小工作区手腕位置控制。
桌面末端位置操作可用 `--mapping wrist --gripper-mode continuous`，现在也使用个人捏合端点与连续开度；G 合拢/O 张开各至少 10 次，每次保持 1.2 秒。

手腕直接模式到工作区边界后不再累积越界位移，返向目标可立即响应。预览默认最多10 Hz，键盘每个控制周期处理；可加 `--preview-hz 5` 降低显示负担。平时不需要机器人 PNG 数据时可加 `--no-images`，诊断录制会自动关闭这些 PNG。未标定时屏幕有黄色 G/O 提示，启动日志打印实际标定路径；达到 `CALIBRATION READY` 后 Q 保存。详见 [边界响应及预览修复](openarm_cloth/docs/wrist_boundary_preview_fix_20261022_CN.md)。

标定的噪声门槛现按每次保持期间的波动计算，避免不同次张开幅度的差异造成过度拒绝；错误标签和真实重叠仍拒绝。原因与本地续标定流程见 [标定质量修复](openarm_cloth/docs/pinch_calibration_quality_fix_20261023_CN.md)。

连续夹爪另用典型端点插值，并将可用且相互一致的二维间距用于保留小开度，避免半开时提前全闭。旧标定原始试次加载时自动重算，无须重采；需同时复测真实捏合能否充分闭合。见 [连续开度范围修复](openarm_cloth/docs/continuous_pinch_range_fix_20261023_CN.md)。

直接手腕路径的IK与碰撞查询已减少无关动力学计算，新增细分耗时。`--diagnostic-recording --diagnostic-protocol response` 可后台录像并提示65秒半开移动/开度分档/遮挡恢复复测，详见 [执行查询优化与复测](openarm_cloth/docs/direct_query_optimization_20261023_CN.md)。

移动时需要可靠保持夹爪开度：直接手腕模式开始后，H固定半开，L保持当前开度／解除保持。屏幕明确显示LOCKED；解除后等待两帧新有效手部观测再接回连续控制。它是显式操作功能，默认不自动锁定，单目视觉漂移仍需继续解决。详见 [开度保持操作与限制](openarm_cloth/docs/explicit_gripper_hold_20261023_CN.md)。
试次数达标后还须看到 `CALIBRATION READY`；`NOT READY` 应按原因处理。若提示 `LABELS REVERSED`，换新个人文件重新采样，G=指尖接触，O=两指分开。
坐站切换后完整入镜仍漏检时，按 R 暂停并重新创建人体/手部视觉跟踪器，等 `VISION REACQUIRING` 消失后检查识别状态。恢复后按空格继续；C 仅重置动作映射，不能替代视觉重识别。若 R 后仍漏检，用 `--save-human-video` 在本地记录一次坐站切换，详见 [站姿诊断流程](openarm_cloth/docs/standing_observation_fix_20261022_CN.md)。
若人体骨架整体消失或落到背景，先做不加载仿真的摄像头诊断：`& .\run.ps1 -Mode python -ScriptArgs 'diagnose_human_tracking.py','--seconds','35','--save-video','--compare'`。1 标记坐姿、2 标记站姿、R 重识别、Q 保存后对同一录像离线比较检测模式；具体证据边界见上述流程。
手背检出不足可对照 `--hand-confidence 0.35`；检测/存在性/跟踪可独立设置，默认仍 .5，降低门限不代表提高准确率。
完整选项、夹爪保持原因与验证边界见 [手部分支集成说明](openarm_cloth/docs/direct_hand_branch_integration_20261022_CN.md)。
要做双臂动作跟随用 `live`，该模式使用独立的无桌面/方块场景，不做抓取任务。
碰撞/IK 拒绝仅保持和提示，抓空、掉落和识别丢失可调整重试；意外程序错误仍停止并保存。
C 重新人体标定、空格暂停、M 镜像预览、Q 保存退出；默认无时限，`--timeout` 在直接模式只提醒。
画面显示 `WRIST_LOW_CONFIDENCE` 时，将右手与躯干错开并保持肩膀/手腕入镜；
`STALE_CAMERA_FRAME` 表示处理后的图像已过期，IK/碰撞拒绝会单独提示，不统称识别失败。
若三维查看器把仿真复位（时钟归零），程序停止该条记录并提示重启；
不要在进行中的任务里使用原生复位快捷键，重新运行命令会建立新的初始工作姿态和记录。
原始人体视频只在 `--save-human-video` 时本地保存。
有真实操作视频时可传 `--video <绝对路径>`；自由摆臂视频不能代替抓取示范。
数据校验、状态/动作回放与能力边界见
[任务控制器说明](openarm_cloth/docs/human_task_20261021_CN.md)。
本轮实际接线、数据字段与验证范围见 [A 原始映射接入说明](openarm_cloth/docs/original_pose2arm_integration_20261021_CN.md)。

## A→B 物理回放

这里是旧的关节轨迹回放。视频示范参与 B 视觉抓取请使用下方的融合入口。

```powershell
& .\run.ps1 -Mode replay
```

默认使用交付的 20 Hz `a_hybrid_orig_20hz.npz`，固定右臂执行，左臂待机。
空格暂停，R 重播，Q 退出。指定新文件时使用绝对路径：

```powershell
& .\run.ps1 -Mode replay -ScriptArgs '--npz','D:\your-data\new_20hz.npz'
& .\run.ps1 -Mode replay -ScriptArgs '--check','--report','outputs\replay_check.json'
```

原始视频需要先由 A 提取并导出 B 接口数据。更换视频文件本身不会自动生成新轨迹。
导出示例（从统一入口运行）：

```powershell
& .\run.ps1 -Mode python -ScriptArgs '..\pose2arm-main\member_b_export.py','--help'
```

## B 自主视觉抓取

```powershell
& .\run.ps1 -Mode grasp
```

## 多颜色、多形状几何视觉抓取（已合入 latest）

与自由动作回放独立，使用同一个 B Python，不改变默认 `fusion` / `fusion-replay`。
支持方块、长方体、竖直圆柱；RGB-D 测量尺寸和位姿，不要求先选择类别或尺寸。
这是几何识别，不是新训练的通用物体/抓取模型。

```powershell
# 打开三维场景，随机布置一个受支持物体；摆好后松开鼠标，按空格识别抓取
& .\run.ps1 -Mode tabletop

# 图形选择器：默认自动识别，可选展开自定义测试物体
& .\run.ps1 -Mode tabletop-ui

# 指定绿色长方体，仅用于构建测试世界，不给识别器答案
& .\run.ps1 -Mode tabletop -ScriptArgs '--shape','box','--color','green','--size','.070','.040','.055','--fixture-pose','.43','.10','30'

# 六个固定样例：三个物理抓取、三个只识别；输出目录需为空或不存在
& .\run.ps1 -Mode tabletop-check
```

Q 退出；R 只在摆放阶段复位物体。开始抓取后不锁坐标、不传送物体。
新报告按时间另存 `openarm_cloth/outputs/generic_objects/`。
冻结红方块场景不改，挡边只进入此模式生成的独立模型。
能力边界、合入文件与验证结果见 [多物体模块合入说明](openarm_cloth/docs/tabletop_merge_20261020_CN.md)。

以上模式均通过 B 的执行器、控制器与场景构建代码进入仿真。
抓取使用 `openarm_cloth\simulation\v1\droid_cube_scene_perception.xml` 及
`task_workspace.json`；真人跟随使用由该场景生成的 `live_motion_scene.py`，移除桌面和方块。
两者保留 B 的物理步长、限位、力矩、相机和夹爪设置。

## 当前：A 原参考 → B 机械臂纯动作回放

> 当前默认已按用户最新要求改为 **A 参考轨迹 → B 机械臂纯动作回放**，不再要求桌面
> 抓取场景，也不启用视觉抓取或额外避障。下面是此前完整场景的历史说明，不能作为
> 当前 `fusion` 命令的使用说明。最新命令见 [纯动作回放说明](pose2arm-main/videos/A_REFERENCE_MOTION_REPLAY_CN.md)。

当前快速入口：

```powershell
# 可旋转/缩放的三维回放，同步原视频
& .\run.ps1 -Mode fusion-replay

# 重新生成：默认从 A 观测重建 clean 参考，运动学回放，无避障
& .\run.ps1 -Mode fusion -ScriptArgs '--check'

# B 仿真执行器动态对照；不替代当前默认回放
& .\run.ps1 -Mode fusion -ScriptArgs '--check','--execution','dynamic','--no-set-default'
```

`fusion` 默认以 `clean` 模式从 A 观测/手势重建 612 帧纯跟随参考，不再沿用历史避障
关节值；动态模式重采样到 20 Hz，使用连续插值与速度前馈。
异常转腕修复及新旧指标见 [修复说明](pose2arm-main/videos/MOTION_REPAIR_20261020_CN.md)。
新视频必须提供与之匹配的 A 参考 NPZ 和哈希绑定，不能只更换视频路径复用旧动作。
下面保留的旧完整场景命令已统一改成 `landmark-fusion`。

## 历史：人体关键点 → B 完整视觉场景

```powershell
# 查看已完成的整段融合回放：左侧人体关键点，右侧 B 实际物理执行
& .\run.ps1 -Mode fusion-replay -ScriptArgs '--recording','outputs\ab_motion\p2a_test1'

# 重新执行 p2a_test1：完整双臂、手腕和夹爪跟随 + B 视觉识别
& .\run.ps1 -Mode landmark-fusion

# 更换视频：首次会重新提取人体/手部关键点
& .\run.ps1 -Mode landmark-fusion -ScriptArgs '--video','D:\your-data\new.mp4','--outdir','outputs\ab_motion\new'

# 无交互运行并保存物理回放
& .\run.ps1 -Mode landmark-fusion -ScriptArgs '--check','--cube-test-pose','0.43','0','30'

# 查看新视频的物理记录
& .\run.ps1 -Mode fusion-replay -ScriptArgs '--recording','outputs\ab_motion\new'
```

历史 `landmark-fusion` 运行 `run_fused_motion.py`，不再只取十几帧小曲线。整段人体输入按源时间
采样到 20 Hz；两臂、三轴手腕和独立开合共同接入 B 的 IK、避障与执行器。
保留 B 的完整桌面、方块、胸部/腕部 RGB-D 与标定。近物闭合时才尝试视觉抓取对齐；
远处握手不会自动启动整套抓取。具体边界、测试结果和换视频命令见
[完整融合说明](pose2arm-main/videos/AB_FULL_MOTION_ON_B_CN.md)。

历史完整场景回放目录为 `openarm_cloth/outputs/ab_motion/p2a_test1`，需显式传 `--recording`。
本次 `COMPLETED` 表示整段执行完成，不代表抓取成功或所有动作准确无抖动。
现有视频未完成夹持；手腕平均方向误差仍约 34–36°，应先查看对照回放再评估质量。

之前的“受约束单臂小曲线 + B 自主抓取”保留为对照，不再混称整段融合：

```powershell
& .\run.ps1 -Mode video-skill
& .\run.ps1 -Mode fusion-replay -ScriptArgs '--recording','outputs\video_fusion\p2a_test1'
```

其三个摆位的历史 PASS 仅属于该受约束抓取模式，说明见
[旧片段抓取记录](pose2arm-main/videos/AB_VIDEO_VISUAL_GRASP_CN.md)。

## 回归与新机器安装

```powershell
& .\run.ps1 -Mode test
```

会在同一个 Python 中依次运行 B 测试和 A 接口测试。临时文件写入
`openarm_cloth\outputs\runtime_tmp`，避免系统沙箱临时目录权限问题。

新机器创建一个 Python 3.12 环境后安装：

```powershell
python -m pip install -r openarm_cloth\requirements_live.txt
python openarm_cloth\setup_models.py
& .\run.ps1 -Python 'C:\your-env\python.exe' -Mode check -ScriptArgs '--check-models'
```

模型资源来自官方 MediaPipe 固定版本 URL。手部、lite/full/heavy 人体模型存放在
`pose2arm-main\models`，该目录不提交 Git。

## 历史工具与恢复

A 的 `scripts/replay_mujoco.py`、`collision_safety.py` 和碰撞诊断保留用于历史
NPZ 的旋转夹爪模型，其 XML 现在来自 B 目录下的 `official_openarm_mujoco/v2/`。
它们的 ±0.7854 rad 语义不能直接用于 B 的 0–0.044 m 滑动夹爪，也不能作为 B
实际场景的碰撞认证。日常运行使用上述 B 入口，B 实时避障直接检查当前模型。

本机旧版本移到 `D:\openpi\_archive\before_github_8df4726_20261019`，可恢复。
旧目录中的私人视频、模型和未上传输出仍保留在归档内。
历史验收 JSON 的绝对路径、哈希与时间戳保留原状；新的运行结果另存 `outputs`，
不会把历史证据改写成当前测试结果。

2026-10-21 又将另一份工作副本、上游旧依赖整包、独立 Notebook 和外层旧笔记
归档至 `D:\openpi\_archive\cleanup_20261021`，逐文件哈希核验，未永久删除。
主项目所需视频、权重、参考、标定与 outputs 保留。
