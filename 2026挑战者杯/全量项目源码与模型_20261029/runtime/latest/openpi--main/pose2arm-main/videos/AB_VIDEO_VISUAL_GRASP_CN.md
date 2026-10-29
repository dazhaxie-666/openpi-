# 视频示范接入 B 自动视觉抓取：第一阶段

> 历史受约束片段方案：现在使用 `-Mode video-skill`。当前默认 `-Mode fusion`
> 已切换为整段双臂/手腕/夹爪映射，见 [完整融合说明](AB_FULL_MOTION_ON_B_CN.md)。
> 下列三摆位 PASS 只属于旧片段方案，不是整段视频抓取成功的证据。

## 已接通的流程

`p2a_test1.mp4` → A 的 MediaPipe 身体/手部识别 → 归一化腕部轨迹和捏合意图 →
B 胸部/腕部 RGB-D 定位方块 → 物体坐标系轨迹重定位 → B IK、路径检查、PD 执行 →
真实摩擦夹持和抬升 → 保存物理回放。

全部沿用 B 的 Python、MuJoCo 场景、桌面、相机、标定、滑动夹爪和执行器。
本次修改未改变 B 的六项冻结物理/任务配置。A 的关节角没有与 B 的关节角叠加。

## 立即查看已完成回放

在 PowerShell 执行：

```powershell
Set-Location -LiteralPath 'D:\openpi\latest\openpi--main'
& .\run.ps1 -Mode fusion-replay -ScriptArgs '--recording','outputs\video_fusion\p2a_test1'
```

左侧为原人体视频，右侧为保存的 B 物理执行。空格暂停/继续、R 重播、Q 退出。
默认循环。物体位姿来自此前的真实物理模拟记录，回放不重新求 IK。

已渲染的 MP4 为：

`D:\openpi\latest\openpi--main\openarm_cloth\outputs\video_fusion\p2a_test1\comparison.mp4`

回放文件包含场景哈希、工作区配置和源视频哈希，加载时验证，避免看错视频或模型。
如果移动了原视频，可以给回放增加 `--video` 指向同一内容的新路径。

## 重新执行与更换视频

```powershell
# 默认视频 p2a_test1.mp4：显示 B 的视觉识别、规划与执行窗口
& .\run.ps1 -Mode video-skill

# 先交互摆位，再让 B 重新视觉定位
& .\run.ps1 -Mode video-skill -ScriptArgs '--interactive-place'

# 替换为新视频，并使用独立输出目录
& .\run.ps1 -Mode video-skill -ScriptArgs '--video','D:\your-video\new.mp4','--outdir','outputs\video_fusion\new'

# 使用人体左手作为示范来源；B 执行臂仍由物体视觉位置选择
& .\run.ps1 -Mode video-skill -ScriptArgs '--side','left'

# 原视频内容或提取参数变化后，可显式重新识别
& .\run.ps1 -Mode video-skill -ScriptArgs '--refresh'

# 指定有效开合片段编号；默认 -1 自动选最强的有效闭合事件
& .\run.ps1 -Mode video-skill -ScriptArgs '--event','0'

# 不开交互窗口，仍进行完整视觉与物理计算
& .\run.ps1 -Mode video-skill -ScriptArgs '--check','--cube-test-pose','0.43','0','30'

# 查看新运行或其他摆位的记录
& .\run.ps1 -Mode fusion-replay -ScriptArgs '--recording','outputs\video_fusion\new'

# 导出并排 MP4
& .\run.ps1 -Mode fusion-replay -ScriptArgs '--recording','outputs\video_fusion\p2a_test1','--render','outputs\video_fusion\p2a_test1\comparison.mp4'
```

所有相对输出目录均从 B 目录 `openarm_cloth` 解析。输入视频建议使用绝对路径。
识别缓存位于 `openarm_cloth/outputs/video_fusion/cache`，内容变化时通过 SHA-256 失效。
源文件名相同但内容不同会重新提取，重算无需手工修改代码中的路径。

## 当前视频实际采用了什么

本轮由当前 A 代码重新提取 612 帧。选用右手的有效片段，腕部曲线来源为
第 140–151 帧，约 4.766–5.140 秒；其后连续闭合段支持夹爪意图。
名义人体到机器人比例约 0.908，测得稳定闭合量约 0.99792，
B 每指闭合目标约 0.0000915 m，而非历史旋转夹爪的 ±0.7854 rad。

由于原视频是自由动作，本阶段迁移其接近路径的**形状**：

- 以肩膀与躯干建立人体参考系，使用人体臂长进行名义归一化。
- 消除曲线端点之间的直线趋势，保留弯曲形状。
- B 固定视觉接近/抓取端点；人体横向残差最大 12 mm，垂向残差最大 4 mm。
- 曲线随物体视觉朝向旋转，精确抓取位置与夹爪朝向由 B 决定。
- 9 个位置锚点求 B IK；使用不超调的 PCHIP 和五次时间映射，限制关节速度，
  起止目标速度/加速度为零。当前接近片段重定时为 1.5 秒。
- 检查 B 当前碰撞几何。若违反限位、IK 误差或新增穿透阈值，降低人体曲线增益；
  任何增益都不可行时拒绝执行，原因写入 `failure.json`。
- 夹爪必须有有效视频闭合意图。缺失、长时遮挡或不存在张手→闭合片段时拒绝执行。
- 仅被 B 选中的执行臂接收开合命令，另一臂在待机位并张开。

本轮人体曲线实际产生约 9.2 mm 的横向偏移，三个工况接受增益均为 1.0，
没有全部退回纯 B 直线轨迹。详细目标点、尝试记录、闭合来源和缩放比例见各运行的 `result.json`。

## 实测结果（2026-10-19）

| 方块初始摆位 X/Y/角度 | B 选择执行臂 | 抬升 | 仿真动作阶段耗时 | 任务门限 |
|---|---|---|---|---|
| 0.43 / 0 / 30° | 右 | 71.9 mm | 14.015 s | PASS |
| 0.39 / -0.08 / 0° | 右 | 74.6 mm | 14.167 s | PASS |
| 0.47 / 0.06 / 45° | 左 | 69.1 mm | 13.973 s | PASS |

原始结果目录分别为 `openarm_cloth/outputs/video_fusion/p2a_test1`、
`near_right0`、`far_left45`。每个目录保存 `result.json`、`manifest.json`、
`episode.npz` 和夹持/抬升截图。时间不包含离线人体识别和前置视觉规划，
也不代表端到端实时性能。

任务门限沿用 B：相对面内侧接触，抬升超过 30 mm，夹爪到物体距离小于
20 mm 并保持至少 0.5 秒，动作阶段不超过 15 秒。

新增专项测试验证输入质量门控、人体形状影响、物体平移/旋转重定位、
长时丢失拒绝、关节插值不超调、速度约束及隐藏物体坐标隔离。
最终完整回归共 165 项：162 通过、3 项因符号链接权限跳过，无失败。
开发中发现的普通三次样条限位间超调已由 PCHIP 修复，没有通过放宽碰撞判据绕过。

## 本阶段边界

这是离线、单执行臂的约束示范迁移。视频的观察/闭合/运动阶段经过重定时，
并排回放不表示机器人等时逐帧模仿完整视频：B 自主观察、接近、抬升期间，
人体画面保持源帧；人体曲线接近阶段才推进对应片段。
闭合强度取该事件中最强的稳定测量窗口，因此也不等于当前显示源帧的瞬时闭合值。

暂未迁移手腕姿态、双手协作搬运或视频张手触发释放。抬升后由 B 保持夹持，
避免尚未完成搬运/放置时直接照搬张手动作。仍未接实时人体输入，也未训练新权重。
三种摆位通过不是任意物体、任意摆位的成功率；离散碰撞采样也不是连续无碰撞保证。

旧 B 独立视觉抓取继续使用 `run.ps1 -Mode grasp`，旧 A 关节回放使用 `-Mode replay`。
这两个入口保留作为对照，本页受约束片段融合现在使用 `-Mode video-skill`。
