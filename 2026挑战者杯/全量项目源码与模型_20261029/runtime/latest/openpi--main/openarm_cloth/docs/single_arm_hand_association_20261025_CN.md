# 单臂手部关联、单模型流与初始姿态更新

> 日期：2026-10-25
>
> 基线：`5e506d6`
> 范围：单臂 direct + `--mapping wrist` 的稳定性优化；未改变双手默认行为

## 背景

手背朝向摄像头时，MediaPipe Hand 本身能检出 21 个手部点，但旧关联逻辑要求手部腕点靠近 Pose 腕点。手向前伸或 Pose 腕点出画时，Pose 腕点漂移会超过 `wrist_tolerance`，导致已经检出的手被 `assign_candidates()` 丢弃，夹爪因此不响应。

同期实测直接手背片段：

```text
手部候选帧：                         509 / 535  (95.1%)
候选到 Pose 右腕距离超过 tolerance： 150 帧
最终 assign 到右手：                 359 / 535  (67.1%)
两帧确认后：                         323 / 535  (60.4%)
```

因此本次优先修关联，而不是替换手部模型。

## 代码变更

### 1. `assign_candidates` 增加 `expected_sides`

文件：`openarm_cloth/live_bimanual_stream.py`

- 单臂任务传入 `expected_sides=('right',)`；
- 画面只有一只可信手时，直接归到右手，不再要求靠近 Pose 腕点；
- 有两只手时仍走原一对一几何关联；
- 双手模式默认 `expected_sides=('left','right')`，行为保持原样；
- 单期望侧时，最终结果只保留该侧，避免单臂任务误绑左手。

### 2. `BimanualHumanStream` 单臂模式只创建一个手部模型

- 单臂模式不再创建左/右 ROI `HandLandmarker`，只保留全图 `num_hands=2` 模型；
- 全图手部检测每帧运行，替代原来的三模型并行；
- metadata 记录 `expected_sides` 和 `hand_search='full-frame every frame; one expected hand'`；
- 双手模式仍保留全图 + 左/右 ROI 三模型。

### 3. `run_human_task.py` 自动传入 active sides

```python
BimanualHumanStream(..., expected_sides=active_sides)
```

单臂任务 `active_sides=('right',)`；`--mapping pose2arm --arms both` 仍为双手。

### 4. 离线视频路径同步

文件：`pose2arm-main/extract_hand.py`、`openarm_cloth/full_video_source.py`

- `extract_hands_from_video(..., expected_sides=('right',))` 同样支持单臂单手归右；
- landmark 缓存记录 `expected_sides`，避免单臂/双手缓存混用；
- `run_human_task.py --video` 传入 `active_sides`。

### 5. 机器人初始 TCP 调整

文件：`openarm_cloth/human_task_robot.py`

```text
旧：[0.43, -0.10, 0.54]   目标区上方
新：[0.43,  0.00, 0.54]   方块上方
```

目的：

- 减少第一次接近方块的相对位移；
- 降低一开始就触发工作区裁剪的概率；
- 已验证 IK 误差约 0.6 mm，起始姿态与方块/桌面无接触，胸/腕相机仍能看到方块。

## 验证结果

### 单臂单手 fallback

同一段 `20261025_135124` 手背片段：

```text
修复前关联确认率： 60.4%
修复后关联确认率： 92.9%
```

完整 `run_human_task --video` 回放：

```text
修复前 human_hand_valid: 0.0%
修复后 human_hand_valid: 91.9%
```

### 新起始位

synthetic assisted 检查：

```text
verdict: PASS
11.15 s
227 帧
最大抬升 61.60 mm
```

### 回归测试

```text
openarm_cloth: 426 项通过，4 项跳过
```

## 使用方式

```powershell
.\run.ps1 -Python "C:\Users\23955\anaconda3\envs\openpi\python.exe" `
  -Mode human-task `
  -ScriptArgs '--arms','right','--camera','0','--control','direct','--mapping','wrist','--gripper-mode','continuous'
```

帧率优先时可加：

```powershell
--pose-model lite --preview-hz 5
```

## 限制

- 单臂 fallback 假设操作者只使用右手；单臂任务应让左手离开画面或保持不动。
- 双手模式默认逻辑未改；未来做双手时应继续使用 `expected_sides=('left','right')`，不要沿用单手归右规则。
- MediaPipe 官方 Windows Python wheel 不支持 GPU delegate；`delegate=GPU` 会报 `GPU processing is disabled in build flags`。
- RTMPose-Hand 的 A/B 显示 GPU 推理可行，但在这批手背片段上没有明显优于 MediaPipe；当前未接入生产链路。
- 手臂映射仍主要受 Pose 腕点出画、工作区裁剪和目标限速影响，后续需要单独调映射。
