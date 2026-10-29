# A 成果适配 B 机械臂：纯动作回放

> 2026-10-20：默认改为 `clean` 纯跟随重建，排除历史避障改写，处理不可信输入和异常
> 转腕。原 `fusion-replay` 命令即打开新结果，见 [修复说明](MOTION_REPAIR_20261020_CN.md)。

## 当前目标与已复用部分

用户最新要求：先复现已有自由动作视频，不要求在 B 的桌面/方块抓取场景执行，
暂不进行安全、抓取验收。

当前从 A 的 `data/p2a_test1_rtmw_hybrid_sim.npz` 读取身体方向观测与连续开合，复用
原片/镜像拆分和时间接口。默认 `clean` 从观测重建纯跟随参考，手臂与腕部解耦，
不把旧避障关节值当示范，不把镜像片段当原视频后半段。

机械臂使用 B 的模型与滑动夹爪；场景复用 B 已有 `build_motion_scene()`，移除桌面和方块。
两臂按关节名匹配，验证多个姿态的关键运动学点一致；旧 A 的旋转夹爪值不直接复制，
连续闭合量转换成 B 每指 `0.044 * (1 - closure)` 米。

当前链路没有物体视觉有效性门控、抓取辅助、额外 RRT 或碰撞投影。
只有显式 `--reference-mode legacy` 才保留旧 A 最终轨迹及其历史避障改动。

## 查看三维回放

在 PowerShell 执行：

```powershell
Set-Location -LiteralPath 'D:\openpi\latest\openpi--main'
& .\run.ps1 -Mode fusion-replay
```

打开两个同步窗口：原人体视频与 B 机械臂三维模型。三维视角不再固定。

- 鼠标左键拖动：旋转；右键拖动：平移；滚轮：缩放。
- 空格：暂停/继续；A/D 或左右方向键：暂停并逐帧前后查看。
- `+`/`-`：倍速/慢放（0.125–4 倍）；R：重播；G：显示/隐藏参考线；Q：退出。
- 洋红线为原 A 参考，青线为 B 目标，实体模型为保存的 B 状态。
  在默认运动学模式中三者应重合，不代表实体机械臂零误差。

默认读 `openarm_cloth/outputs/a_motion/latest.json` 指向的最近一次成功记录。
旧的完整桌面场景仍可显式打开，但不是当前默认：

```powershell
& .\run.ps1 -Mode fusion-replay -ScriptArgs '--recording','outputs\ab_motion\p2a_test1'
```

## 两种执行结果不能混称

### 1. clean 参考的运动学回放（默认）

直接写 B 模型关节位置后计算正运动学，不运行物理执行器，不启用避障。
用于查看从人体观测重建的动作；按原视频约 29.375 Hz 保留 612 帧。不能以零适配
误差替代人体识别/映射精度评估。

```powershell
# 生成并打开三维回放
& .\run.ps1 -Mode fusion

# 只生成记录
& .\run.ps1 -Mode fusion -ScriptArgs '--check'

# 指定新的输出目录
& .\run.ps1 -Mode fusion -ScriptArgs '--check','--outdir','outputs\a_motion\my_motion_test'
```

默认输出已存在时另加时间戳，保留旧结果；显式指定已完成的目录则拒绝覆盖。

### 2. B 仿真执行器动态跟随（对照）

沿用 B 的 PD 参数、重力补偿、力矩限制及物理子步，动作模式额外使用连续插值与
速度前馈；B 自主抓取控制器不变。目标重采样到 20 Hz。
没有主动避障，但引擎原有物理接触仍保留，因此可能有跟随偏差；不是实体机器人验证。

```powershell
& .\run.ps1 -Mode fusion -ScriptArgs '--check','--execution','dynamic','--no-set-default'

# 查看已生成的本次动态对照
& .\run.ps1 -Mode fusion-replay -ScriptArgs '--recording','outputs\a_motion\p2a_test1_dynamic'
```

`--no-set-default` 保持默认回放仍指向运动学结果。终端会打印实际输出路径。
记录中 `execution_mode` 区分两类结果，不能用运动学零误差宣称执行器零误差。

## MP4 导出与播放

MP4 本身是固定视角；需要自由视角请用上面的原生三维回放。
导出帧率随记录变化，不再把 29.375 Hz 的轨迹误放成 20 Hz。

```powershell
& .\run.ps1 -Mode fusion-replay -ScriptArgs '--render','outputs\a_motion\comparison.mp4'
Start-Process -FilePath 'D:\openpi\latest\openpi--main\openarm_cloth\outputs\a_motion\comparison.mp4'
```

## 更换视频与参考轨迹

当前复用已验证的 A 输出，所以**仅换视频路径不会自动生成新动作**。
先用 A 原有流程为新视频生成匹配的参考 NPZ，再提供：

```powershell
& .\run.ps1 -Mode fusion -ScriptArgs '--check','--video','D:\videos\new.mp4','--reference','D:\data\new_reference.npz','--reference-bindings','D:\data\new_bindings.json','--outdir','outputs\a_motion\new'
```

绑定 JSON 格式为 `{"schema_version":1,"references":[{"reference_sha256":"NPZ的SHA256","source_sha256":"视频的SHA256"}]}`。
绑定前必须确认二者确实来自同一次 A 流程；哈希只能防止串用文件，不能证明识别质量。
可以用 PowerShell 的 `Get-FileHash -Algorithm SHA256 -LiteralPath '绝对路径'` 获取哈希，填入时使用小写。

## p2a_test1 旧版本结果（保留对照，非当前默认）

| 项目 | 结果 |
|---|---|
| 原样映射 | 612 帧，源帧 0–611，约 20.834 秒 |
| A/B 关键运动学点核对 | 两臂最大约 1.62 微米 |
| 原样映射的关节值差 | 0；这是数据适配一致性，不是人体识别精度 |
| 前 4 秒动作 | 两臂均有摆动；肘部关节范围约左 104°、右 87° |
| 动态对照 | 417 帧，20 Hz；平均关节偏差约 1.88° |
| 动态末端平均偏差 | 左约 3.03 cm，右约 3.55 cm，相对 A 保存参考 |
| 动态瞬时最大关节偏差 | 约 61.5°；仍需控制跟随调试，不能宣称全程精确 |

完整回归 185 项：182 通过，3 项因符号链接权限跳过，无失败。原生三维查看器已进行
实际启动/退出检查，并验证 612 帧 MP4 的帧数、帧率和前段摆臂画面。
动态末端偏差 P95 约两臂各 8.9 cm，平均值不能代表每一帧的复现质量。

旧完整场景前 4 秒相对同一 A 参考的平均末端偏差约左 30 cm、右 43 cm；
本次动态对照约左 3.4 cm、右 6.0 cm。场景与控制模式已按用户要求改变，
这用于定位漏动作与查看复现效果，不能视为同场景的避障算法性能比较。

`episode.npz` 保存源帧、A 参考、B 目标与状态、开合、输入可信标记和帧率；
`manifest.json` 保存视频、参考 NPZ、场景源码哈希及执行模式；`result.json` 保存指标。
源输入不可信帧继续保留 A 已生成的参考处理结果，并保留不可信标记，不伪称为新检测。

当前没有训练抓取策略，也未形成合格抓取数据集。以后再将规范抓取示范接回 B 完整场景，
采集同步机器人相机、状态、动作与成功标签，再进行训练和独立测试。

## B 的视觉识别是否“定式”

B 原视觉抓取是规则式、特定对象的识别：按红色区域、RGB-D 深度、相机标定与已知
方块尺寸估计位置和朝向，再用 IK 与阶段控制抓取。物体位置不是写死的，能随观测变化；
对象规则和任务流程相对固定，不是自行训练得到的通用抓取策略。
本页的纯动作回放完全不依赖这套物体识别。
