# 完整 A 动作映射接入 B 视觉环境

> 历史完整场景方案。用户已将当前目标调整为“不在桌面抓取场景中复现，先保留 A 原有
> 动作，不启用额外避障”。当前 `fusion` / 默认回放见
> [A 原参考适配 B 机械臂](A_REFERENCE_MOTION_REPLAY_CN.md)。本页旧执行入口现在为
> `landmark-fusion`，旧记录需显式指定 `--recording outputs\ab_motion\p2a_test1`。

## 这次融合的含义

当前主入口不再只取视频中十几帧、毫米级曲线去修饰 B 的自主抓取。
它读取整个视频，让两臂、手腕和开合持续控制 B 的机械臂，同时保留 B 的视觉定位与完整物理场景。

执行链路：

```text
整段视频 → A 身体/手部关键点 → 人体参考系、方向归一化、滤波、连续开合
                                  ↓
B 胸部/腕部 RGB-D → 视觉物体位姿 → B 模型 IK → 共享控制/碰撞协调 → B 执行器
```

A 提供动作输入，不提供另一套叠加关节命令。B 的桌面、方块、相机标定、物理步长、
力矩和滑动夹爪保持原环境。新入口初始关节姿态使用 B 已有的弯肘跟随姿态，避免
抓取待机位约 7.16 mm 的连杆间距低于原有 8 mm 安全裕量而启动即锁住；未放宽安全裕量。

## 立即看回放

在 PowerShell 执行以下两行：

```powershell
Set-Location -LiteralPath 'D:\openpi\latest\openpi--main'
& .\run.ps1 -Mode fusion-replay
```

空格暂停/继续，R 重播，Q 退出，默认循环。
左侧显示对应原视频帧及肩—肘—腕、21 点手部连线；右侧是保存的 B 物理状态，
不是重新求 IK 的理想目标。青色标记是视觉估计中心上方 4 cm 的显示标记，不是物理物体。

也可以直接用系统播放器看 MP4：

```powershell
Start-Process -FilePath 'D:\openpi\latest\openpi--main\openarm_cloth\outputs\ab_motion\p2a_test1\comparison.mp4'
```

回放文件验证源视频 SHA-256、B 场景哈希和工作区配置；视频读取采用顺序解码，
支持尾帧与循环重播，不依赖 MP4 尾部随机定位。

## 重新运行、更换视频

```powershell
# 默认视频：pose2arm-main/videos/p2a_test1.mp4
& .\run.ps1 -Mode fusion

# 不开交互窗口，仍运行视觉渲染、避障与全部物理子步
& .\run.ps1 -Mode fusion -ScriptArgs '--check'

# 新视频使用独立输出目录，防止覆盖上一段结果
& .\run.ps1 -Mode fusion -ScriptArgs '--video','D:\your-video\new.mp4','--outdir','outputs\ab_motion\new'

# 看新视频结果
& .\run.ps1 -Mode fusion-replay -ScriptArgs '--recording','outputs\ab_motion\new'

# 输出并排视频，无交互窗口
& .\run.ps1 -Mode fusion-replay -ScriptArgs '--recording','outputs\ab_motion\new','--render','outputs\ab_motion\new\comparison.mp4'

# 原视频移动后：内容须与原录制一致，哈希不同会拒绝
& .\run.ps1 -Mode fusion-replay -ScriptArgs '--video','D:\moved-video\p2a_test1.mp4'

# 强制重新识别 / 指定方块测试摆位 / 短时试跑
& .\run.ps1 -Mode fusion -ScriptArgs '--refresh','--check','--cube-test-pose','0.43','0','30','--seconds','5'

# 对比：关闭近物视觉抓取辅助，保留整段人体跟随、B 视觉及避障
& .\run.ps1 -Mode fusion -ScriptArgs '--no-assist-grasp','--outdir','outputs\ab_motion\follow_only'

# 默认三轴转腕、拇指-食指连续捏合；可改成握拳识别或简化转腕
& .\run.ps1 -Mode fusion -ScriptArgs '--gripper-mode','fist','--wrist-mode','twist','--outdir','outputs\ab_motion\fist_twist'
```

输入视频请用绝对路径。相对输出目录从 `openarm_cloth` 解析。
同一输出目录重新运行会替换本次运行产物，重要结果请使用新的 `--outdir`。
人体缓存位于 `openarm_cloth/outputs/ab_motion/cache`，视频内容变化时哈希失效并重提取。
统一入口固定使用 B 的 Python；本机为 `D:\ProgramData\Anaconda3\envs\openpi\python.exe`。

## 坐标和控制权怎样融合

- 使用肩、髋构建人体躯干参考系，把上臂/前臂转为单位方向，再由 B 自身连杆长度求 IK。
  不是把人体米制坐标原封不动放大到机器人。手掌平面/手指方向提供腕部姿态；
  单目深度和遮挡误差仍会影响方向，归一化不能消除识别误差。
- 整段 612 帧参与人体滤波，控制器按原视频时间以 20 Hz 采样，而非挑一个闭合事件。
  高于 20 Hz 的源帧不逐一成为独立物理控制目标；回放显示与控制目标对应的源帧。
- 捏合距离按手部尺度归一化后形成 0–1 闭合量，转换为 B 每指 0–0.044 m。
  没有沿用旧 A 旋转夹爪的弧度。身体、手部使用 One Euro 滤波，目标通过 B 限速、
  避障和 PD 物理执行，夹爪命令另有限速；这些措施不等于已经证明零抖动。
- B 每 0.5 秒尝试胸部和腕部 RGB-D 识别；腕部估计与胸部一致时才精化。
  规划使用视觉物体位置与 yaw，不读取隐藏物体真值决定目标。目标超过 2 秒未更新则保持。
- 双手远离物体时完整跟随。只有机械臂指垫靠近视觉物体 10 cm 内且出现有效闭合意图，
  才尝试 B 的视觉抓取候选；仅所选臂切到视觉接近/下降/闭爪，另一臂继续人体跟随。
- 只有所选两根手指与物体的预期接触被放行；手臂、桌面、躯干和另一臂约束仍在。
  确认夹持后有基于人体相对腕部增量的受限搬运分支；张手交回跟随。
  目前不把该分支宣称为已验收的搬运/安全放置。

## 本次 p2a_test1 实测

记录：`openarm_cloth/outputs/ab_motion/p2a_test1`。

| 检查项 | 结果 |
|---|---|
| 源视频 | 612 帧，约 20.834 秒 |
| 物理控制记录 | 417 帧，20 Hz，源帧范围 0–611 |
| 两臂/夹爪 | 全部 14 个臂关节均有运动范围，两个夹爪独立输出 |
| B 视觉目标可用 | 417/417 控制帧 |
| 双臂物理接触 | 本次全部物理子步记录为 0 |
| 近物视觉辅助 | 触发 1 次，夹持确认 0 次 |
| 平均上臂/前臂/手掌方向误差 | 左约 6.6°/19.5°/33.8°；右约 14.2°/17.7°/35.7° |
| 控制计算耗时 P95 | 约 51.6 ms；不含离线人体识别，不是端到端实时指标 |

本次状态 `COMPLETED` 只表示整段执行结束，**不是抓取成功**。
原视频在视觉对齐后、完成夹持前张手，因此返回人体跟随。
仍有规划/绕行以及 3 帧不可行状态；无双臂接触不能推导为永不卡住或连续安全保证。
桌面托住方块、指垫夹持属于有意物理接触，不能将“无双臂接触”写成“所有物体零接触”。

完整回归共 175 项：172 通过、3 项因符号链接权限跳过，无失败。
新增 10 项专项检查覆盖全时间轴、独立双臂输入、源时钟、手腕数据保留、视觉与隐藏
物体坐标隔离、视觉缺失保持、安全初始化、有限接触权限、远处闭合不触发抓取、
回放跳帧/暂停/尾帧/重播。B 冻结清单的工作区、物理、相机、任务定义、执行器和
控制器六项结构对比保持一致；这是物理配置核验，不等于新融合行为已通过所有任务验收。

`episode.npz` 保存实际 qpos/qvel、关节/夹爪目标、对应源帧、人体有效标记、
视觉位置及来源、控制阶段、避障模式和方向误差。`result.json` 是指标，
`manifest.json` 是视频/场景校验信息，`comparison.mp4` 是可直接查看的回放。

## 尚未验收与下一步

当前接入的是现有视频，不是完整 RGB-D 抓取场景下的实时人体摄像头输入；
旧 `live` 模式仍是无桌面/方块的动作跟随诊断，不能冒充当前完整场景闭环。
红色已知尺寸方块的识别能力也不等于任意物体识别。

优先用有明确“接近方块—闭合—搬运—张开”的示范视频验证共享控制，并改善单目
腕部方向估计、姿态标定与可达空间映射。随后对同一输入多摆位重复运行，比较姿态误差、
目标/实际抖动、延迟和抓取成功率，再接实时摄像头。若要求持物自然绕障，仍需
加入物体随手运动的扫掠体、支撑面约束和双臂协作验证。

旧受约束小曲线抓取保留为 `-Mode video-skill`；其三摆位 PASS 不能替代这次整段
人体映射的抓取验收，见 [历史说明](AB_VIDEO_VISUAL_GRASP_CN.md)。
