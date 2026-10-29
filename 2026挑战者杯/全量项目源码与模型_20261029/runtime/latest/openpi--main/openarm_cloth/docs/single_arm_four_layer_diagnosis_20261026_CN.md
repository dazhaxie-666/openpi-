# 执行用户提供的单目前后排障方案

> 50 秒新现场记录已完成，定位与后续修改见
> [短时掌部尺度补偿及显示负载调整](single_arm_scale_preview_20261026_CN.md)。

依据用户提供的 `D:\openpi\RGB_MONOCULAR_MEDIAPIPE_TELEOP_TROUBLESHOOTING_CN.md`，
将“感知—映射—IK—执行”排查接入当前 B 仿真。保留现有掌部相对前后输入，
不恢复已被现场证实不可靠的 Pose 深度，也不把肩宽代理替换成伸手/收手信号。
肩宽能用于观察身体表观尺度变化，但肩膀不动时的手部前伸应由手部信号表达。

## 已落实

| 文档步骤 | 当前实现和边界 |
|---|---|
| 只测视觉 | `depth-diagnose --camera-only` 使用项目现有 MediaPipe Tasks API；显示并保存腕点 image/world XYZ、肩宽像素、可见度，R 重新取参考。无机器人/仿真初始化。 |
| 对齐各层记录 | 普通单臂运行也保存映射前目标、工作区裁剪后目标、超前限幅目标、实际 IK 目标、IK 结果和模拟执行器命令。每行保留同一控制时刻及来源时刻。 |
| 检查 IK | position 策略保存求解状态、求值次数、位置残差、局部/初始腕姿偏差和未限速的七关节解。几何接受与求解器收敛分别记录；保持时不沿用上一帧成功状态。 |
| 绕过视觉注入前后轴 | `--check-actuation` 在原 B 模型中注入 0、+2 cm、0、-2 cm、0，原碰撞检查和执行器生效。 |
| 绕过 IK 检查执行器 | 同一检查再给第一个关节 +0.01 rad 合法微小位置目标，原 PD/力矩限制不变。 |
| CSV 和曲线 | `--episode` 导出 `chain.csv`、`summary.json`、`chain.png`。记录各层信号及七关节目标/实际值，正运动学还原关节命令对应的 TCP。原数据不改写。 |
| 静止与单轴定量 | 新增 `depth-chain` 50 秒引导：静止、近/中/远、重复伸缩、左右、上下、末尾静止；按新来源观测统计有效率和静止 p95 波动。提示标签不当作动作真值。 |

没有重装依赖或改用旧 `mp.solutions.pose`，没有引入新的识别模型。
已实现的相对映射、显示镜像隔离、连续死区、滤波、工作区、速度与碰撞限制保持原行为。
ArUco/AprilTag、RGB-D 和双目属于后续硬件/佩戴选项，本轮未切换。

## 已执行的检查与结论

`outputs/depth_diagnostics/20261026_handbook_actuation/actuation_check.json`：

- ±2 cm 五位置检查全部通过；各位置保持后的三维位置误差最大约 0.46 mm。
- 绕过 IK 后，+0.010 rad 关节目标产生约 +0.0090 rad 实际变化，最大关节误差约 0.00103 rad。
- 仅为指定仿真工况，不是实体机械臂、视觉测距或真人动作精度验收。

`outputs/depth_diagnostics/20261026_handbook_live_000731/`：

- 对刚才真人录像的 320 个活动控制行，去重后有 307 个新来源观测，前后有效约 66.8%。
  主要保持原因是重投影不确定（64 个新来源）。按控制行统计的 212/320 与此口径不同。
- 七关节限速后命令对应的 TCP X 跨度约 146.1 mm，实际 TCP X 跨度约 145.7 mm；
  对齐曲线总体相随，执行链路并未把前后命令全部丢弃。这些跨度不是跟踪误差。
- 控制工作中位数约 73.1 ms，p95 约 104.1 ms。原记录物理推进中位数约 32.2 ms、
  UI 约 20.4 ms、同步录像约 6.0 ms；因此仅调识别/滤波不能解决全部卡顿。
- 旧记录没有保存送入 IK 的笛卡尔目标及求解残差，报告明确留空；不根据机械臂动了
  就补造 IK 成功或“控制器 ACK”。新记录补齐这些字段。模拟执行器无硬件 ACK。

另已对修复前的 `20261025_232741_221304` 导出同格式报告，
保存在 `outputs/depth_diagnostics/20261026_handbook_before_232741/`。
80 帧软件记录检查也确认新字段可保存、导出，79 次实际调用 IK 的记录均被几何接受；
该软件片段手部输入静止，不能用它证明运动识别精度。

新增回归覆盖原始特征单位、来源去重、历史字段缺失、保持状态不能伪装成新 IK 成功、
新引导时序。完整回归 B 489 项（3 项跳过，其余通过）、A 11 项通过，日志位于
`outputs/depth_diagnostics/full_regression_handbook.log`。

## 下一次现场操作

在主项目根目录运行：

```powershell
& .\run.ps1 -Mode single-arm -ScriptArgs '--diagnostic-recording','--diagnostic-protocol','depth-chain','--diagnostic-seconds','50'
```

先把右手、右腕和肩部清楚放进画面，再按 SPACE。前 5 秒保持不动；随后按提示
朝镜头伸出、回中、向身体收回，并在各位置停约 2 秒；肩膀尽量保持位置。
之后单独做左右、上下，最后保持不动。50 秒后自动保存退出，Q 可提前保存。
视频采用已有后台队列写入，不再使用普通运行的同步录像路径。
这次不需要重新采集 G/O，也不要求髋部入镜。

数据位于 `openarm_cloth/outputs/human_task/<本次时间>/`；保存后导出报告：

```powershell
& .\run.ps1 -Mode depth-diagnose -ScriptArgs '--episode','outputs/human_task/<本次时间>'
```

只检查视觉或重复执行器测试：

```powershell
& .\run.ps1 -Mode depth-diagnose -ScriptArgs '--camera-only','--seconds','35'
& .\run.ps1 -Mode depth-diagnose -ScriptArgs '--check-actuation'
```

上述相对路径由根入口解析到 `openarm_cloth/`；默认分析输出在
`outputs/depth_diagnostics/<时间>/`。CSV 全部保留控制行并标记重复来源，统计分开
报告新来源观测与控制周期。`command_fk` 是离线运动学计算，不能冒充新的动力学执行。

## 采用范围

本轮直接落实的是用户提供文档的分层诊断、对齐记录、小位移注入和定量录制方法，
未新增论文算法。既有 OAT 掌点投影的采用范围仍见
[前后输入修复](single_arm_depth_decoupling_20261026_CN.md)。
下一步依据这次有明确静止段和单轴段的数据，再调投影质量、死区或显示负载；
当前不以盲目放宽识别门限替代测量。
