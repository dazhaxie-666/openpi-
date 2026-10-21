# 自由动作偏差与异常转腕修复

## 直接查看新结果

默认指针已切换到本次修复的运动学记录，不用重新生成，执行原命令即可：

```powershell
Set-Location -LiteralPath 'D:\openpi\latest\openpi--main'
& .\run.ps1 -Mode fusion-replay
```

终端打印实际目录及 `reference=clean / execution=kinematic`，避免串看旧结果。
鼠标旋转/缩放，空格暂停，A/D 逐帧，`+/-` 变速，G 切换参考线，Q 退出。

本次固定目录与已导出 MP4：

```powershell
# 新运动学回放：检查映射，不代表执行器零误差
& .\run.ps1 -Mode fusion-replay -ScriptArgs '--recording','outputs\a_motion\p2a_test1_repaired_kinematic'

# 新动态对照：B 仿真执行器跟随新目标
& .\run.ps1 -Mode fusion-replay -ScriptArgs '--recording','outputs\a_motion\p2a_test1_repaired_dynamic'

# MP4 视角固定，原生三维窗口可以转动
Start-Process -FilePath 'D:\openpi\latest\openpi--main\openarm_cloth\outputs\a_motion\p2a_test1_repaired_kinematic\comparison.mp4'
```

## 修复内容

旧默认回放与 A 保存关节值一致，不等于与人体一致。旧轨迹包含历史避障修正；
缺少可靠手掌朝向时，腕部还会在位置 IK 的冗余自由度中转动。约 20.5 秒，
左臂输入不可信，旧参考仍出现 26–34°/帧的转腕。

现在默认 `clean` 模式：

1. 复用 A 人体方向观测与 RTMW/MediaPipe 连续开合，但不把旧最终关节值或历史避障
   结果作为动作目标。原文件不改，新参考另存 `clean_reference.npz`。
2. 按真实上臂/前臂中心线拟合，不把肩部安装偏置或腕部工具尖端当作人体骨段。
   先求手臂，再使用实际 B 模型和可靠手掌朝向独立求腕部。
3. 无可靠手部/身体输入时，不继续推断转腕。手掌突变需连续确认；加入旋转连续性、
   目标角速度、软加速度与限位处理，防止突然切换 IK 分支。
4. 3D 不可信但已有 RTMW 二维可靠时，仅补充可观测的图像平面运动，深度保持旧值；
   开头未知深度取中立值。无可用二维或三维输入时才保持该臂，另一臂独立运行。
   RTMW 的非米制 `model_xyz` 不进入 IK；文件哈希与源视频绑定。
5. 动态模式采用物理子步上的连续 PCHIP 目标和速度前馈，不再每 50 ms 跳一次位置、
   同时以零目标速度阻尼。B 抓取控制器、力矩限制和冻结物理参数不变。

回放显示输入模式 `3D`、`2D(depth held)`、`HOLD`，二维补充时绘制实际使用的 RTMW 点。
保留图像平面运动不等于恢复真实深度；缺少可靠手部时仍保持腕部。

## 同一 p2a_test1 的对照

| 指标 | 旧结果 | 本次修复 |
|---|---|---|
| 可信帧左臂上臂/前臂平均方向误差 | 17.70° / 9.32° | 0.09° / 0.10° |
| 可信帧右臂上臂/前臂平均方向误差 | 4.87° / 11.54° | 0.48° / 0.46° |
| 相邻帧最大腕关节变化 | 约 34.3° | 约 7.8° |
| 末尾左腕 | 大幅来回变化 | 无可靠手部，此段保持角度 |
| 动态平均关节跟随偏差 | 1.88° | 0.60° |
| 动态平均末端跟随偏差（左/右） | 3.03 / 3.55 cm | 1.08 / 1.05 cm |
| 动态末端偏差 P95（左/右） | 约 8.9 / 8.9 cm | 3.16 / 3.80 cm |

方向误差相对 A 识别方向，不是人工标注的人体真值精度；新骨段定义在新旧关节轨迹上
统一计算。动态偏差相对各自目标，新旧目标和控制器均改变，是完整流程对照，不是
控制器单变量实验。运动学零适配误差也不等于人体零误差。

共 612 源帧、约 20.834 秒；动态记录 417 帧、20 Hz。二维补充左 59 帧、右 34 帧；
左臂仍有 5 帧无可靠补充而保持。单目深度、遮挡与机器人可达范围仍限制精度，不能
保证任意动作完全一致。本轮不验收碰撞、抓取或实体机器人安全。

完整回归 197 项：194 通过、3 项因符号链接权限跳过，无失败。检查覆盖历史避障隔离、
低可信输入、腕部翻转/速度/限位、二维输入不使用非米制三维坐标、前馈力矩限制与
跟随时钟。原生查看器启动/退出、整段 MP4 与开头/中段/末尾画面对照已验证。

## 以后运行

```powershell
# 默认重建 clean 参考；已有结果自动保留
& .\run.ps1 -Mode fusion -ScriptArgs '--check'

# 动态模式，不替换默认运动学回放指针
& .\run.ps1 -Mode fusion -ScriptArgs '--check','--execution','dynamic','--no-set-default'

# 仅用于旧问题对照，不是当前默认
& .\run.ps1 -Mode fusion -ScriptArgs '--check','--reference-mode','legacy','--execution','dynamic','--tracking','step','--no-set-default'
```

每个新目录保存参考 NPZ、源文件哈希绑定、`repair_report.json`、`episode.npz`、
`manifest.json`、`result.json`。旧记录、源视频与原 A NPZ 保持不变。
