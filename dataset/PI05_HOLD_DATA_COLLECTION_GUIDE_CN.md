# OpenArm π0.5 首个技能数据采集指南

更新日期：2026-11-17

工作目录示例：`D:\openpi_handoff\dataset`

首个训练目标：**靠近红色方块 → 抓取 → 抬升 → 静止保持 3 秒 → 原位放下 → 撤离**

本指南只针对右臂、单个红色方块、MuJoCo 仿真环境中的 `Hold` 任务。第一版不采集横向搬运、放入绿色区域、失败恢复、换手或多物体任务。

---

## 0. 交给其他采集人员前的环境部署

采集负责人必须先把运行环境部署并验证好，再把机器交给采集人员。采集人员不应在采集当天临时安装依赖、更新代码或修改场景。

### 0.1 操作系统与目录结构

首次安装先按 [新电脑交接说明](HANDOFF_CN.md) 从 main 克隆、安装依赖并下载模型。本指南中的固定路径以克隆到 `D:\openpi_handoff` 为例；如果使用其他 ASCII 路径，替换此前缀。

```text
D:\openpi_handoff\
├─ run.ps1
├─ .runtime.local.json         每台电脑独立设置，不提交
├─ openarm_cloth\             生产仿真、人体识别和控制
├─ pose2arm-main\models\      setup_models.py 下载识别权重
└─ dataset\
   ├─ dataset.ps1
   ├─ requirements.txt
   ├─ config\
   ├─ scripts\
   ├─ tests\
   └─ calibration\ / raw\ / reports\ / logs\ / snapshots\ / exports\
```

- `dataset` 位于仓库根目录中，与 `run.ps1` 同级；启动器按脚本位置寻找生产代码。
- 整个仓库必须使用纯英文/ASCII 路径。采集位置应使用本地磁盘。
- 旧本机 `D:\openpi\dataset` 外置工作区仍可独立使用，其历史数据不在新克隆中。
- 当前只需普通人体 RGB 摄像头；胸部和左右腕 RGB-D 都由 MuJoCo 仿真生成。

### 0.2 Python环境

采集基线为 Windows / Python 3.12，固定版本见 [requirements.txt](requirements.txt)。在新电脑的PowerShell中执行：

```powershell
Set-Location -LiteralPath 'D:\openpi_handoff'
conda create -n openpi python=3.12 -y
conda activate openpi
python -m pip install -r .\dataset\requirements.txt
python -m pip check
python .\openarm_cloth\setup_models.py

$pythonExe = (python -c 'import sys; print(sys.executable)').Trim()
@{python=$pythonExe} | ConvertTo-Json | Set-Content -LiteralPath '.runtime.local.json' -Encoding UTF8
```

已有同名环境时先核对依赖；解释器路径必须是这台电脑实际使用的环境。`setup_models.py`从官方站点下载MediaPipe手部和人体权重到`pose2arm-main\models`；离线电脑可领取四个官方`.task`文件后运行模型检查，具体文件清单见[交接说明](HANDOFF_CN.md)。

也可以在单次命令中显式指定同一环境：

```powershell
& '.\dataset\dataset.ps1' -Mode Check -Python $pythonExe
```

启动器按本机`.runtime.local.json`或环境变量`OPENPI_PYTHON`选择解释器；不提交本机配置，不照抄研发电脑的Anaconda路径。每次新开终端要先激活openpi环境，并切换到仓库目录。

### 0.3 环境首次验收

部署完成后，由环境负责人先执行：

```powershell
Set-Location -LiteralPath 'D:\openpi_handoff'
& '.\run.ps1' -Mode check -ScriptArgs '--check-models'

Set-Location -LiteralPath 'D:\openpi_handoff\dataset'
& '.\dataset.ps1' -Mode Check
& '.\dataset.ps1' -Mode Test
```

必须确认：

- 输出的 `B Python` 是预期的 `openpi` 环境；
- MuJoCo场景和模型可以加载；
- MediaPipe模型可以加载；
- 胸部、左腕、右腕三路机器人RGB-D可以渲染；
- 数据采集接口、回放接口和质量审核测试全部通过；
- 显卡驱动和OpenGL可以正常打开MuJoCo窗口。

首次验收失败时，不要让采集人员通过修改脚本“绕过检查”。

### 0.4 摄像头和现场条件

需要一台稳定安装的前置或USB人体摄像头。当前验证基线为：

- 设备编号通常为 `0`；其他电脑可能是 `1` 或更高编号；
- 建议分辨率 1280×720；
- 建议摄像头读取能力约30 FPS；
- 摄像头固定不动，操作者正对摄像头；
- 肩膀、右肘、右腕和右手保持在画面内；
- 光照均匀，避免强逆光和手部运动模糊；
- 关闭会议软件、浏览器摄像头页面和其他会占用摄像头的程序。

检查摄像头编号：

```powershell
& 'D:\openpi_handoff\dataset\dataset.ps1' -Mode Camera -Camera 0
# 如果失败，再尝试：
& 'D:\openpi_handoff\dataset\dataset.ps1' -Mode Camera -Camera 1
```

确定正确编号后，整批数据保持同一摄像头和同一机位。

### 0.5 操作者编号与个人标定

每位采集人员必须有唯一且稳定的编号，例如：

```text
operator01
operator02
operator03
```

采集命令中的 `-Operator` 必须填写真实操作者，不能让所有人都使用默认的 `operator01`。

当前采集器读取：

```text
D:\openpi_handoff\dataset\calibration\pinch.json
```

注意：

- 该标定包含个人捏合开度，不能默认在不同操作者之间通用。
- 更换操作者、移动摄像头或显著改变人与相机距离后，需要重新标定。
- 多人共用同一台电脑时，应在采集记录中注明当前 `pinch.json` 属于谁，并保留每位操作者的标定备份。
- 不允许在同一session中途更换操作者或标定文件。
- 按 `C` 只会重新对齐人体与机械臂的相对映射，不等于重新生成个人捏合标定。

首次标定使用：

```powershell
& 'D:\openpi_handoff\dataset\dataset.ps1' -Mode Calibrate -Camera 0 -Operator operator02
```

在摄像头窗口中，不按 SPACE 开始任务；右手捏合按 G、自然张开按 O，各采10次，每次保持1.2秒。看到 CALIBRATION READY 后按 Q 保存。确认 `dataset\calibration\pinch.json` 已生成，并在标定记录中登记操作者。换人时负责人备份并移开旧标定，再重新采集端点；程序不会从其他工作区自动复制个人标定。

### 0.6 数据具体保存位置

采集命令不会把数据保存到命令所在的临时目录，而是统一写入 `D:\openpi_handoff\dataset`。

#### 原始数据

```text
D:\openpi_handoff\dataset\raw\<数据集名>\batch_XXX\<时间_session>\
```

例如正式Core数据：

```text
D:\openpi_handoff\dataset\raw\openarm_grasp_hold_return_p02_core30_v1\batch_001\20261113_143000_xxxxxx\
```

每个session通常包含：

```text
manifest.json              运行环境、机器人、相机和动作合同
collection.json            阶段、任务、操作者、批次和源码哈希
episode.npz                状态、动作目标、时间戳和控制记录
shared_rgbd\               胸部、左腕、右腕RGB与深度数据
shared_frames.jsonl        图像与控制行索引
shared_phases.json         plan/close/lift/hold/lower/release等阶段事件
shared_trials.json         每条试次的开始、结束和物理结果
events.json                操作事件
scene.mjb                  本次回放使用的MuJoCo场景
raw_action_replay.json     运行Replay后生成的一致性报告
review.json                人工接受/拒绝记录
```

原始session一旦写完，不得手工改动作、图像、试次边界或物理成功标签。

采集程序正常结束时会在终端输出 `SAVED: <session路径>`。采集人员必须把这条绝对路径抄入交接表；如果没有看到 `SAVED`，应将该次运行标记为异常并联系负责人检查，不能自行拼接残留文件。

#### 自动审核和回放报告

```text
D:\openpi_handoff\dataset\reports\<数据集名>\batch_XXX\
```

例如：

```text
D:\openpi_handoff\dataset\reports\openarm_grasp_hold_return_p02_core30_v1\batch_001\
```

#### 环境与摄像头前置检查

```text
D:\openpi_handoff\dataset\reports\preflight\
```

#### 命令日志

```text
D:\openpi_handoff\dataset\logs\
```

#### 采集参数、配置和源码哈希快照

```text
D:\openpi_handoff\dataset\snapshots\<数据集名>\batch_XXX\
```

#### 后续LeRobot/π0.5转换输出

```text
D:\openpi_handoff\dataset\exports\
```

`exports` 只能保存从审核批准样本生成的版本化导出，不能反向覆盖 `raw`。

### 0.7 多人采集后的数据交接规则

其他采集人员完成一个session后，应把以下信息交给负责人：

- 操作者编号；
- 电脑编号和摄像头编号；
- 数据集名、批次、位置和session绝对路径；
- 尝试数、物理成功数、候选数；
- Audit和Replay结果；
- 人工接受/拒绝结论及备注；
- 当天是否改过代码、标定、场景或相机位置。

交接和备份时：

- 必须复制完整session目录，不能只复制 `episode.npz` 或视频；
- 必须同时复制对应的 `reports`、`logs` 和 `snapshots`；
- 保持原来的数据集名、`batch_XXX` 和session目录层级；
- 不要把多个session文件直接混到同一目录；
- 不要在采集进行时同步到移动硬盘或网络盘；
- 复制完成后比较文件数量和总字节数，再由负责人合入主数据集；
- 未完成Audit、Replay和Review的数据只能标为“待审核”，不能计入30条正式目标。

建议每位采集人员使用下面的交接记录：

| 日期 | 操作者 | 电脑 | Camera | 数据集/批次 | Pose | Session路径 | 尝试 | 候选 | Replay | Review | 备注 |
|---|---|---|---:|---|---|---|---:|---:|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |

---

## 1. 数据阶段和用途

所有阶段由 [config/stages.json](config/stages.json) 定义。不同阶段必须存入不同目录，不能通过改目录名把试采数据变成训练数据。

| 阶段 | 命令模式 | 数据集目录 | 目标 | 能否训练 |
|---|---|---|---:|---|
| 软件自检 | `Smoke` | `openarm_grasp_hold_return_p00_smoke_v1` | 验证采集与回放链路 | 否 |
| 试采 | `Pilot` | `openarm_grasp_hold_return_p01_pilot5_v1` | 验证操作和数据质量 | 否 |
| 正式核心集 | `Collect` | `openarm_grasp_hold_return_p02_core30_v1` | 30 条审核通过的完整示范 | 是 |
| 扩展覆盖 | `Coverage` | `openarm_grasp_hold_return_p03_coverage_v1` | 首轮训练后按失败类型补采 | 是 |
| 恢复示范 | `Recovery` | `openarm_grasp_hold_return_p04_recovery_v1` | 后续采集成功恢复轨迹 | 是 |
| 隔离参考 | `Eval` | `openarm_grasp_hold_return_p05_eval5_v1` | 5 条未见条件参考 | 否 |

本机历史12条补录数据属于 `Pilot`，没有随新克隆上传。它们可用于验证转换、回放和数据格式，但不能进入正式 π0.5 训练。

---

## 2. 正式采集前的阻断检查

以下任一项不满足时，不要开始 `Collect`。

### 2.1 固定任务定义

首版统一使用以下任务语义：

> Grasp the red cube, lift it, hold still for 3 seconds, then place it back at its starting position.

对应中文语义：

> 抓住红色方块，将它抬起，稳定保持3秒，然后放回起始位置。

必须确认：

- `task_profile` 为 `hold`。
- `hold_seconds` 为 `3.0`。
- `return_to_source` 为 `true`。
- `source_tolerance_m` 为 `0.03`。
- 不能使用“放入绿色区域”的任务标签。

**已修正（2026-11-17）：** 新采集的 `manifest.json.task_instruction` 按实际任务和保持时长生成，与 Hold 合同一致。旧原始数据不重写；采完第一条仍需核对这两个字段。

采完第一条后检查：

```powershell
$manifest = Get-Content -LiteralPath (Join-Path $session 'manifest.json') -Raw | ConvertFrom-Json
$context  = Get-Content -LiteralPath (Join-Path $session 'collection.json') -Raw | ConvertFrom-Json
$manifest.task_instruction
$context.task_profile
$context.task_contract
```

预期：任务指令描述原位抓起、保持和放回，`task_profile` 输出 `hold`。

### 2.2 固定采集版本

一个正式批次期间不要修改以下内容：

- `D:\openpi_handoff\openarm_cloth` 控制代码；
- `dataset\scripts` 采集和审核脚本；
- `dataset\config\stages.json`；
- MuJoCo 场景、相机位姿和标定；
- 控制频率、图像频率、机械臂和夹爪动作定义。

若必须修改，停止当前批次，完成测试后使用新批次或新数据版本继续，不要静默混合。

### 2.3 检查操作者和标定

- 第一版固定使用 `operator01`。
- 人体相机位置、焦距、分辨率和操作者站位保持不变。
- 确认 `D:\openpi_handoff\dataset\calibration\pinch.json` 存在。
- 更换操作者、移动人体相机或明显改变距离后，必须重新标定并发布新批次。

### 2.4 检查磁盘空间

```powershell
Get-PSDrive -Name D | Select-Object Used, Free
```

建议正式采集前至少保留 30 GB 空间。不要在采集过程中压缩、移动或同步正在写入的 session。

---

## 3. 每个采集日开始前

在 PowerShell 中执行：

```powershell
Set-Location 'D:\openpi_handoff\dataset'

# 环境、场景、依赖、磁盘和三路机器人相机检查
& '.\dataset.ps1' -Mode Check

# 人体摄像头检查；默认摄像头编号为0
& '.\dataset.ps1' -Mode Camera -Camera 0

# 数据合同与控制逻辑回归测试
& '.\dataset.ps1' -Mode Test

# 查看采集前数量基线
& '.\dataset.ps1' -Mode Summary
```

通过标准：

- `Check` 无错误退出；
- 人体摄像头能稳定打开并读取图像；
- 胸部、左腕、右腕三路 RGB-D 可用；
- 全部自动测试通过；
- 当前数量与上一轮记录一致。

软件版本、场景或相机配置发生变化后，额外运行一次 Hold 软件自检：

```powershell
& '.\dataset.ps1' -Mode Smoke -Task Hold -Steps 1500
```

Smoke 数据永远不能进入训练。

---

## 4. 单条示范的标准动作

### 阶段 A：复位与准备

- 等待界面显示下一条试次可以开始。
- 机械臂、方块和夹爪必须处于复位状态。
- 操作者保持自然站姿，右手张开。
- 不要在倒计时结束前靠近或捏合。

### 阶段 B：靠近

- 用右手控制右臂，以连续、缓慢、近似直线的路径接近方块。
- 保持方块在胸部和右腕视野中。
- 不要快速来回修正，不要突然改变方向。
- 不要在距离方块较远时提前捏合。

### 阶段 C：抓取

- 进入抓取范围后，做一次明确并持续的捏合动作。
- 等待系统完成 `plan → approach → descend → close`。
- 自动抓取期间不要反复张合手指或大幅移动手腕。
- 若识别丢失、方块被遮挡或抓取失败，保留失败记录并重采，不要强行把该段标成成功。

### 阶段 D：抬升

- 确认夹爪抓稳后，让系统完成 `lift`。
- 目标抬升高度约 6–8 cm。
- 不进行横向搬运，不朝绿色区域移动。

### 阶段 E：静止保持

- 抬升完成后保持右手和机械臂稳定至少 3 秒。
- 这段近零动作是训练“停止/保持”的有效数据，后续转换时必须保留。
- 不要为了“制造动作”而晃动机械臂。

### 阶段 F：原位放下

- 保持完成后，做一次新的张手动作并持续约 0.35 秒。
- 系统执行 `lower → release → retract → verify`。
- 必须先让方块接触原始支撑面，再打开夹爪。
- 方块放回误差必须不超过 3 cm。

### 阶段 G：结束与下一条

- 等待当前试次显示完成。
- 在3秒受保护复位倒计时内恢复手部姿势。
- 不要手动跳过自动复位，也不要把复位帧当成下一条示范。

建议单条完整示范控制在约 18–30 秒。超过时间通常意味着停顿、识别丢失或过多修正，应重点检查。

---

## 5. 第一阶段：采集30条正式 Core 数据

推荐分成3个位置批次、6个独立 session，每个 session 采5次尝试。每个位置命令执行两次。

以下命令以 `operator01` 和摄像头 `0` 为示例。其他采集人员必须把 `-Operator operator01` 改成自己的唯一编号；如果前置检查确认摄像头不是设备0，还要在每条采集命令末尾增加 `-Camera 1`（或实际编号）。

| 批次 | 位置 | Session数 | 每个Session尝试数 | 合格目标 |
|---|---|---:|---:|---:|
| `batch_001` | center | 2 | 5 | 10 |
| `batch_002` | left | 2 | 5 | 10 |
| `batch_003` | right | 2 | 5 | 10 |

注意：`Trials` 是尝试数，不是成功数。若某次失败，需要在同一位置继续补采，直到获得目标数量的审核批准示范。

### 5.1 中心位置

下面命令执行两次：

```powershell
& 'D:\openpi_handoff\dataset\dataset.ps1' `
  -Mode Collect -Task Hold -Pose center `
  -Trials 5 -ResetSeconds 3 -Seconds 0 `
  -Batch 1 -Operator operator01
```

### 5.2 左侧位置

下面命令执行两次：

```powershell
& 'D:\openpi_handoff\dataset\dataset.ps1' `
  -Mode Collect -Task Hold -Pose left `
  -Trials 5 -ResetSeconds 3 -Seconds 0 `
  -Batch 2 -Operator operator01
```

### 5.3 右侧位置

下面命令执行两次：

```powershell
& 'D:\openpi_handoff\dataset\dataset.ps1' `
  -Mode Collect -Task Hold -Pose right `
  -Trials 5 -ResetSeconds 3 -Seconds 0 `
  -Batch 3 -Operator operator01
```

正式数据保存到：

```text
D:\openpi_handoff\dataset\raw\openarm_grasp_hold_return_p02_core30_v1\batch_001\<session>
D:\openpi_handoff\dataset\raw\openarm_grasp_hold_return_p02_core30_v1\batch_002\<session>
D:\openpi_handoff\dataset\raw\openarm_grasp_hold_return_p02_core30_v1\batch_003\<session>
```

---

## 6. 每个Session采集后立即处理

不要等30条全部采完后才审核。每个 session 结束后立即执行以下步骤。

### 6.1 找到刚完成的Session

```powershell
$coreRoot = 'D:\openpi_handoff\dataset\raw\openarm_grasp_hold_return_p02_core30_v1'
$session = (Get-ChildItem -LiteralPath $coreRoot -Recurse -File -Filter manifest.json |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1).DirectoryName

$session
```

人工确认输出路径确实是刚完成的 session，禁止审核错目录。

### 6.2 自动质量审核

```powershell
& 'D:\openpi_handoff\dataset\dataset.ps1' -Mode Audit -Session $session
```

审核结果保存在：

```text
D:\openpi_handoff\dataset\reports\openarm_grasp_hold_return_p02_core30_v1\<batch>\<session>.audit.json
```

### 6.3 动作回放

先运行无窗口一致性回放：

```powershell
& 'D:\openpi_handoff\dataset\dataset.ps1' -Mode Replay -Session $session
```

需要可视化查看时：

```powershell
& 'D:\openpi_handoff\dataset\dataset.ps1' -Mode Replay -Session $session -Visible
```

通过标准：

- `verdict` 为 `PASS`；
- `max_qpos_error` 为 `0.0`；
- `max_qvel_error` 为 `0.0`；
- 自动复位次数与试次边界一致；
- 回放中没有掉落、错误释放或非预期碰撞。

### 6.4 人工逐条审核

接受示例：

```powershell
& 'D:\openpi_handoff\dataset\dataset.ps1' `
  -Mode Review -Session $session -Trial 1 `
  -Decision accept -Operator operator01 `
  -Note '完整抓取、抬升、稳定保持3秒并原位释放；运动连续，无明显碰撞'
```

拒绝示例：

```powershell
& 'D:\openpi_handoff\dataset\dataset.ps1' `
  -Mode Review -Session $session -Trial 2 `
  -Decision reject -Operator operator01 `
  -Note '靠近阶段反复修正且抓取后出现明显滑移，需要重采'
```

每条候选都必须单独审核。不要为了凑数量接受有明显问题的示范。

### 6.5 再次查看汇总

```powershell
& 'D:\openpi_handoff\dataset\dataset.ps1' -Mode Summary
```

只有 `training_approved_successes` 计入正式30条目标。

---

## 7. 数据质量门槛

### 7.1 自动硬门槛

| 指标 | 当前门槛 |
|---|---:|
| 三路相机目标频率 | 10 Hz |
| 每路相机最低实际频率 | 8 Hz |
| 最大相机间隔 | 0.25 s |
| 最大控制间隔 | 0.20 s |
| 试次内允许暂停行数 | 0 |
| 源位置与目标最小距离 | 0.08 m |
| 稳定持物时间 | ≥3.0 s |
| 原位放回误差 | ≤0.03 m |
| 动作回放 | 必须PASS |

### 7.2 必须人工查看的问题

- `manual target speed spike`；
- `frequent manual target reversals`；
- 手腕或机械臂明显抖动；
- 接近路径反复往返；
- 方块在夹爪内滑动；
- 抬升高度不足；
- 空中提前打开夹爪；
- 方块没有回到原始支撑位置；
- 胸部或右腕关键阶段看不到方块；
- 试次过长并包含无意义停顿。

运动警告不是自动失败，但必须在视频和可视回放中确认后才能接受。

---

## 8. 何时停止当前批次

出现以下情况应立即停止，不要继续大量采集：

- `Check`、`Camera` 或 `Test` 失败；
- 人体相机、胸部相机或任一腕部相机持续缺帧；
- 第一条 `manifest.json.task_instruction` 与 Hold 任务不一致；
- 源码哈希在同一批次中发生变化；
- 连续两条出现相同的抓取、回放或识别故障；
- Replay 不是 `PASS`；
- qpos/qvel 回放误差不为0；
- 自动复位行进入训练片段；
- 操作者明显疲劳，动作开始频繁反向修正；
- 磁盘空间不足或异步写盘报错。

失败和取消记录保留用于排查，不要手动删除或改成成功标签。

---

## 9. Core30完成后的训练前整理

30条是首次验证单技能训练的起点，不代表已经具备跨场景泛化能力。

### 9.1 冻结正式样本清单

只选择同时满足以下条件的试次：

- 位于 `openarm_grasp_hold_return_p02_core30_v1`；
- `candidate=true`；
- Replay 已验证；
- 人工 `Review=accept`；
- `training_approved=true`；
- 不属于 Smoke、Pilot、Eval 或失败记录。

冻结样本ID、session、trial、源码哈希、质量合同摘要和训练/验证划分。原始目录保持只读。

### 9.2 转换为π0.5输入

首版建议使用：

- 图像：胸部 RGB + 右腕 RGB；
- 左腕 RGB 和三路深度继续保留用于审核，不作为首版模型输入；
- `state`：右臂7关节角 + 1个归一化夹爪值，共8维；
- `action`：7个归一化关节速度 + 1个夹爪指令，共8维；
- Prompt：统一使用本指南第2.1节的 Hold 指令；
- 观测按约10 Hz图像时间戳对齐；
- 从30 Hz控制记录构造未来动作块；
- 保留任务内部3秒稳定保持段，删除试次外复位和无关空闲行。

不能把原始双臂18维绝对动作直接送入 π0.5。

### 9.3 划分训练与验证集

必须按 session 划分，不能随机拆帧。建议：

- 约24条训练；
- 约6条验证；
- 同一 session 的所有试次只能出现在同一划分中。

先用3–5条正式 Core 数据做过拟合测试，确认 LeRobot 转换、归一化、训练、推理和本地执行闭环；通过后再训练完整 Core30。

---

## 10. 第二阶段：隔离参考数据

Core30完成并冻结后，再采5条 `Eval` 参考示范。Eval始终不能进入训练。

示例：

```powershell
& 'D:\openpi_handoff\dataset\dataset.ps1' -Mode Eval -Task Hold -Pose near  -Trials 1 -Batch 1 -Operator operator01
& 'D:\openpi_handoff\dataset\dataset.ps1' -Mode Eval -Task Hold -Pose far   -Trials 1 -Batch 1 -Operator operator01
& 'D:\openpi_handoff\dataset\dataset.ps1' -Mode Eval -Task Hold -Pose yaw0  -Trials 1 -Batch 1 -Operator operator01
& 'D:\openpi_handoff\dataset\dataset.ps1' -Mode Eval -Task Hold -Pose yaw60 -Trials 1 -Batch 1 -Operator operator01
& 'D:\openpi_handoff\dataset\dataset.ps1' -Mode Eval -Task Hold -Pose random -Seed 20261113 -Trials 1 -Batch 1 -Operator operator01
```

这些记录是隔离参考示范，不等同于模型自主评测。模型训练完成后，还需要单独执行多次自主 rollout 并统计成功率。

---

## 11. 第一版训练完成后的扩采原则

不要预先盲目采集大量同质数据。先根据模型真实失败类型再进入下一阶段：

- 看不准边缘位置：补 `Coverage` 的 near/far/left/right；
- 对方块角度敏感：补 yaw0/yaw60；
- 抓到后滑落：补不同接近方向和夹持稳定性数据；
- 不会停止：增加合格的稳定保持段，但不能增加试次外无意义 idle；
- 不会正确放下：补原位下降、接触支撑后释放的成功示范；
- 发生失败后无法继续：另建 `Recovery` 数据，不与干净 Core 混为同一来源。

新增场景、操作者、相机或任务语义时，应发布新版本，不要直接混入 `core30_v1`。

---

## 12. 每日收尾清单

- [ ] 每个新 session 已执行 Audit。
- [ ] 每个新 session 已执行 Replay，结果为PASS。
- [ ] 每个候选 trial 已人工接受或拒绝。
- [ ] 失败和取消记录仍被保留。
- [ ] Summary数量与人工记录一致。
- [ ] 新数据未混入 Pilot、Smoke 或 Eval。
- [ ] 原始数据没有被重命名、平滑或覆盖。
- [ ] 日志、报告、review.json和源码哈希已保存。
- [ ] 磁盘空间足够下一次采集。

本次新电脑部署与软件采集回放验证见 [VERIFICATION_20261117.md](VERIFICATION_20261117.md)。本机历史补录报告与原始数据另行保存，未包含在代码克隆中。
