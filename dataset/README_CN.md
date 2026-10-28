# 数据集采集工作区：测试、正式采集与操作规范

更新：2026-10-28
本机工作区：`D:\openpi\dataset`
GitHub 仓库内目录：`<仓库>\dataset`
统一入口：`dataset\dataset.ps1`

下文使用 `D:\openpi\dataset\dataset.ps1` 展示当前机器可直接复制的命令；在其他电脑克隆 GitHub 仓库后，把该前缀替换为 `<仓库路径>\dataset\dataset.ps1`，或在仓库根目录运行 `& .\dataset\dataset.ps1 ...`。

## 1. 先了解本轮准备了什么

新增加的采集脚本、配置、测试、标定副本、原始数据、日志与审核结果都在本文件夹。机器人仿真、人体识别与执行器继续复用生产仓库的 `openarm_cloth`，不复制另一套控制器。本机外置工作区会自动找到 `D:\openpi\latest\openpi--main`；从 GitHub 克隆后则直接使用仓库根目录。

生产入口只增加了可选的采集扩展接口。使用本入口时，双相机以 **10 Hz 目标频率**保存动作执行前的 RGB-D，并对应到保存的控制行；原有 B 视觉判断仍按其原有节奏运行。编码异步写盘，机器人控制目标仍为 30 Hz。实际频率、缺口和写盘错误由审核报告给出，不把目标值当作实测值。

本轮保留原始、已经被 B 接受的动作目标，不对保存后的轨迹偷偷平滑。新增了手动阶段目标速度突变、反向修正和动作警告占比的检查，帮助发现抖动与受限操作。

当前是 **MuJoCo 仿真共享示教数据源**，还不是已完成的 LeRobot 训练集；π0.5 的正式转换、训练和自主客户端属于后续工作。

```text
D:\openpi\dataset\
  dataset.ps1                 统一命令入口
  README_CN.md                本操作指南
  config\stages.json          阶段名称、目标数量、初始条件与质量阈值
  scripts\                    采集、前置检查、审核与回放工具
  tests\                      数据隔离、定时采样与控制回归测试
  calibration\pinch.json      从现有标定复制的本工作区标定；后续独立保存
  raw\<数据集名>\batch_001\<session>\   原始记录，按批次和 session 保存
  reports\<数据集名>\batch_001\         对应批次的审核、回放、自检报告
  reports\preflight\          环境与相机检查报告
  logs\                      每次命令的日志
  snapshots\<数据集名>\batch_001\       采集参数、配置与源码哈希
  exports\                   预留给后续训练数据转换
  tmp\                       本工作区运行临时文件
```

旧的历史数据保留在原处。本文件夹中的新数据不会覆盖旧记录，也不会自动上传 GitHub。

## 2. 不同阶段的数据集名称

### 2.1 当前优先技能：抓取、保持 3 秒、原位放下

使用 `-Task Hold` 时，系统不会要求把方块搬到绿色区域。机器人抓取并抬起后会自动锁定当前位置，界面倒计时保持 3 秒；倒计时结束后，操作者保持手腕不动并张手约 0.35 秒，机器人自动回到抓取源位置下降并释放。

该任务使用独立目录，绝不与下面的搬运任务混合：

| 模式 | 数据集目录名 | 建议数量 | 训练用途 |
|---|---|---:|---|
| `Pilot -Task Hold` | `openarm_grasp_hold_return_p01_pilot5_v1` | 5 条成功试采 | 否 |
| `Collect -Task Hold` | `openarm_grasp_hold_return_p02_core30_v1` | 30 条审核通过的成功示范 | 是 |
| `Eval -Task Hold` | `openarm_grasp_hold_return_p05_eval5_v1` | 5 条隔离参考 | 否 |

先采 5 条试采；全部通过物理判定、回放和人工查看后，再采 30 条正式训练数据。30 条是用于首次验证单一技能训练的适量起点，不代表已经获得跨场景泛化能力。

### 2.2 原有搬运到绿色目标区任务

| 命令模式 | 数据集目录名 | 用途与数量 | 能否进入正式训练 |
|---|---|---|---|
| `Smoke` | `openarm_pickplace_p00_smoke_v1` | 软件意图自检与动力学回放 | 否，明确标记 synthetic |
| `Pilot` | `openarm_pickplace_p01_pilot_v1` | 约 20 条试采，验证操作、时间与格式 | 否，先隔离验证 |
| `Collect` | `openarm_pickplace_p02_core100_v1` | 100 条审核合格的基础成功示范 | 审核通过后作为候选训练源 |
| `Coverage` | `openarm_pickplace_p03_coverage170_v1` | 再采约 170 条，补位置、朝向与边界覆盖 | 同上 |
| `Recovery` | `openarm_pickplace_p04_recovery30_v1` | 约 30 条经过审核的成功恢复示范 | 同上，单独保留来源 |
| `Eval` | `openarm_pickplace_p05_eval_holdout_v1` | 隔离的参考示范/场景，不混入训练 | 否 |

正式目标合计为 **100＋170＋30＝300 条合格成功示范**，试采和软件测试不计入这个数量。一次失败尝试仍会保存，但不能按一条成功示范计数。

`Eval` 当前记录的是隔离的共享示教参考，不是 π0.5 自主验收。后续接好模型客户端后，自主结果也应独立命名和保存。

每个 session 会保存 `collection.json`，明确阶段、用途、操作者、初始条件、源码哈希以及是否允许用于训练。不要仅靠手工改目录名，把试采或自检冒充正式数据。

### 用 1、2、3 区分后续采集批次

`Collect` 的正式数据集名称是 **`openarm_pickplace_p02_core100_v1`**。在同一阶段内，用 `-Batch 1`、`-Batch 2`、`-Batch 3` 区分不同批次；省略时默认是第 1 批。

```powershell
# 第 1 批
& 'D:\openpi\dataset\dataset.ps1' -Mode Collect -Pose center -Batch 1
# 第 2 批
& 'D:\openpi\dataset\dataset.ps1' -Mode Collect -Pose center -Batch 2
# 第 3 批
& 'D:\openpi\dataset\dataset.ps1' -Mode Collect -Pose center -Batch 3
```

对应保存位置：

```text
raw\openarm_pickplace_p02_core100_v1\batch_001\<时间_session>\
raw\openarm_pickplace_p02_core100_v1\batch_002\<时间_session>\
raw\openarm_pickplace_p02_core100_v1\batch_003\<时间_session>\
```

同一批可以重复执行，每次仍生成独立 session，不覆盖之前的数据。批次编号由你指定，不自动跳到下一批；第 1、2、3 批也不是试采、正式、恢复阶段的编号。阶段用于区分用途，批次用于区分同一用途下不同轮的采集。

历史未分批的 session 原地保留，审核和汇总兼容旧路径。`Summary` 同时显示阶段总量和各批次数量；正式阶段的 100 条目标是各批次合计的合格示范数量，不是每一批都要 100 条。

批次编号、自动场景复位和分段边界已纳入完整回归测试，覆盖同批次不覆盖、跨批次目录隔离、历史记录汇总，以及重置行不产生虚假试次。

## 3. 采集前的测试命令

以下命令可直接复制到 PowerShell，不要求先切换当前目录。

### 3.1 环境、模型、相机标定与机器人 RGB-D 检查

```powershell
& 'D:\openpi\dataset\dataset.ps1' -Mode Check
```

它不会打开人的摄像头窗口。检查 B 场景、七关节/双指接口、MediaPipe 模型加载、双相机 RGB-D 渲染、磁盘空间和采集接口，结果保存到 `reports\preflight`。

### 3.2 电脑前置摄像头读取检查

先关闭占用摄像头的会议软件和其他遥操程序，再执行：

```powershell
& 'D:\openpi\dataset\dataset.ps1' -Mode Camera -Camera 0
```

读取约 3 秒并释放摄像头，报告实际尺寸和读取吞吐，不保存人体照片，也不将读取吞吐误认为经验证的独立曝光帧率。若摄像头不在设备 0，可尝试 `-Camera 1`。

### 3.3 数据契约和共享抓放回归

```powershell
& 'D:\openpi\dataset\dataset.ps1' -Mode Test
```

检查 10 Hz 采样、相机与控制索引、机器人状态提取、夹爪单位、阶段数据隔离、抖动统计边界，以及当前共享抓取和操作回归。日志在 `logs`。

### 3.4 不依赖真人的软件完整流程测试

```powershell
& 'D:\openpi\dataset\dataset.ps1' -Mode Smoke
```

该命令在无窗口模式下运行软件意图夹具，保存双相机数据，完成物理抓放检查并执行独立动作回放。它只验证软件与仿真链路，不能当作真人采集效果或实时帧率成绩。数据全部进入 `p00_smoke_v1`，明确禁止训练使用。

等待命令结束，检查退出结果以及 `reports\openarm_pickplace_p00_smoke_v1\batch_001\*.smoke.json`。应为 `verdict: PASS`。软件测试的仿真时间不等于现场墙钟时间。

### 3.5 真人试采：先从当前保持技能开始

```powershell
& 'D:\openpi\dataset\dataset.ps1' -Mode Pilot -Task Hold -Pose center -Trials 5 -ResetSeconds 3 -Seconds 0 -Batch 1 -Operator operator01
```

打开当前熟悉的遥操窗口。按一次 **SPACE** 开始，然后用手靠近并捏合。捏合确认一次后抓取请求即被锁存，可以自然放松手指，不必持续捏合；机器人会完成闭爪、抬起并保持夹爪闭合。3 秒保持倒计时结束后，只有一次新的张手动作才会让机器人自动原位放下。若已经提前张手并一直保持张开，需先合手或放松到非张手状态，再重新张手。失败也保存，可随时按 **Q** 保存退出。

采集过程不新增逐阶段模式切换键，不要求每次重新做分轴检测。标定沿用复制过来的个人标定；只有操作者、相机位置或识别尺度明显变化时再校准。`C` 只重新对齐人手到机械臂的相对映射，不复位机械臂或方块。

当前保持技能先做 5 条试采并全量查看，覆盖靠近、抓起、稳定保持与原位放下。确认全部稳定后再进入 `Collect -Task Hold`。

## 4. 正式采集命令

### 4.0 当前保持技能：首批 30 条

建议分成 3 个批次，每批 10 段；每批结束后先审核，再继续下一批：

```powershell
& 'D:\openpi\dataset\dataset.ps1' -Mode Collect -Task Hold -Pose center -Trials 10 -ResetSeconds 3 -Seconds 0 -Batch 1
& 'D:\openpi\dataset\dataset.ps1' -Mode Collect -Task Hold -Pose center -Trials 10 -ResetSeconds 3 -Seconds 0 -Batch 2
& 'D:\openpi\dataset\dataset.ps1' -Mode Collect -Task Hold -Pose center -Trials 10 -ResetSeconds 3 -Seconds 0 -Batch 3
```

这里只统计审核通过的成功段。失败、取消或未保持满 3 秒的数据会保留用于排查，但不计入 30 条训练示范。

### 4.1 基础 100 条

先从中心位置熟悉操作：

```powershell
& 'D:\openpi\dataset\dataset.ps1' -Mode Collect -Pose center -Operator operator01
```

换一个初始位置：

```powershell
& 'D:\openpi\dataset\dataset.ps1' -Mode Collect -Pose left -Operator operator01
```

从候选初始条件表中随机选取，并记录种子：

```powershell
& 'D:\openpi\dataset\dataset.ps1' -Mode Collect -Pose random -Seed 101 -Operator operator01
```

可选条件为 `center / left / right / near / far / yaw0 / yaw60 / random`。它们只设置 session 开始时的仿真物体状态，不给感知或规划提供物体真值。除中心软件测试外，其他条件应先通过人工试采审核，不能将这些候选条件视为已保证全部可达。

每次启动生成新的 session，默认一个尝试后自动保存，方便从新的初始条件采样。更换命令不覆盖之前的数据。

### 4.2 位置与朝向覆盖补采

```powershell
& 'D:\openpi\dataset\dataset.ps1' -Mode Coverage -Pose random -Seed 201 -Operator operator01
```

### 4.3 恢复示范

```powershell
& 'D:\openpi\dataset\dataset.ps1' -Mode Recovery -Pose right -Operator operator01
```

只有发生过可控的小偏差、随后正确恢复并完成放置的过程，才标为成功恢复示范。模式名不会自动证明轨迹中存在有效恢复；审核备注应说明恢复了什么问题。不故意制造碰撞或让物体掉出工作台。

### 4.4 隔离参考数据

```powershell
& 'D:\openpi\dataset\dataset.ps1' -Mode Eval -Pose far -Operator operator01
```

该阶段始终排除训练。正式留出条件应先登记，不能一边采入训练、一边又称作未见条件。

### 4.5 保留原来连续操作方式

如果要连续练习并一次保存整个 session：

```powershell
& 'D:\openpi\dataset\dataset.ps1' -Mode Pilot -Trials 0 -Seconds 0
```

也可以设置一个 session 做若干尝试：

```powershell
& 'D:\openpi\dataset\dataset.ps1' -Mode Pilot -Trials 3 -Seconds 300
```

`Trials` 是尝试数，不是成功数。一次尝试从新鲜 RGB-D 姿态下确认的新捏合开始；该请求帧现在会原子冻结给规划器，下一相机帧的瞬时缺失不会再生成“还没抓取就结束”的空失败段。非最后一段结束后，机械臂与方块会自动回到该 session 的初始状态；随后进入受保护的操作者重置时间，默认 3 秒，试采与正式采集一致。界面会显示段号、开始、完成和重置倒计时，倒计时结束前不会触发下一段。可用 `-ResetSeconds` 调整时长：

```powershell
& 'D:\openpi\dataset\dataset.ps1' -Mode Pilot -Task Hold -Trials 3 -ResetSeconds 3 -Seconds 0
```

自动场景复位会记录独立标记，并在原始动作回放时重现，但不会被分段器误当成训练轨迹。操作者应利用倒计时恢复手部和身体姿势，看到下一段开始提示后再靠近、捏合。

## 5. 人的动作可能抖动，采集时具体怎样操作

### 5.1 为什么要先控制采集质量

模型学习的是机器人在图像与自身状态下对应的动作目标。手部估计噪声、频繁修正和目标反复跳动可能被学成犹豫或抖动。应区分：

- 人实际在来回纠正：改进示教操作。
- 人手停稳但目标仍跳：检查识别、光照、标定和映射。
- 目标平稳而机器人实际抖动：检查执行反馈与接触，不能靠修改人体轨迹标签掩盖。

目前新增报告统计的是**手动阶段已接受目标的速度、反向修正与警告占比**，是筛查线索，不是自动识别生理手抖或精确区分以上原因的分类器。

### 5.2 推荐的一条演示动作

1. **坐姿和取景固定。** 肩膀、手腕和手指保持清楚入镜，手指能分辨。前臂可轻靠桌缘、肘部有支撑，避免悬空用力；不要为了追机器人而前后晃动身体。减少逆光、阴影和手在画面边缘移动。
2. **开始后停稳约半秒。** 按 SPACE 后让系统建立当前相对位置与前后参考。不要一开始就快速挥手。只有中心明显不合适时用 C 重新定中心；任务中途重定中心的数据不直接当首版标准示范。
3. **靠近使用连续、匀速的移动。** 以手腕和前臂整体带动，方向确定后连续送过去。避免同一个位置前后、左右多次试探修正，也不要一帧看起来没动就突然大幅补偿。
4. **接近阶段自然张手。** 进入绿色“可以抓取”提示后，先停稳约 0.3–0.5 秒再捏合。无需额外张手准备步骤或切换模式。捏合保持到请求确认；现有入口要求新鲜二维/三维捏合持续约 0.2 秒。
5. **辅助抓取阶段不要继续追着对准。** 机器人精细对准、闭合和提起时，手保持舒适稳定，避免大幅摆动或明确长时间张手取消。不要把夹住方块后仍有爪间开口误认为没有夹紧。
6. **交回搬运后走清晰路径。** 手指保持捏合，用一条平滑路线把方块送到目标上方。可以先抬高少量再水平搬运；避免贴着桌面拖动或用多个小幅反复修正来回磨。
7. **目标上方停稳，再张手。** 看方块中心是否对准绿色区域，停稳后张手保持约 0.35 秒发出放下请求。让系统完成下降、支撑、松爪、退出和验证，不要同时继续大幅移动手腕。
8. **等结果封口。** 等稳定验证和自动保存结束，查看本条是完整成功、失败还是中断，再开始下一条。少量自然停顿是有效行为；不要为了追求全程不停而牺牲抓稳和放准。

这里的停稳是改善示教动作的建议，不是新增控制门槛，也不是每个阶段都要按键确认。

### 5.3 哪些情况重录或拒收

- 持续抖动、方向反复变换、反复撞到边界或机械臂频繁保持。
- 人手/身体长时间丢失，过程中暂停、重新标定或重新放置物体。
- 抓起后滑落、放到目标外、没有松爪、过程超时或依靠外力摆放才成功。
- 图像缺口过大、实际新观察频率不够、动作与图像不对应。

这些原始数据仍保留供排查。自动阶段手暂时不可见，不应据此删除机器人本来正确的对准、闭合和放下数据。

### 5.4 不要直接平滑保存后的动作标签

闭合、接触和松爪发生在特定时刻。事后平滑目标可能改变接触时机和物理结果，使“这张图—这个状态—这个动作”不再对应一次真实执行。

本轮先通过稳定示教、单次任务分段和审核改善质量。若后续确认需要在线滤波，只在人工自由运动阶段以有界、可回放方式接入，重新比较延迟与任务成功；不直接平滑夹爪和接触动作。任何离线处理都保留原始数据，并用处理后的动作重新执行和评价。

## 6. 采后审核、回放与数量统计

采集结束会打印完整 session 路径。以下示例中的 `$session` 必须替换为实际路径：

```powershell
$session = 'D:\openpi\dataset\raw\openarm_pickplace_p02_core100_v1\batch_001\实际session目录'
```

### 6.1 重新审核

```powershell
& 'D:\openpi\dataset\dataset.ps1' -Mode Audit -Session $session
```

重点查看：

- `recording_valid`：结构与原始索引是否正确。
- `physical_success_segments`：原有物理成功段数，不保证满足新数据质量要求。
- `candidate_successes`：通过新增时间、图像、任务与接口检查，等待人工查看的段数。
- `training_approved_successes`：允许训练的阶段中，原始动作回放通过且经过明确人工审核的合格段数。
- `trials[].issues`：不能进入完整示范的硬性问题。
- `trials[].motion.review_flags`：速度突变、反向修正或警告较多，需重点查看。

抖动阈值是工程筛查起点，不能靠某个数字自动判断所有动作质量。少量必要的纠偏不等于坏数据；需要结合图像和实际执行结果。

### 6.2 原始动作动力学回放

无窗口校验：

```powershell
& 'D:\openpi\dataset\dataset.ps1' -Mode Replay -Session $session
```

打开窗口查看机器人完整执行：

```powershell
& 'D:\openpi\dataset\dataset.ps1' -Mode Replay -Session $session -Visible
```

这是相同模型下的原始动作一致性检查，不是重新进行一次真人测试，也不代表以后降采样/平滑后的动作已经通过验证。

### 6.3 明确接受或拒绝一条示范

先看完整回放和审核报告，再执行，`Trial 1` 是 session 内试次编号：

```powershell
& 'D:\openpi\dataset\dataset.ps1' -Mode Review -Session $session -Trial 1 -Decision accept -Note '完整抓放成功，人工阶段平稳，位置与释放正确'
```

拒绝示例：

```powershell
& 'D:\openpi\dataset\dataset.ps1' -Mode Review -Session $session -Trial 1 -Decision reject -Note '搬运时反复修正，重录'
```

接受前须完成 `Replay`。审核写入单独的 `review.json`，绑定当前原始记录和质量契约的摘要，不改原始物理成功/失败标签；若原始记录或质量契约变化，旧审核不再自动生效。软件夹具和存在硬性质量失败的数据不能被命令强行接受为真人示范。试采和隔离参考即使看过，也不会计入正式训练批准数。

### 6.4 查看各阶段进度

```powershell
& 'D:\openpi\dataset\dataset.ps1' -Mode Summary
```

统计 session、尝试、候选成功和审核批准数，并保存汇总。达到数量目标后仍须执行分组划分、训练输入隔离和格式转换，不能直接把整个 session 喂给训练器。

## 7. 本轮验证与下一步

本轮运行记录见 `logs` 与 `reports`。软件自检只能证明录制、双相机、分段与动力学回放链路；真人试采还需验证实际取景、手部识别、操作舒适度和实时负载。

本次已完成的检查：

- 104 项数据契约、审核隔离、自动复位与共享控制回归测试通过。
- 环境、MediaPipe 模型加载、B 场景与双相机 RGB-D 渲染检查通过。
- 前置摄像头本机检查通过：1280×720，约 28 次/秒读取吞吐，未保存人体图像。
- 两段连续软件测试完成 2 次物理成功抓放，段间恰好执行 1 次机械臂与方块复位；回放应用相同复位边界，最大 qpos/qvel 差异为 0。
- 双相机在软件测试仿真时间轴上达到 10 Hz，复位行不进入任一试次分段，全部 synthetic 数据明确禁止训练。

精确记录与本轮集成改动摘要见 `reports\preparation_verification_20261028.json` 及 `snapshots\integration_20261028`。受限工具环境首次相机打开失败，本机权限下重测通过；失败日志也已保留。

当前首先运行 `Pilot`，检查几条完整演示的实际图像频率与抖动报告。试采达到稳定质量后，再按 `Collect → Coverage → Recovery` 组织正式采集。相机、动作接口、阶段名称或主要质量阈值发生变更时，发布新版本目录，不与 v1 默默混合。
