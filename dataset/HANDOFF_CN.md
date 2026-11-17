# 新电脑数据采集交接说明

更新：2026-11-17。接收人按本文件依次执行，命令均在 **Windows PowerShell** 中运行。

任务：右臂靠近红色方块、抓取、抬升、静止保持3秒、原位放下。人体输入来自普通RGB摄像头，机器人及胸部/双腕RGB-D来自MuJoCo仿真。采集结果是后续π0.5转换与训练的数据源。

## 1. 领取代码和确认设备

准备 Windows、Git、Conda、一台正常工作的RGB摄像头，以及支持MuJoCo窗口与离屏渲染的显卡驱动。建议采集盘至少空余50 GiB；预检会检查实际空间。新电脑先关闭会议软件等摄像头占用程序。

从 main 克隆到ASCII路径。以下路径只是统一示例；没有D盘可换成其他本地英文路径：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
git clone --branch main --single-branch https://github.com/dazhaxie-666/openpi-.git D:\openpi_handoff
Set-Location -LiteralPath 'D:\openpi_handoff'
$repo = (Get-Location).Path
$dataRoot = Join-Path $repo 'dataset'
git rev-parse HEAD
```

登记提交号、电脑编号、摄像头编号和操作者编号。一个批次内冻结代码、场景与质量配置；新批次由负责人统一分配。不要在采集期间运行 git pull。

执行策略只对当前PowerShell进程生效。每次新开终端先切换到仓库目录，必要时重设这项进程策略。若PowerShell不识别Conda，在Anaconda/Miniconda Prompt中执行 `conda init powershell` 后重新打开PowerShell。

克隆后应有 `run.ps1`、`openarm_cloth`、`pose2arm-main` 和 `dataset\dataset.ps1`。不需要复制研发电脑的 `_archive`、历史 `outputs` 或旧数据。

## 2. 安装固定环境和识别模型

```powershell
conda create -n openpi python=3.12 -y
conda activate openpi
python -m pip install -r .\dataset\requirements.txt
python -m pip check
python .\openarm_cloth\setup_models.py
```

如果已有同名环境，先由负责人核对Python版本及依赖，避免覆盖用于其他任务的环境。首次模型下载需要访问 `storage.googleapis.com`。离线电脑可单独领取下列四个官方权重，放到 `<仓库>\pose2arm-main\models\`：

`requirements.txt`同时读取本次Windows/Python 3.12实测的间接依赖约束；[验证结果](VERIFICATION_20261117.md)记录了独立环境和软件采集回放结果。

```text
hand_landmarker.task
pose_landmarker_lite.task
pose_landmarker_full.task
pose_landmarker_heavy.task
```

本任务使用MediaPipe，不需要额外下载RTMPose/RTMW权重。

将本机环境路径写入本地配置，启动器以后会自动读取：

```powershell
$pythonExe = (python -c 'import sys; print(sys.executable)').Trim()
@{python=$pythonExe} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $repo '.runtime.local.json') -Encoding UTF8
& .\run.ps1 -Mode check -ScriptArgs '--check-models'
```

必须得到 `verdict: PASS`。四个识别模型、机器人场景和本仓库内的模块都应加载成功。`.runtime.local.json`是本机路径，不能照抄研发电脑的Anaconda路径。

## 3. 软件链路验收

```powershell
& .\dataset\dataset.ps1 -Mode Check
& .\dataset\dataset.ps1 -Mode Test
& .\dataset\dataset.ps1 -Mode Smoke -Task Hold -Steps 1500 -Batch 1
```

验收标准：Check为PASS、Test全部通过、最后出现 `SOFTWARE SMOKE: PASS`。Smoke会生成带明确synthetic标记的数据，执行一次软件抓取/保持/放回和原始动作回放，**不得计入真人示范或训练数**。

初次Check可能提示没有个人捏合标定，这是下一步要完成的事项。软件PASS不能代替新电脑的摄像头检查和真人试采。

## 4. 摄像头和个人标定

下面三个变量由负责人填写，全文后续命令沿用它们：

```powershell
$operator = 'operator02'
$camera = 0
$batch = 101
& .\dataset\dataset.ps1 -Mode Camera -Camera $camera
```

设备编号不正确时尝试1、2等编号。Camera应PASS，人体RGB摄像头能读取图像；其读取吞吐不是完整采集时的实际帧率。操作者的肩膀、右肘、手腕与手指保持入镜，固定机位和光照。

首次标定：

```powershell
& .\dataset\dataset.ps1 -Mode Calibrate -Camera $camera -Operator $operator
```

在人体摄像头预览窗口获取键盘焦点，保持任务待机，不按SPACE：

1. 右手拇指、食指自然捏合，按G，保持约1.2秒；重复10次。
2. 右手自然张开，按O，保持约1.2秒；重复10次。
3. 看到 `CALIBRATION READY` 后按Q正常保存退出。
4. 检查文件存在，并登记它属于哪位操作者：

```powershell
Test-Path -LiteralPath (Join-Path $dataRoot 'calibration\pinch.json')
& .\dataset\dataset.ps1 -Mode Check
```

标定失败按界面提示补采端点。G/O是个人捏合标定，C只重新对齐人体与机械臂的相对位置。标定过程输出存于 `dataset\tmp`，不属于正式采集。

更换操作者时，由负责人先备份并移开旧 `pinch.json`，再重新标定。程序不会自动继承研发电脑的个人标定。同一session不得换人、移动机位或修改标定。

## 5. 试采5条并验收

```powershell
& .\dataset\dataset.ps1 -Mode Pilot -Task Hold -Camera $camera -Operator $operator -Pose center -Batch $batch -Trials 5 -Seconds 900
```

SPACE开始；人手平稳靠近，绿色“可以抓取”提示后捏合请求抓取；机器人自动对准、闭合与抬升；手腕停稳，等待3秒保持结束，再作一次新张手并保持约0.35秒，机器人原位下降释放。界面结束/复位期间等待倒计时。Q提前保存退出；尽量在完整试次结束后退出。

`-Trials 5`表示5次尝试，失败也占一次；不足5条合格成功时继续新session。`-Seconds 900`为本次采集的时间上限。每次运行生成独立session，不覆盖之前的数据。

完成后复制终端打印的 `SAVED:` 绝对路径到下一步的 `$session`，检查Audit、Replay及完整画面。五条试采通过后负责人签字，才能正式采集。Pilot始终隔离，不改目录名转成Core。

## 6. 正式采集30条

```powershell
& .\dataset\dataset.ps1 -Mode Collect -Task Hold -Camera $camera -Operator $operator -Pose center -Batch $batch -Trials 5 -Seconds 900
```

重复小批次采集并审核，达到30条 `training_approved_successes`。需要位置覆盖时，负责人明确分配 `-Pose near`、`far`、`yaw0`、`yaw60` 或 `random`；不要由不同采集人自行改场景。完整标准见[采集指南](PI05_HOLD_DATA_COLLECTION_GUIDE_CN.md)。

采集模式必须显式带 `-Task Hold`；省略时旧接口默认是搬运到绿色区域。根目录 `start_shared_grasp.ps1` 用于原有交互演示，正式Hold采集使用本说明的dataset入口。

## 7. 每个session采后审核

```powershell
# 把下一行替换为终端实际打印的SAVED路径
$session = Join-Path $dataRoot 'raw\openarm_grasp_hold_return_p02_core30_v1\batch_101\实际session目录'
& .\dataset\dataset.ps1 -Mode Audit -Session $session
& .\dataset\dataset.ps1 -Mode Replay -Session $session
& .\dataset\dataset.ps1 -Mode Replay -Session $session -Visible
```

审核 `recording_valid`、`trials[].issues`、图像缺口、抖动提示和候选段。无界面Replay必须PASS；再查看完整可见回放和机器人相机图像，确认抓取、3秒保持、原位释放均正确，手动段没有明显抖动或长期丢失。Replay只验证相同模型中的动作一致性，不等于人工看过画面。

第一条新记录还必须检查任务文字：

```powershell
(Get-Content -LiteralPath (Join-Path $session 'manifest.json') -Raw | ConvertFrom-Json).task_instruction
(Get-Content -LiteralPath (Join-Path $session 'collection.json') -Raw | ConvertFrom-Json).task_contract
```

应描述抓取、保持3秒、放回起始位置，不能写成放入绿色区域。

逐条填写明确的人工Review；`-Trial`是session中的试次编号：

```powershell
& .\dataset\dataset.ps1 -Mode Review -Session $session -Operator $operator -Trial 1 -Decision accept -Note '完整查看：抓稳，保持3秒，原位放回，手动段平稳'
# 不合格的段改用reject，写明真实原因；不得使用未实际检查的模板备注
& .\dataset\dataset.ps1 -Mode Summary
```

原始动作、图像、标签不得手工修改。Review是单独注释，绑定原始记录摘要；失败和中断数据保留供排查。Smoke/Pilot/Eval不进入正式训练。

## 8. 数据保存和回交

数据由 **执行的dataset.ps1所在目录** 决定，和终端当前目录无关：

```text
<仓库>\dataset\raw\<数据集名>\batch_XXX\<session>\
<仓库>\dataset\reports\<数据集名>\batch_XXX\
<仓库>\dataset\snapshots\<数据集名>\batch_XXX\
<仓库>\dataset\logs\
<仓库>\dataset\calibration\pinch.json
```

每个session包含 `collection.json`、`manifest.json`、`episode.npz`、三路 `shared_rgbd`、帧索引、试次/阶段记录及场景。运行Replay和Review后还有对应验证记录。必须复制整个session，不能只交NPZ、视频或场景。

采集人员正常关闭程序后，连同对应reports、snapshots、logs、使用的标定和下表交回负责人；保留数据集/批次/session层级。交接前后比较文件数量、总字节数，并由接收端复跑Audit与Replay。跨机器重放失败时保留报告，核对MuJoCo版本、模型和源码，不修改原始记录来“对齐”。

| 日期 | 操作者 | 电脑 | Git提交 | Camera | 数据集/批次 | Pose | Session | 尝试/候选/批准 | Replay | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |

代码通过GitHub交接；新采集数据和个人标定默认只保存在本机，另行通过负责人约定的存储介质回交。新电脑安装完成、Camera通过、个人标定完成、5条真人试采经审核后，才签收为正式采集工位。

## 9. 常见问题

- **找不到Python**：确认激活openpi环境，并重新生成本机 `.runtime.local.json`；也可给dataset.ps1传 `-Python <实际python.exe>`。
- **缺少.task模型**：重新执行setup_models.py，或离线复制四个官方权重；随后运行模型检查。
- **找不到XML或mesh**：确认代码克隆完整且整个路径无中文；不要单独复制几个Python脚本。
- **相机打不开**：关闭其他摄像头程序、检查Windows桌面应用相机权限、更换Camera编号后重跑Camera。
- **没有抓取请求**：检查个人标定READY、目标新鲜可见、右手入镜及绿色可抓取提示；先修正试采，不继续积累正式数据。
- **找不到SAVED或运行异常**：保留日志与残留session，标记为待排查；不要拼接文件冒充完整示范。

正式数据通过审核后仍需转换为训练格式；本交接没有启动π0.5训练或部署自主模型。
