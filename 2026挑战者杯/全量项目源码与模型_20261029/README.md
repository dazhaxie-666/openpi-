# 挑战杯全量主项目源码与模型交付包

日期：2026-10-29。本包在采集版交付的基础上，补齐当前唯一主项目 `D:\openpi\latest\openpi--main` 的全部代码与模型资源。原工程不修改，上一份采集版也保留。

## 全量范围

- `runtime/latest/openpi--main/openarm_cloth/`：完整当前机器人与仿真代码、相机与标定、机械臂模型、视觉感知、执行器、遥操作、共享抓取、采集、回放、审核、测试及文档。
- 同目录下 `simulation/`、`official_openarm_mujoco/`、`camera_cad/`：场景、MJCF/XML、网格、CAD、官方模型及本地已存在的编译模型快照。
- `runtime/latest/openpi--main/pose2arm-main/`：完整 Pose2Arm 源码、人体/手部提取、重定向、IK、平滑、碰撞与轨迹规划、scripts、tests、rl、Isaac/USD、Gazebo/URDF、模型权重、data、videos、out 和原文档。
- `runtime/dataset/`：全部采集工具源码、测试、配置、标定和文档；附两段历史真人成功操作对应的原始 session。
- `runtime/启动共享抓取.ps1`、`runtime/启动修复版遥操.ps1`：工作区顶层入口原样复制。

训练与 RL 代码作为原项目完整性的一部分保留，比赛当前验收仍只要求数据采集，不要求训练。历史 Isaac/Gazebo/RL 功能各有依赖，源码与已有资产收录不等于每个历史入口都已重新验收。

## 哪些没有复制

排除 `.git` 数据库、虚拟环境、Python 缓存、node_modules、全部历史 `outputs` 与凭据类文件。它们不属于本包完整源码与模型的范围。原项目代码中若引用历史 outputs 内的某次私人实验结果，需要单独准备对应历史记录；不承诺每个旧命令有现成输出。

外层 `_archive/`、旧分支工作副本不并入，避免混淆当前工程。数据工作区的所有历史失败/试采批次也不全部重复搬运；本包带一份可核验真人样例和对应报告。原始工程未删除任何文件。

## 如何核对完整性

`full_source_manifest.json` 列出当前主项目相对路径、每个复制文件来源、大小、SHA-256 和明确排除项。`evidence/full_package_verification.json` 给出新增全量包检查结果，并比较与上一份已验证采集核心的内容是否一致。

`evidence/package_verification.json` 与 `human_sample_replay.json` 保留上一份相同采集核心的 123 项测试、4,292 帧真人样例重执行结果。它们路径指向当时的验证副本，只作来源证据；本次全量包另行执行 Check 和 Test，当前路径见新报告。

## 使用

本包目录含中文时，先运行 `部署副本.ps1`，复制 `runtime` 到一个全新的纯英文目录，例如：

```powershell
& '.\部署副本.ps1' -Destination 'D:\openpi\challengecup_full_runtime_20261029'
```

随后以该目录为工作区，运行：

```powershell
& 'D:\openpi\challengecup_full_runtime_20261029\dataset\dataset.ps1' -Mode Check -Python 'D:\ProgramData\Anaconda3\envs\openpi\python.exe'
& 'D:\openpi\challengecup_full_runtime_20261029\dataset\dataset.ps1' -Mode Pilot -Task Hold -Trials 2 -Seconds 0 -Python 'D:\ProgramData\Anaconda3\envs\openpi\python.exe'
```

其他电脑应传入自己的 Python 路径。包内保留 `.runtime.local.json` 作为来源机器配置，显式 `-Python` 可覆盖它；不包含整个 Conda 环境。

具体方案及操作见 [项目方案](docs/项目方案.md)、[采集指南](docs/运行与采集指南.md)。指南里的部署目录是采集版示例，使用本全量包时替换为实际英文目录。

技术接入与模型位置见 [完整源码与模型索引](docs/完整源码与模型索引.md)。

当前支持三路仿真 RGB-D；附带真人样例为历史双相机记录，三相机另有软件自检。保留全部原始标签，未将试采重新标成正式训练集。

本包包含原项目的视频、个人标定与本地预训练模型，属于本地汇总，未向比赛网站或远程平台上传。
