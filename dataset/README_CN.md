# 数据采集工作区

更新：2026-11-17。当前交接任务为 **右臂靠近红色方块 → 抓取 → 抬升 → 保持3秒 → 原位放下**，运行在 MuJoCo 仿真中。

- 首次部署和每日命令：[新电脑交接说明](HANDOFF_CN.md)
- 操作标准、阶段、采后审核：[完整采集指南](PI05_HOLD_DATA_COLLECTION_GUIDE_CN.md)
- 阶段名称与质量阈值：[config/stages.json](config/stages.json)
- 固定环境版本：[requirements.txt](requirements.txt)

## 目录和保存位置

克隆仓库后，`dataset/` 位于 `run.ps1` 旁边：

```text
<仓库>/
  run.ps1
  openarm_cloth/
  pose2arm-main/
  dataset/
    dataset.ps1
    requirements.txt
    scripts/        采集、审核、回放工具
    config/         任务和阶段合同
    tests/          数据与控制回归测试
    calibration/    本机操作者捏合标定
    raw/            原始 session
    reports/        预检、审核、回放、统计报告
    logs/           每次命令日志
    snapshots/      配置与源码哈希
    tmp/            标定和软件测试临时文件
    exports/        后续版本化训练导出
```

运行目录由脚本位置决定，不取决于当前终端所在目录。执行首次 Check 后自动创建输出目录。
外置旧布局 `D:\openpi\dataset` 仍可寻找 `../latest/openpi--main`，与克隆内的 dataset 独立保存数据。

## 基本命令

在仓库根目录、已激活 openpi 环境的 PowerShell 中：

```powershell
& .\dataset\dataset.ps1 -Mode Check
& .\dataset\dataset.ps1 -Mode Camera -Camera 0
& .\dataset\dataset.ps1 -Mode Calibrate -Camera 0 -Operator operator02
& .\dataset\dataset.ps1 -Mode Pilot -Task Hold -Camera 0 -Operator operator02 -Batch 101 -Trials 5 -Seconds 900
& .\dataset\dataset.ps1 -Mode Collect -Task Hold -Camera 0 -Operator operator02 -Batch 101 -Trials 5 -Seconds 900
& .\dataset\dataset.ps1 -Mode Summary
```

`Pilot`、`Smoke`、`Eval` 不进入正式训练。正式示范按完整试次分段，必须通过 Audit、原始动作 Replay 和人工 Review；数量以 `training_approved_successes` 为准。

原始数据、个人标定、报告和日志由本目录的 .gitignore 排除，不随代码上传；数据交接时另行复制完整 session 及对应报告、配置快照。新电脑不需要旧录像、outputs、历史归档或旧数据即可开始采集。

这是带时间对应关系的共享示教源，尚未完成 LeRobot/π0.5 训练格式转换。完整仿真状态和物体真值不能直接作为策略输入。
