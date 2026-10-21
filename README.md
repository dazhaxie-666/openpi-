# OpenArm / Pose2Arm A+B 主项目

更新日期：2026-10-21。日常只使用 `D:\openpi\latest\openpi--main`。
基于上游 `8df4726` 加本地集成改动；本次整理不包含 GitHub 上传或训练新模型。

后续代码发布目标与数据排除范围见 [GitHub 发布说明](docs/github_publish_20261021.md)。
新克隆不包含本机视频、模型权重和 outputs；示范回放需另行准备对应数据。

## 快速开始

PowerShell：

```powershell
Set-Location -LiteralPath 'D:\openpi\latest\openpi--main'

# 查看整段视频动作回放，可旋转、缩放、暂停和逐帧
& .\run.ps1 -Mode fusion-replay

# 视觉识别并抓取蓝色方块；摆好物体后松开鼠标，空格开始
& .\run.ps1 -Mode tabletop -ScriptArgs '--shape','cube','--color','blue'

# 可选的几何抓取图形选择器
& .\run.ps1 -Mode tabletop-ui

# 真人摄像头双臂跟随
& .\run.ps1 -Mode live

# 同一个 B 环境中的资源检查、完整回归
& .\run.ps1 -Mode check
& .\run.ps1 -Mode test
```

回放窗口不能当作物体摆放窗口。`tabletop` 中双击物体选中，再 Ctrl＋鼠标右键拖动；
普通拖动用于调整视角，当前没有 WASD 摆物快捷键。相机图像窗口不能拖动三维物体。
抓取开始后不再进行摆放约束或复位。鼠标交互仍需专项使用验收，不能用离屏测试替代。

## 项目结构

| 路径 | 作用 |
| --- | --- |
| `openarm_cloth/` | B 机器人、相机标定、控制器、仿真、视觉抓取及 A→B 适配 |
| `pose2arm-main/` | A 人体/手部识别、动作参考、视频、模型及历史研究工具 |
| `openarm_cloth/outputs/` | 本地回放和验证证据；历史结果不随整理删除 |
| `docs/` | 本次工作区整理与归档说明 |
| `RUNNING.md` | 统一入口和完整命令 |
| `PROJECT_STATUS_20261021.md` | 当前实现、已有指标、真实失败情况和下一步 |

全部日常入口沿用 B 的 Python，配置保存在忽略提交的 `.runtime.local.json`。
旧依赖整包、独立 Notebook 和另一份工作副本已移到工作区外层 `_archive/`，
避免误用。源码、运行模型、标定、数据集、私人视频与有效输出均保留。

## 当前范围

人体动作复现与几何视觉抓取已共存，但仍是**独立运行模式**，不是已训练的完整任务策略。
默认 `fusion` 是纯动作运动学回放；`--execution dynamic` 才测试仿真执行器。
`tabletop` 使用 RGB-D 几何识别方块/长方体/竖直圆柱，当前每次物理演示一个目标。
没有承诺任意物体、任意摆位、多人/遮挡精度、全程安全、零抖动或实体硬件效果。

最近固定仿真验证六例通过（3 次物理抓取、3 次仅识别），但最新交互记录也有一次
“无足够对齐的抓取候选”失败；完整记录见状态文档，不把固定工况当作泛化成功率。
