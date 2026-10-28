# 2026-10-28 数据采集与自动场景复位更新

目标仓库：<https://github.com/dazhaxie-666/openpi->。本次继续基于远端 `main` 新增提交，不改写历史。

## 主要更新

- 新增仓库内 `dataset/` 统一入口、阶段配置、数据契约测试、审核、动作回放和人工准入流程。
- 支持一个 session 采集指定数量的试次；界面明确显示当前段号、开始、结束、最终保存和操作者重置倒计时。
- 非最后一段结束后自动恢复机械臂、夹爪、方块和任务内部状态，再进入默认 8 秒受保护重置时间。
- 采集初始 TCP 与方块保持更大距离，降低启动时误触和遮挡风险；普通非采集入口的默认行为不变。
- `C` 的含义明确为人手到机械臂相对映射重新对齐，不再让操作者误认为它会重置场景。
- 原始控制行新增场景复位标记和计数；分段器排除复位/最终提示行，避免生成虚假试次。
- 动作回放识别复位边界并恢复记录的环境状态，保证多段 session 可以独立复现。
- `collection.json` 在 session 建立时即写入阶段、用途、训练许可、参数和源码哈希；正式训练仍要求结构审核、原始动作回放和人工接受。

## 运行

在仓库根目录执行：

```powershell
& .\dataset\dataset.ps1 -Mode Check
& .\dataset\dataset.ps1 -Mode Camera -Camera 0
& .\dataset\dataset.ps1 -Mode Test
& .\dataset\dataset.ps1 -Mode Smoke -Trials 2 -ResetSeconds 2
& .\dataset\dataset.ps1 -Mode Pilot -Camera 0 -Trials 3 -ResetSeconds 8 -Seconds 0 -Batch 1
```

详细的阶段隔离、操作方式、质量指标、回放和审核命令见 `dataset/README_CN.md`。

## 验证

- 104 项数据契约、共享抓取、分段、界面、自动复位和回放回归测试通过。
- 两段连续软件夹具完成 2 次物理成功抓放，段间恰好应用 1 次环境复位。
- 原始动作回放通过，最大 `qpos` / `qvel` 差异均为 0，并报告 `environment_resets_applied: 1`。
- 环境、MediaPipe 模型、MuJoCo 场景、机器人 RGB-D 和源码哈希预检通过。

软件夹具只证明代码与仿真链路，不代表真人示教质量；`Smoke`、`Pilot` 和 `Eval` 数据均不能通过改标签进入正式训练。

## 发布排除范围

上传源码、测试、说明和必要仿真资源。以下内容保持在本机：

- `dataset/raw` 原始采集与相机数据；
- `dataset/logs`、`reports`、`snapshots`、`exports` 和临时目录；
- 个人夹爪标定、私人视频、检测器权重、编译模型和本机解释器配置。

这次发布不包含用户采集记录，也不声明已经训练或验收了自主策略。
