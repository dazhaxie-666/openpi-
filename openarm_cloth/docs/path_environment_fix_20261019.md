# 最新代码与 B 环境统一（2026-10-19）

> 2026-10-21 补充整理：上游自带旧整包、Notebook 与新增来源副本也已可恢复归档。
> 本文保留 10-19 证据，不代表最新完整目录；见 [清理清单](../../docs/cleanup_20261021.md)。

## 来源与目录

- GitHub：`https://github.com/sishui667/openpi-`
- 下载基线：`8df472635dab0f199939a6b7ff8fefaa8070f76b`
- 当前唯一工作副本：`D:\openpi\latest\openpi--main`，包含可追踪的 `.git`。
- 旧本地 `pose2arm-main`、依赖打包目录、Joint-action 目录和先前的
  `latest\openpi--main` 已移出工作位置，归档到
  `D:\openpi\_archive\before_github_8df4726_20261019`，可恢复。
- 当时保留了上游自带历史整包；10-21 将其移出主项目，生产模型资源仍保留在 B 中。

## 修复

统一使用根目录 `run.ps1` 选择 B 的 Python，覆盖 live/replay/grasp/check/test。
本地解释器是 `D:\ProgramData\Anaconda3\envs\openpi\python.exe`，Python 3.12.14、
MuJoCo 3.8.1、MediaPipe 0.10.35；未更换系统环境包版本。
原 `start_simulation.ps1`、桌面 `.pyw` 启动器也委托到同一入口。

移除了三个 A 工具对旧依赖包/绝对路径的引用，改为 B 目录中的模型资源；
规划回归测试的导入路径改为从自身文件定位。新入口从任意工作目录可用，
在启动前拒绝 Windows MuJoCo 不支持的中文路径。测试临时目录位于可写的 outputs，
入口退出时恢复调用终端的 TEMP/TMP/PYTHONUTF8。

从旧本地目录迁移 hand/heavy 模型，并从官方固定版本地址补齐 lite/full 模型。
新增 `setup_models.py`、`check_runtime.py`、`requirements_live.txt` 与根目录
`RUNNING.md`，环境检查会验证实际模块来源和模型加载。

## 本次验证

| 检查 | 结果 |
|---|---|
| B 单元/集成测试 | 144 项：141 通过，3 个符号链接权限跳过 |
| A 接口测试 | 11/11 通过 |
| 模型初始化 | hand/lite/full MediaPipe CPU 检测器均成功创建 |
| B 场景 | 抓取 nq=25、nu=18；真人动作场景 nq=18、nu=18 |
| A→B 回放 | 417 帧完成，无急停；14 帧原有速度限制干预 |
| B 视觉抓取 | 测试摆位 (0.43, 0, 30°)，任务门限 PASS；抬升 69.9 mm |
| 历史 A 碰撞诊断 | 1224 帧，旧 NPZ 参考轨迹碰撞 mesh 与视觉 mesh 接触均为 0 |
| B 编译配置对照 | physics/workspace/actuators/cameras/task_definition/controller 六项与原冻结清单一致 |

当前报告保存在 `openarm_cloth/outputs/runtime_check_20261019.json`、
`replay_check_20261019.json`、`grasp_path_check_20261019.json`。
outputs 属于本地运行数据，不提交 Git。

## 边界

本次修复路径、引用、解释器选择与资源交付。真人摄像头动作跟随未重新实拍验收。
回放和自主抓取仍是不同任务入口，离线回放默认只执行右臂。

B 的实时执行与避障直接使用当前 B 场景及其滑动夹爪；A 历史诊断/运动学回放
仍使用 B 资源目录保存的旋转夹爪模型，以保持旧 NPZ 语义。
历史 A 的安全结论不能替代 B 当前实际模型的安全检查。

历史验收清单包含运行源文件哈希。本次路径修改后，其完整源码哈希自然不再与旧清单
一致；历史证据未被覆盖。需要新的冻结清单时可在 B 目录执行：

```powershell
& 'D:\ProgramData\Anaconda3\envs\openpi\python.exe' freeze_member_b_scene.py --write --manifest outputs\path_fix_scene_manifest.json
```

运行命令统一见 [RUNNING.md](../../RUNNING.md)。
