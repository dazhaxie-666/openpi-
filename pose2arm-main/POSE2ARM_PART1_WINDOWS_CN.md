# 在 Windows 上应用 pose2arm —— Part1（姿态 → 双臂关节角）操作指南

> 本指南把 `pose2arm` 的 **纯 Python 部分**（人形姿态 → 双臂 14 维关节角）接入 `openarm_cloth`
> 项目，让你能用一段预录视频生成驱动 MuJoCo 双臂模型所需的关节角度。
>
> **平台**：Windows，使用你已配置好的 `openpi` conda 环境
> **目标（a）**：跑通 `视频 → 人体骨架 → IK → 输出 (N,14) 双臂关节角`
> **不含**：Isaac Lab / ROS / RL 训练（那是 pose2arm 的 Part2，与 MuJoCo 体系无关，暂不涉及）

---

## 1. 总体流程

```
视频/摄像头
  → MediaPipe PoseLandmarker（提取人体骨架，LIVE_STREAM）
  → 躯干局部坐标系的肩 / 肘 / 腕几何
  → One Euro 平滑
  → 18 维目标观测 [上臂方向 | 前臂方向 | 臂平面] × 2 臂
  → 7-DOF IK / 14-DOF 联合（含双臂碰撞惩罚）
  → 输出 (2N, 14) 双臂关节角
       左边 7：openarm_left_joint1..7
       右边 7：openarm_right_joint1..7
```

关键点：`pose2arm` 的双臂关节名与你 MuJoCo 模型
([openarm_bimanual.xml](../../openarm_cloth_with_dependencies_20260929/openarm_cloth/official_openarm_mujoco/v2/openarm_bimanual.xml))
中的关节名**完全一致**，因此输出的关节角可以直接 `set` 进 MuJoCo 双臂模型。

---

## 2. 前置条件

- 已安装并激活 `openpi` conda 环境（Python 3.12）
- 已能访问 `pose2arm-main` 的源码目录
- 准备一段包含人形挥动手臂动作的 `mp4` 视频

---

## 3. 安装依赖

```powershell
conda activate openpi
pip install mediapipe==0.10.35 opencv-python==4.13.0.92 numpy==2.2.6 scipy==1.15.3
```

---

## 4. 下载 MediaPipe 模型

```powershell
cd <你的 pose2arm-main 路径>
mkdir models
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task" -OutFile models\pose_landmarker_heavy.task
```

---

## 5. 冒烟测试（验证环境 OK）

```powershell
python -m py_compile extract_pose.py ik_solver.py openarm_common.py retarget.py smoothing.py
python smoothing.py
python scripts\smoke_test_openarm_ik.py
python scripts\smoke_test_retarget_geometry.py
```

---

## 6. 实现目标 a：用预录视频生成双臂关节角

> ⚠️ **关于示例视频**：`pose2arm` 文档中的 `videos/sample01.mp4` 是项目**本地文件**，
> 仓库未打包它（`videos/` 被 gitignore），因此**没有可靠的公开直链**。
> 请使用你自己录制/已有的一段"人会挥动手臂"的 mp4 即可（手机随手录一段也行）。

```powershell
python scripts\generate_ik_dataset.py --input 你的视频.mp4 --output data\rl_poses_output.npz
```

运行参数（可选）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input` | `videos/sample01.mp4` | 输入视频路径 |
| `--output` | `data/rl_poses.npz` | 输出 npz 路径 |
| `--delegate` | `cpu` | `cpu` 或 `gpu`（后台 GPU 委托） |
| `--restarts` | `20` | 单臂 Phase-1 IK 重启次数 |
| `--restarts-14d` | `5` | 双臂 14-DOF 细化重启次数 |

### 输出文件 `rl_poses_output.npz` 字段

| 字段 | 形状 | 含义 |
|---|---|---|
| `joint_angles` | `(2N, 14)` | **每帧双臂关节角【左7 + 右7】**，目标数据 |
| `target_obs` | `(2N, 18)` | 中间 18 维目标观测 |
| `frame_indices` | `(2N,)` | 原始帧号（前 N 为原始，后 N 为镜像） |
| `fps` | `float` | 视频帧率 |

`joint_angles` 每行 14 维对应：`[左 joint1..7 , 右 joint1..7]`。

---

## 7. 验证生成的关节角

在 `<pose2arm-main>` 下新建 `check_output.py`：

```python
import numpy as np
d = np.load("data/rl_poses_output.npz")
j = d["joint_angles"]
print("joint_angles", j.shape)          # 期望 (2N,14)
print("fps", d["fps"])
print("左臂前3帧关节角:")
print(np.round(j[:3, :7], 3))
print("右臂前3帧关节角:")
print(np.round(j[:3, 7:], 3))
# N = 原始帧数；前 N 行=原始剪辑，后 N 行=左右镜像剪辑
```

运行：

```powershell
python check_output.py
```

看到 `(2N, 14)` 且每臂 7 个关节角数值逐帧变化，**目标 a 即实现成功**。

---

## 8. 下一步展望（目标 b，另开任务）

把 `joint_angles` 实时/离线喂给你的 MuJoCo 双臂模型，让双臂在仿真里跟随人体动作：

- 输入的 14 维 = 左 7 + 右 7，与
  [openarm_bimanual.xml](../../openarm_cloth_with_dependencies_20260929/openarm_cloth/official_openarm_mujoco/v2/openarm_bimanual.xml)
  关节名 `openarm_left_joint1..7` / `openarm_right_joint1..7` 一一对应
- 方法：逐帧 `mjData.qpos` 对应索引赋值后 `mj_forward`，或接入 `position` 控制器
- （待打开关）写一个桥接脚本在 MuJoCo 场景中回放，确认双臂跟随效果

---

## 9. 常见问题

| 问题 | 排查 |
|---|---|
| `No module named 'mediapipe'` | 确认已在 `openpi` 环境执行安装，且 python 指向该环境 |
| 模块 `smoothing` 报错 | 必须在 `pose2arm-main` 根目录下运行（依赖相对路径导入） |
| 模型文件缺失 | 确认 `models\pose_landmarker_heavy.task` 已下载且非 0 字节 |
| 输出帧数异常 | 视频需保证人形完整、光照充足，骨架丢失帧会被跳过 |