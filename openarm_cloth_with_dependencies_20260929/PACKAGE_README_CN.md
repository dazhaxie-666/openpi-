# OpenArm Cloth 发布包

本压缩包包含 `openarm_cloth` 的源码、MuJoCo 场景和网格资源、CAD 文件、数据集、最终标定结果，以及当前 `openpi` 环境所需的 Python wheel 离线安装包。

## 目录说明

- `openarm_cloth/`：项目源码和仿真资源。
- `dependencies/wheels/`：可离线安装的 Python wheel。
- `dependencies/requirements_openarm_cloth.txt`：主运行环境依赖。
- `dependencies/environment_pinocchio.yml`：Pinocchio 4.0 的 Conda 环境定义。Pinocchio 没有放入普通 pip wheel，而是通过 Conda-forge 安装。

## 安装主环境

在 Windows PowerShell 中执行：

```powershell
conda env create -f .\dependencies\environment_openpi.yml
conda activate openpi
python -m pip install --no-index --find-links .\dependencies\wheels -r .\dependencies\requirements_openarm_cloth.txt
```

## 安装 Pinocchio

```powershell
conda env create -f .\dependencies\environment_pinocchio.yml
```

如果环境名称已经存在，可以直接在已有环境中安装：

```powershell
conda activate openpi-pinocchio-py312
conda install -c conda-forge pinocchio=4.0
```

## 运行示例

```powershell
conda activate openpi
cd .\openarm_cloth
python .\handeye_grasp_red_cube.py --arm left --preview
```

需要访问远程 OpenPI 服务时，另行安装或使用 `openarm_cloth/requirements_remote_openpi.txt` 中的客户端依赖。conda env create -f .\dependencies\environment_openpi.yml
conda activate openpi
python -m pip install --no-index --find-links .\dependencies\wheels -r .\dependencies\requirements_openarm_cloth.txt

## 打包策略

发布包排除了 `__pycache__`、Python 字节码、官方仓库 `.git` 历史、MuJoCo 日志、历史调试标定会话和中间图像。保留了左臂、右臂、胸部的最终标定 JSON；标定数据可以使用包内脚本重新采集。
