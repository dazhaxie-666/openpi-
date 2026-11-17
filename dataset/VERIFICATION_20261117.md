# 新电脑交接准备验证

日期：2026-11-17。发布前基础提交：`27b6c004e3d6d70978ea054f549ad46776980775`。

## 验证方法

从暂存Git树导出干净副本，不携带研发电脑的原始数据、个人标定、outputs或模型缓存；使用新建、未继承site-packages的Python 3.12.14虚拟环境安装 `dataset/requirements.txt`。在干净副本执行官方 `setup_models.py`，实际下载四个MediaPipe权重。生产模块导入检查确认路径均位于该副本。

直接依赖和此次实测的间接依赖固定在 `requirements.txt` 与 `constraints-windows-py312.txt`。OpenCV两个分发包保持4.13.0.92一致，运行时cv2为4.13.0。

## 结果

| 检查 | 结果 |
|---|---|
| 新建环境安装 | 成功 |
| pip check | No broken requirements found |
| 官方模型下载 | 四个.task成功下载并加载 |
| run.ps1 -Mode check --check-models | PASS |
| dataset.ps1 -Mode Check | PASS；仅提示尚无新操作者捏合标定 |
| dataset.ps1 -Mode Test | 125项全部通过 |
| dataset.ps1 -Mode Smoke -Task Hold -Steps 1500 | SOFTWARE SMOKE: PASS |
| 软件记录结构审核 | recording_valid=true，issues=[] |
| 原始动作执行器回放 | PASS，499帧，qpos/qvel最大误差均为0 |
| Hold任务文字 | manifest.task_instruction与Hold配置描述一致 |
| 交接说明PowerShell示例 | 12段命令块语法检查通过 |
| 提交文件范围 | 源码、配置、依赖清单、测试与说明；无原始数据或个人标定 |

软件session：`20261117_193630_088398_2862a3`，位于本机独立验证副本的dataset/raw，不随代码提交。记录摘要：`64cb664ef2d63947ab4800a4f846f360be52ed7a99dd15f8f51351ae1ba92479`。

完成1次抓取、抬升、3秒保持和原位释放，原位误差约4.5 mm。三路仿真相机在**软件仿真时间轴**上每路147帧、10 Hz、最大间隔约0.1秒。synthetic_input=true、training_allowed=false、training_approved_successes=0。

## 交接验收范围

这些结果验证了干净源码与独立依赖环境的加载、采集、审核和同模型回放。软件测试按离线仿真时钟执行，本次499帧墙钟约114.6秒；其10 Hz图像记录不能当作新电脑实时吞吐保证，也没有验证新的真人操作或跨电脑确定性。

接收电脑仍按 [HANDOFF_CN.md](HANDOFF_CN.md) 进行Camera检查、个人G/O端点标定、5条完整真人试采、实际图像频率审核及人工画面复核；这些是接收端签收条件。运行时出现的MediaPipe遥测连接失败没有使模型加载或本轮功能检查失败。

历史原始数据未重写，Hold文字修正只影响本次之后的新记录。正式数据需审核、回放、人工批准和后续训练格式转换；本次没有执行π0.5训练。
