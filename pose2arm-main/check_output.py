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
