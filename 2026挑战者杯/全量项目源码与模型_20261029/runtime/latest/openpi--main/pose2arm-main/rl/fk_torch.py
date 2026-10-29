"""Batched forward kinematics for OpenArm in PyTorch.

Mirrors the NumPy FK in ik_solver.py but vectorized over a batch of
joint-angle configurations so it runs on GPU inside the Isaac Lab reward.

Shoulder (base link) is always at the origin [0, 0, 0] by convention.
Returns elbow and wrist/tool positions for an (N, 7) batch of joint angles.
"""
import torch

# --- constants from ik_solver.py (kept as plain tensors, registered below) ---

_RIGHT_ORIGINS = torch.tensor([
    [0.0, -0.0625, 0.0],
    [0.0, -0.0600000000000001, 0.0],
    [0.0, 0.0, -0.0662500000000008],
    [0.0, 0.0, -0.15375],
    [0.0, 0.0, -0.0955000000000005],
    [0.0, 0.0, -0.120499999999998],
    [0.0, 0.0, 0.0],
], dtype=torch.float32)

_RIGHT_AXES = torch.tensor([
    [0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
], dtype=torch.float32)

_LEFT_ORIGINS = torch.tensor([
    [0.0, 0.0625, 0.0],
    [0.0, 0.0600000000000001, 0.0],
    [0.0, 0.0, -0.0662500000000008],
    [0.0, 0.0, -0.15375],
    [0.0, 0.0, -0.0955000000000005],
    [0.0, 0.0, -0.120499999999998],
    [0.0, 0.0, 0.0],
], dtype=torch.float32)

_LEFT_AXES = torch.tensor([
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
], dtype=torch.float32)

_TOOL_TRANSLATION = torch.tensor([0.0205, 0.0, 0.0], dtype=torch.float32)
_ELBOW_IDX = 3  # zero-based: record position before applying joint-4 rotation


def _rodrigues(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Batch Rodrigues rotation.

    axis  : (3,)  fixed unit axis
    angle : (N,)  per-env angles in radians
    returns (N, 3, 3) rotation matrices
    """
    ax, ay, az = axis[0], axis[1], axis[2]
    c = torch.cos(angle)   # (N,)
    s = torch.sin(angle)
    C = 1.0 - c
    return torch.stack([
        c + ax*ax*C,    ax*ay*C - az*s, ax*az*C + ay*s,
        ay*ax*C + az*s, c + ay*ay*C,    ay*az*C - ax*s,
        az*ax*C - ay*s, az*ay*C + ax*s, c + az*az*C,
    ], dim=-1).reshape(-1, 3, 3)


def batch_fk(
    q: torch.Tensor,
    origins: torch.Tensor,
    axes: torch.Tensor,
    tool: torch.Tensor,
    elbow_idx: int = _ELBOW_IDX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward kinematics for a batch of joint configurations.

    q       : (N, 7) joint angles in radians
    origins : (7, 3) joint origin offsets
    axes    : (7, 3) joint rotation axes
    tool    : (3,)   tool-frame translation from last joint
    returns   elbow (N, 3), wrist/tool (N, 3)  — both in the arm's base frame
    """
    N = q.shape[0]
    dev = q.device

    # Move constants to the same device as q (cached after first call)
    origins = origins.to(dev)
    axes = axes.to(dev)
    tool = tool.to(dev)

    T_rot = torch.eye(3, device=dev, dtype=q.dtype).unsqueeze(0).expand(N, -1, -1).clone()
    T_pos = torch.zeros(N, 3, device=dev, dtype=q.dtype)
    elbow = None

    for i in range(7):
        # translate by joint origin in current frame
        T_pos = T_pos + torch.einsum("nij,j->ni", T_rot, origins[i].to(q.dtype))
        if i == elbow_idx:
            elbow = T_pos.clone()
        # rotate by joint angle
        R_i = _rodrigues(axes[i].to(q.dtype), q[:, i])
        T_rot = torch.bmm(T_rot, R_i)

    # tool translation
    T_pos = T_pos + torch.einsum("nij,j->ni", T_rot, tool.to(q.dtype))
    return elbow, T_pos  # (N, 3), (N, 3)


def fk_right(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FK for the right arm. q: (N, 7). Returns (elbow, wrist)."""
    return batch_fk(q, _RIGHT_ORIGINS, _RIGHT_AXES, _TOOL_TRANSLATION)


def fk_left(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FK for the left arm. q: (N, 7). Returns (elbow, wrist)."""
    return batch_fk(q, _LEFT_ORIGINS, _LEFT_AXES, _TOOL_TRANSLATION)


def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True).clamp(min=eps))
