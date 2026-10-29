"""Isaac Lab RL environment: dual OpenArm dynamic trajectory tracking.

The agent controls 14 joint angles (7 per arm). Each episode tracks a video
clip: the target pose advances one frame per control step (DeepMimic-style).

Observation (92-D):
  [target_t (18) | target_{t+1} (18) | sin(q) (14) | cos(q) (14) | q_vel (14) | prev_action (14)]

Episode:
  - Each env samples a random start frame within the original or mirror clip.
  - RSI: robot is initialised to the IK solution at the start frame.
  - Early termination: forearm direction deviates > 60° from target (cos < 0.5).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from .fk_torch import fk_left, fk_right, normalize
from .robot_cfg import OPENARM_BI_CFG

from openarm_common import (
    TARGET_OBS_DIM,
    T_UPPER_L, T_FORE_L, T_PLANE_L,
    T_UPPER_R, T_FORE_R, T_PLANE_R,
)

_RIGHT_HOME = torch.tensor([0.0,  0.6, 0.0, 0.9, 0.0, 0.0, 0.0], dtype=torch.float32)
_LEFT_HOME  = torch.tensor([0.0, -0.6, 0.0, 0.9, 0.0, 0.0, 0.0], dtype=torch.float32)

# Reward kernel widths
_SIGMA_DIR   = 0.55   # direction error (unit-vector L2); tighter → sharpens gradient for small errors
_SIGMA_VEL   = 1.5    # joint velocity smoothness; at 2-4 rad/s gives reward 0.5-0.8
_SIGMA_RATE  = 0.12   # action-rate penalty (normalised [-1,1] space)
_SIGMA_ELBOW = 0.30   # explicit elbow bend angle error (rad); ~17° half-width

# Shoulder offsets (base frame) — subtract to make FK positions shoulder-relative,
# matching the dataset convention for elbow and wrist targets.
_SHOULDER_L = torch.tensor([0.0,  0.0625, 0.0], dtype=torch.float32)
_SHOULDER_R = torch.tensor([0.0, -0.0625, 0.0], dtype=torch.float32)

_TARGET_DIM = TARGET_OBS_DIM   # 18
_ARM_JOINTS = 14               # 7L + 7R

# obs = target_t(18) | target_{t+1}(18) | sin(q)(14) | cos(q)(14) | q_vel(14) | prev_action(14)
_OBS_DIM = _TARGET_DIM * 2 + _ARM_JOINTS * 4   # 92

_W_UPPER  = 0.40   # shoulder→elbow direction (upper arm)
_W_FORE   = 0.40   # elbow→wrist direction (forearm)
_W_PLANE  = 0.22   # arm-plane normal (elbow swivel) — up from 0.12; does work of DeepMimic pose reward
_W_ELBOW  = 0.15   # explicit elbow bend angle — direct gradient through joint 4
_W_SMOOTH = 0.20   # velocity smoothness (recalibrated: σ=1.5 gives signal during motion)
_W_RATE   = 0.20   # action-rate penalty (direct oscillation suppression)
_REWARD_NORM = (_W_UPPER + _W_FORE + _W_PLANE + _W_ELBOW) * 2 + _W_SMOOTH + _W_RATE

# Terminate if forearm direction deviates > 60° from target (cos < 0.5)
_FORE_TERM_COS = 0.5


@configclass
class OpenArmRetargetEnvCfg(DirectRLEnvCfg):
    observation_space: int = _OBS_DIM    # 92
    action_space:      int = _ARM_JOINTS  # 14
    state_space:       int = 0

    # 90 Hz physics, 3 substeps → 30 Hz policy
    decimation:       int   = 3
    episode_length_s: float = 5.0   # 150 steps at 30 Hz
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 90.0, render_interval=3)

    # Stage 1: no contact penalty.  Set ~0.10 for Stage 2.
    w_contact: float = 0.0

    robot_cfg: ArticulationCfg = OPENARM_BI_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256, env_spacing=2.0, replicate_physics=True
    )
    pose_dataset_path: str = "data/rl_poses.npz"

    # Small RSI noise prevents exact state memorisation
    home_noise_std: float = 0.05


class OpenArmRetargetEnv(DirectRLEnv):
    cfg: OpenArmRetargetEnvCfg

    # ── Initialisation ────────────────────────────────────────────────────────

    def __init__(self, cfg: OpenArmRetargetEnvCfg,
                 render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._left_ids, _  = self.robot.find_joints("openarm_left_joint[1-7]")
        self._right_ids, _ = self.robot.find_joints("openarm_right_joint[1-7]")
        self._arm_ids = self._left_ids + self._right_ids

        limits = self.robot.data.soft_joint_pos_limits[0, self._arm_ids, :]
        self._q_low  = limits[:, 0]
        self._q_high = limits[:, 1]

        # Dataset: (M, 18) poses and (M, 14) IK joint angles
        dataset_path = REPO_ROOT / cfg.pose_dataset_path
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Pose dataset not found: {dataset_path}\n"
                "Run: python3 scripts/generate_ik_dataset.py"
            )
        raw = np.load(dataset_path)
        self._pose_bank = torch.tensor(
            raw["target_obs"], dtype=torch.float32, device=self.device
        )  # (M, 18) — [0:N_half] original, [N_half:] mirror, both temporally ordered

        M = len(self._pose_bank)
        self._N_half = M // 2   # frames per clip (original or mirror)

        if "joint_angles" in raw:
            self._joint_bank = torch.tensor(
                raw["joint_angles"], dtype=torch.float32, device=self.device
            )  # (M, 14)
            print(f"[Env] RSI enabled — {M} total frames ({self._N_half} per clip)")
        else:
            self._joint_bank = None
            print("[Env] RSI disabled (no joint_angles in dataset)")

        # Per-env clip tracking
        # _clip_start: 0 for original clip, N_half for mirror clip
        # _frame_ptr:  current position within [0, N_half)
        self._clip_start = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._frame_ptr  = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Current and one-step-ahead targets  (N_envs, 18)
        self._target      = torch.zeros(self.num_envs, _TARGET_DIM, device=self.device)
        self._next_target = torch.zeros(self.num_envs, _TARGET_DIM, device=self.device)

        # Written by _get_rewards, read by _get_dones
        self._fore_term  = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        home = torch.cat([_LEFT_HOME, _RIGHT_HOME]).to(self.device)
        self._home = home.unsqueeze(0).expand(self.num_envs, -1)

        # Previous normalised actions for action-rate penalty and obs
        self._prev_actions_norm = torch.zeros(self.num_envs, _ARM_JOINTS, device=self.device)
        self._actions_norm      = torch.zeros(self.num_envs, _ARM_JOINTS, device=self.device)

        # Shoulder offsets on device — pre-moved to avoid per-step GPU allocation
        self._sh_L = _SHOULDER_L.to(self.device)
        self._sh_R = _SHOULDER_R.to(self.device)

    # ── Scene setup ───────────────────────────────────────────────────────────

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        contact_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/openarm_(left|right)_link[1-7]",
            history_length=1,
            debug_vis=False,
        )
        self.contact_sensor = ContactSensor(contact_cfg)
        self.scene.sensors["contact"] = self.contact_sensor

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ── Step hooks ────────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Capture target for this step BEFORE advancing the pointer.
        # The reward will measure how well the robot matches this frame.
        abs_cur = self._clip_start + self._frame_ptr
        self._target = self._pose_bank[abs_cur]

        # Advance: clamp at clip end rather than wrapping (episode will end first)
        self._frame_ptr = (self._frame_ptr + 1).clamp(max=self._N_half - 1)

        # One-step lookahead for the observation
        abs_nxt = self._clip_start + self._frame_ptr
        self._next_target = self._pose_bank[abs_nxt]

        self._prev_actions_norm = self._actions_norm.clone()
        self._actions_norm = actions.clamp(-1.0, 1.0)
        self._actions = self._unscale_actions(self._actions_norm)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._actions, joint_ids=self._arm_ids)

    def _get_observations(self) -> dict:
        q     = self.robot.data.joint_pos[:, self._arm_ids]         # (N, 14)
        q_vel = self.robot.data.joint_vel[:, self._arm_ids]         # (N, 14)
        q_enc = torch.cat([torch.sin(q), torch.cos(q)], dim=-1)    # (N, 28)
        # target slots are unit vectors — no scaling needed
        # q_vel normalised by max limit (20.944 rad/s) so magnitude matches sin/cos
        q_vel_n = q_vel * (1.0 / 20.944)                           # (N, 14) in ~[-1, 1]
        obs = torch.cat([self._target, self._next_target,
                         q_enc, q_vel_n,
                         self._actions_norm], dim=-1)               # (N, 92)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        q     = self.robot.data.joint_pos[:, self._arm_ids]
        q_vel = self.robot.data.joint_vel[:, self._arm_ids]
        contact_forces = self.contact_sensor.data.net_forces_w
        reward, fore_term = _compute_reward(
            q[:, :7], q[:, 7:], q_vel, self._target,
            self._actions_norm, self._prev_actions_norm,
            contact_forces, self.cfg.w_contact,
            self._sh_L, self._sh_R,
        )
        self._fore_term = fore_term
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # _fore_term is written by _get_rewards; Isaac Lab's step() always calls
        # rewards before dones, so this read is safe in normal training.
        time_out   = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self._fore_term
        return terminated, time_out

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n = len(env_ids)

        # Sample which clip (0 = original, 1 = mirror) and start frame.
        # Leave enough headroom so the episode doesn't overrun the clip.
        max_start = max(1, self._N_half - self.max_episode_length - 1)
        clip_idx = torch.randint(0, 2, (n,), device=self.device)
        self._clip_start[env_ids] = clip_idx * self._N_half
        self._frame_ptr[env_ids]  = torch.randint(0, max_start, (n,), device=self.device)

        # Pre-set targets so the first _get_observations call after reset is correct
        abs_cur = self._clip_start[env_ids] + self._frame_ptr[env_ids]
        abs_nxt = self._clip_start[env_ids] + (self._frame_ptr[env_ids] + 1).clamp(max=self._N_half - 1)
        self._target[env_ids]      = self._pose_bank[abs_cur]
        self._next_target[env_ids] = self._pose_bank[abs_nxt]

        # RSI: initialise robot to the IK solution at the episode start frame
        if self._joint_bank is not None:
            q_reset = self._joint_bank[abs_cur].clone()
            noise   = torch.randn(n, _ARM_JOINTS, device=self.device) * self.cfg.home_noise_std
            q_reset = (q_reset + noise).clamp(self._q_low, self._q_high)
        else:
            noise   = torch.randn(n, _ARM_JOINTS, device=self.device) * self.cfg.home_noise_std
            q_reset = (self._home[env_ids] + noise).clamp(self._q_low, self._q_high)

        joint_pos_all = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel_all = self.robot.data.default_joint_vel[env_ids].clone()
        for local_i, arm_jid in enumerate(self._arm_ids):
            joint_pos_all[:, arm_jid] = q_reset[:, local_i]
            joint_vel_all[:, arm_jid] = 0.0
        self.robot.write_joint_state_to_sim(joint_pos_all, joint_vel_all, None, env_ids)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

        # Initialise action history to the RSI position so the first-step rate reward
        # sees (policy_output vs IK_start) instead of (IK_start vs zero).
        q_norm = (2.0 * (q_reset - self._q_low) / (self._q_high - self._q_low) - 1.0).clamp(-1.0, 1.0)
        self._fore_term[env_ids]         = False
        self._prev_actions_norm[env_ids] = q_norm
        self._actions_norm[env_ids]      = q_norm

    # ── Helper ────────────────────────────────────────────────────────────────

    def _unscale_actions(self, a: torch.Tensor) -> torch.Tensor:
        """[-1, 1] → actual joint-angle range."""
        return self._q_low + (a + 1.0) * 0.5 * (self._q_high - self._q_low)


# ── Reward (module-level) ──────────────────────────────────────────────────────

def _dir_reward(actual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Exponential reward for direction-vector alignment. Returns (N,) in [0, 1]."""
    err = torch.sum((normalize(actual) - target) ** 2, dim=-1)
    return torch.exp(-err / (2.0 * _SIGMA_DIR ** 2))


def _smooth_reward(q_vel: torch.Tensor) -> torch.Tensor:
    """Penalise high joint velocities. Returns (N,) in [0, 1]."""
    err = torch.mean(q_vel ** 2, dim=-1)
    return torch.exp(-err / (2.0 * _SIGMA_VEL ** 2))


def _rate_reward(a: torch.Tensor, a_prev: torch.Tensor) -> torch.Tensor:
    """Penalise action changes (oscillation suppression). Returns (N,) in [0, 1]."""
    err = torch.mean((a - a_prev) ** 2, dim=-1)
    return torch.exp(-err / (2.0 * _SIGMA_RATE ** 2))


def _elbow_reward(upper: torch.Tensor, fore: torch.Tensor,
                  t_upper: torch.Tensor, t_fore: torch.Tensor) -> torch.Tensor:
    """Explicit elbow bend angle reward. Returns (N,) in [0, 1].
    Angle between upper-arm and forearm direction vectors — direct gradient through joint 4.
    """
    cos_fk  = torch.sum(upper * fore,     dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
    cos_tgt = torch.sum(t_upper * t_fore, dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
    err = (torch.acos(cos_fk) - torch.acos(cos_tgt)) ** 2
    return torch.exp(-err / (2.0 * _SIGMA_ELBOW ** 2))


def _contact_penalty(forces_w: torch.Tensor, threshold: float = 5.0) -> torch.Tensor:
    """Fraction of arm links with contact force above threshold. Returns (N,) in [-1, 0]."""
    if forces_w.dim() == 4:   # (N, 1, B, 3) — history dim not squeezed by some IsaacLab versions
        forces_w = forces_w.squeeze(1)
    mag = torch.norm(forces_w, dim=-1)
    return -(mag > threshold).float().mean(dim=-1)


def _compute_reward(
    q_left:         torch.Tensor,           # (N, 7)
    q_right:        torch.Tensor,           # (N, 7)
    q_vel:          torch.Tensor,           # (N, 14)
    target:         torch.Tensor,           # (N, 18) — all unit vectors
    actions_norm:   torch.Tensor,           # (N, 14) current normalised action [-1,1]
    prev_actions:   torch.Tensor,           # (N, 14) previous normalised action [-1,1]
    contact_forces: torch.Tensor | None,    # (N, B, 3)
    w_contact:      float = 0.0,
    sh_L:           torch.Tensor | None = None,
    sh_R:           torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:     # (reward (N,), fore_term (N,))
    elbow_L, wrist_L = fk_left(q_left)
    elbow_R, wrist_R = fk_right(q_right)

    if sh_L is None:
        sh_L = _SHOULDER_L.to(q_left.device, q_left.dtype)
    if sh_R is None:
        sh_R = _SHOULDER_R.to(q_right.device, q_right.dtype)

    # Direction vectors from FK (shoulder→elbow, elbow→wrist)
    upper_L = normalize(elbow_L - sh_L)
    fore_L  = normalize(wrist_L - elbow_L)
    upper_R = normalize(elbow_R - sh_R)
    fore_R  = normalize(wrist_R - elbow_R)

    # Arm plane normal: cross(shoulder→elbow, elbow→wrist)
    plane_L = torch.linalg.cross(elbow_L - sh_L, wrist_L - elbow_L)
    plane_R = torch.linalg.cross(elbow_R - sh_R, wrist_R - elbow_R)

    r = (
        _W_UPPER  * _dir_reward(upper_L, target[:, T_UPPER_L])
      + _W_FORE   * _dir_reward(fore_L,  target[:, T_FORE_L])
      + _W_PLANE  * _dir_reward(plane_L, target[:, T_PLANE_L])
      + _W_ELBOW  * _elbow_reward(upper_L, fore_L, target[:, T_UPPER_L], target[:, T_FORE_L])
      + _W_UPPER  * _dir_reward(upper_R, target[:, T_UPPER_R])
      + _W_FORE   * _dir_reward(fore_R,  target[:, T_FORE_R])
      + _W_PLANE  * _dir_reward(plane_R, target[:, T_PLANE_R])
      + _W_ELBOW  * _elbow_reward(upper_R, fore_R, target[:, T_UPPER_R], target[:, T_FORE_R])
      + _W_SMOOTH * _smooth_reward(q_vel)
      + _W_RATE   * _rate_reward(actions_norm, prev_actions)
    )
    r = r / _REWARD_NORM

    if w_contact > 0.0 and contact_forces is not None:
        r = r + w_contact * _contact_penalty(contact_forces)

    # Early termination: forearm direction deviates > 60° from target (cos < 0.5)
    cos_L = torch.sum(fore_L * target[:, T_FORE_L], dim=-1)
    cos_R = torch.sum(fore_R * target[:, T_FORE_R], dim=-1)
    fore_term = (cos_L < _FORE_TERM_COS) | (cos_R < _FORE_TERM_COS)

    return r, fore_term
