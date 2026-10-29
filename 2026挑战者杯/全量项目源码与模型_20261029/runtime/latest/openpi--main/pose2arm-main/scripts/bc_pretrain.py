#!/usr/bin/env python3
"""Behavioral Cloning pretraining for the dynamic-tracking IK policy.

Trains the policy network with MSE loss on (obs → joint_angles) pairs from the
dataset, before PPO fine-tuning. Does NOT require Isaac Sim.

Obs layout (92D, matching env._get_observations):
  target_t(18) | target_t+1(18) | sin(q)(14) | cos(q)(14) | q_vel/20.944(14) | prev_action(14)

At the IK reference state q_vel = 0 and prev_action = normalized(q_IK),
matching the env RSI reset state. The RL fine-tuning teaches the policy to
use velocity and action history; BC provides a warm start for the IK mapping.

Architecture matches skrl GaussianMixin [512, 256, 128] ELU → 14D exactly,
so the output checkpoint can be loaded directly by train_retarget.py.

Usage:
    python3 scripts/bc_pretrain.py
    python3 scripts/bc_pretrain.py --epochs 300 --lr 5e-4
    # Then launch RL from BC weights:
    python3 train_retarget.py --headless --checkpoint logs/bc_pretrained.pt
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from openarm_common import (
    LEFT_JOINT_LIMITS, RIGHT_JOINT_LIMITS,
)

# Joint limits stacked in dataset order: [left(7), right(7)]
_Q_LOW  = np.concatenate([LEFT_JOINT_LIMITS[:, 0], RIGHT_JOINT_LIMITS[:, 0]]).astype(np.float32)
_Q_HIGH = np.concatenate([LEFT_JOINT_LIMITS[:, 1], RIGHT_JOINT_LIMITS[:, 1]]).astype(np.float32)


# q_vel normalisation divisor — matches env._get_observations
_VEL_NORM = 20.944   # max velocity limit (rad/s, joints 5-7)


def _next_target(target_obs: np.ndarray, n_half: int) -> np.ndarray:
    """Build next-frame lookahead (N, 18), clamped at clip boundary.

    Dataset layout: [0:n_half] = original clip, [n_half:2*n_half] = mirror clip.
    Mirrors env._pre_physics_step: advance frame_ptr then clamp at N_half-1.
    """
    N = len(target_obs)
    nxt = np.empty_like(target_obs)
    # Original clip: i -> min(i+1, n_half-1)
    nxt[:n_half - 1] = target_obs[1:n_half]
    nxt[n_half - 1]  = target_obs[n_half - 1]
    # Mirror clip: same within its own half
    nxt[n_half:N - 1] = target_obs[n_half + 1:N]
    nxt[N - 1]         = target_obs[N - 1]
    return nxt


def build_obs(target_obs: np.ndarray, next_obs: np.ndarray,
              joint_angles: np.ndarray) -> np.ndarray:
    """Build 92D obs identical to env._get_observations at the RSI reset state.

    q_vel = 0 (stationary at reset).
    prev_action = normalized IK solution (env initialises _actions_norm = q_norm at reset).
    """
    sin_q    = np.sin(joint_angles)          # (N, 14)
    cos_q    = np.cos(joint_angles)          # (N, 14)
    q_vel    = np.zeros_like(joint_angles)   # (N, 14)
    prev_act = joint_to_action(joint_angles) # (N, 14) — matches env._actions_norm at reset
    return np.concatenate([target_obs, next_obs, sin_q, cos_q, q_vel, prev_act],
                          axis=1).astype(np.float32)  # (N, 92)


def joint_to_action(joint_angles: np.ndarray) -> np.ndarray:
    """Convert joint angles to [-1, 1] action space (inverse of env._unscale_actions)."""
    a = 2.0 * (joint_angles - _Q_LOW) / (_Q_HIGH - _Q_LOW) - 1.0
    return np.clip(a, -1.0, 1.0).astype(np.float32)


class PolicyNet(nn.Module):
    """Exact replica of the skrl GaussianMixin policy net_container."""
    def __init__(self, obs_dim: int = 92, act_dim: int = 14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ELU(),
            nn.Linear(512, 256),     nn.ELU(),
            nn.Linear(256, 128),     nn.ELU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_checkpoint(net: PolicyNet, log_std_init: float = -1.0) -> dict:
    """Pack weights into skrl checkpoint format (loadable by runner.agent.load)."""
    sd = net.net.state_dict()
    policy = {
        "net_container.0.weight": sd["0.weight"],
        "net_container.0.bias":   sd["0.bias"],
        "net_container.2.weight": sd["2.weight"],
        "net_container.2.bias":   sd["2.bias"],
        "net_container.4.weight": sd["4.weight"],
        "net_container.4.bias":   sd["4.bias"],
        "net_container.6.weight": sd["6.weight"],
        "net_container.6.bias":   sd["6.bias"],
        "log_std_parameter":      torch.full((14,), log_std_init),
    }
    return {"policy": policy}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=200,
                        help="Training epochs over the dataset")
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--noise-std",  type=float, default=0.05,
                        help="Noise added to q_IK in obs (matches RSI perturbation in RL env)")
    parser.add_argument("--log-std",    type=float, default=-1.0,
                        help="Initial log_std saved into checkpoint (skrl default=-1.0)")
    parser.add_argument("--data",       type=str,   default="data/rl_poses.npz",
                        help="Dataset npz (must contain target_obs + joint_angles)")
    parser.add_argument("--out",        type=str,   default="logs/bc_pretrained.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_path = REPO_ROOT / args.data
    raw = np.load(data_path)
    target_obs   = raw["target_obs"].astype(np.float32)    # (N, 18)
    joint_angles = raw["joint_angles"].astype(np.float32)  # (N, 14)
    N = len(target_obs)
    n_half = N // 2   # frames per clip (original or mirror)
    print(f"Dataset: {N} samples ({n_half} per clip)")

    next_obs = _next_target(target_obs, n_half)             # (N, 18) one-step lookahead

    # Noise augmentation: add ±noise_std to q_IK in obs so the policy learns to
    # correct back to q_IK when slightly perturbed — matches RSI noise in RL env.
    # Target action always stays at q_IK. q_vel slot stays 0 (static IK state).
    q_t           = torch.tensor(joint_angles).to(device)   # (N, 14)
    target_norm_t = torch.tensor(target_obs).to(device)   # (N, 18) — unit vectors
    next_norm_t   = torch.tensor(next_obs).to(device)     # (N, 18)
    actions_t     = torch.tensor(joint_to_action(joint_angles)).to(device)  # (N, 14)

    q_low_t  = torch.tensor(_Q_LOW,  device=device)
    q_high_t = torch.tensor(_Q_HIGH, device=device)

    # Base obs without noise (used for eval)
    obs_np = build_obs(target_obs, next_obs, joint_angles)
    obs_t  = torch.tensor(obs_np).to(device)

    loader = DataLoader(
        TensorDataset(target_norm_t, next_norm_t, q_t, actions_t),
        batch_size=args.batch_size, shuffle=True, drop_last=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    net = PolicyNet(obs_dim=92).to(device)
    opt = optim.Adam(net.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n{'epoch':>6}  {'MSE loss':>10}  {'MAE (rad)':>10}")
    for epoch in range(1, args.epochs + 1):
        net.train()
        total_loss = 0.0
        for tgt_b, nxt_b, q_b, a_b in loader:
            # Add noise to q in obs; target action is still clean q_IK.
            # q_vel stays 0 (robot is at the static IK state in RSI reset).
            q_noisy    = q_b + args.noise_std * torch.randn_like(q_b)
            q_vel_b    = torch.zeros_like(q_b)
            # prev_action = normalized noisy IK solution, matching env._actions_norm at reset
            prev_act_b = (2.0 * (q_noisy - q_low_t) / (q_high_t - q_low_t) - 1.0).clamp(-1.0, 1.0)
            x_b = torch.cat([tgt_b, nxt_b,
                              torch.sin(q_noisy), torch.cos(q_noisy),
                              q_vel_b, prev_act_b], dim=-1)
            pred = net(x_b)
            loss = nn.functional.mse_loss(pred, a_b)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(q_b)
        sched.step()

        if epoch % 20 == 0 or epoch == 1:
            net.eval()
            with torch.no_grad():
                pred_all = net(obs_t)
                q_pred = q_low_t + (pred_all + 1.0) * 0.5 * (q_high_t - q_low_t)
                q_true = q_low_t + (actions_t + 1.0) * 0.5 * (q_high_t - q_low_t)
                mae_rad = (q_pred - q_true).abs().mean().item()
            print(f"{epoch:>6d}  {total_loss/N:>10.6f}  {mae_rad:>10.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(make_checkpoint(net, args.log_std), out_path)
    print(f"\nSaved BC checkpoint → {out_path}")
    print("Launch RL with:")
    print(f"  python3 train_retarget.py --headless --checkpoint {out_path}")


if __name__ == "__main__":
    main()
