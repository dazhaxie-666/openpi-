"""Train dual-arm pose retargeting policy with Isaac Lab + skrl PPO.

Must be run from the pose2arm repo root with the Isaac Lab venv active:

    source ~/isaac_venv/bin/activate
    python3 train_retarget.py
    python3 train_retarget.py --num_envs 64
    python3 train_retarget.py --checkpoint logs/openarm_retarget/.../agent.pt
"""

# ── Isaac Sim must be launched before any other omni/pxr imports ──────────────
import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train OpenArm pose-retarget policy.")
parser.add_argument("--num_envs",   type=int, default=256)
parser.add_argument("--max_iters",  type=int, default=None,
                    help="PPO iterations (not total timesteps).")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Resume from checkpoint path.")
parser.add_argument("--lr",         type=float, default=None,
                    help="Override learning rate (e.g. 1e-4). Useful when fine-tuning from a checkpoint.")
parser.add_argument("--seed",       type=int, default=42)
AppLauncher.add_app_launcher_args(parser)   # adds --headless, --device, etc.
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Everything below runs after Isaac Sim is up ───────────────────────────────
import os
import time
import yaml
from datetime import datetime

import gymnasium as gym
from skrl.utils.runner.torch import Runner

from isaaclab_rl.skrl import SkrlVecEnvWrapper

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import rl  # noqa: F401  — triggers gym.register("Isaac-OpenArmRetarget-Direct-v0")
from rl.env import OpenArmRetargetEnvCfg


def main():
    # ── Env config ────────────────────────────────────────────────────────────
    env_cfg = OpenArmRetargetEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device:
        env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed

    # ── Agent (PPO) config from YAML ─────────────────────────────────────────
    yaml_path = REPO_ROOT / "rl" / "agents" / "skrl_ppo_cfg.yaml"
    with open(yaml_path) as f:
        agent_cfg = yaml.safe_load(f)

    agent_cfg["seed"] = args_cli.seed
    if args_cli.max_iters is not None:
        agent_cfg["trainer"]["timesteps"] = (
            args_cli.max_iters * agent_cfg["agent"]["rollouts"]
        )

    # ── Logging ──────────────────────────────────────────────────────────────
    log_root = str(REPO_ROOT / "logs" / "skrl" /
                   agent_cfg["agent"]["experiment"]["directory"])
    log_dir  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_ppo_torch"
    agent_cfg["agent"]["experiment"]["directory"]       = log_root
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    env_cfg.log_dir = os.path.join(log_root, log_dir)
    print(f"[INFO] logging → {env_cfg.log_dir}")

    # ── Build env ─────────────────────────────────────────────────────────────
    env = gym.make("Isaac-OpenArmRetarget-Direct-v0", cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # ── Train ─────────────────────────────────────────────────────────────────
    runner = Runner(env, agent_cfg)
    if args_cli.checkpoint:
        print(f"[INFO] resuming from {args_cli.checkpoint}")
        runner.agent.load(args_cli.checkpoint)
        if args_cli.lr is not None:
            for pg in runner.agent.optimizer.param_groups:
                pg["lr"] = args_cli.lr
            print(f"[INFO] optimizer LR overridden → {args_cli.lr}")

    t0 = time.time()
    runner.run()
    print(f"[INFO] training done in {time.time() - t0:.0f}s")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
