#!/usr/bin/env python3
"""Re-run IK pass on existing target_obs with updated weights.

Use when target_obs is already extracted and only joint_angles need regenerating
(e.g., after changing _OFFLINE_WEIGHTS or IK restart logic).

Usage:
    python3 scripts/rerun_ik.py
    python3 scripts/rerun_ik.py --input data/rl_poses.npz --output data/rl_poses.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_ik_dataset import (
    _run_ik_pass, _OFFLINE_WEIGHTS,
    OpenArmIK, _APPROX_SIGNS,
    N_RESTARTS, N_RESTARTS_14D, MAX_NFEV, MAX_NFEV_14D,
)
from ik_solver import POSE_LIKENESS_WEIGHTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",        default="data/rl_poses.npz")
    p.add_argument("--output",       default="data/rl_poses.npz")
    p.add_argument("--restarts",     type=int, default=N_RESTARTS)
    p.add_argument("--restarts-14d", type=int, default=N_RESTARTS_14D)
    p.add_argument("--max-nfev",     type=int, default=MAX_NFEV)
    p.add_argument("--max-nfev-14d", type=int, default=MAX_NFEV_14D)
    args = p.parse_args()

    raw = np.load(args.input)
    target_obs = raw["target_obs"]           # (2N, 18)
    fps        = float(raw.get("fps", 25.0))
    frame_indices = raw.get("frame_indices", np.arange(len(target_obs)))

    N = len(target_obs)
    N_half = N // 2
    target_obs_orig   = target_obs[:N_half]
    target_obs_mirror = target_obs[N_half:]

    print(f"Dataset: {N} total frames ({N_half} orig + {N_half} mirror)")
    print(f"Active IK weights: { {k:v for k,v in _OFFLINE_WEIGHTS.items() if v != 0.0} }")

    iks = {"left": OpenArmIK(side="left"), "right": OpenArmIK(side="right")}

    print(f"\n[1/2] IK on {N_half} original frames ...")
    joint_angles_orig, _ = _run_ik_pass(
        target_obs_orig, iks, label="orig ",
        n_restarts=args.restarts, n_restarts_14d=args.restarts_14d,
        max_nfev=args.max_nfev, max_nfev_14d=args.max_nfev_14d,
    )

    print(f"\n[2/2] IK on {N_half} mirrored frames ...")
    warm_l = joint_angles_orig[:, 7:] * _APPROX_SIGNS
    warm_r = joint_angles_orig[:, :7] * _APPROX_SIGNS
    joint_angles_mirror, _ = _run_ik_pass(
        target_obs_mirror, iks, warm_starts_l=warm_l, warm_starts_r=warm_r,
        label="mirror ",
        n_restarts=args.restarts, n_restarts_14d=args.restarts_14d,
        max_nfev=args.max_nfev, max_nfev_14d=args.max_nfev_14d,
    )

    joint_angles_all = np.concatenate([joint_angles_orig, joint_angles_mirror], axis=0)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             target_obs=target_obs,
             joint_angles=joint_angles_all,
             frame_indices=frame_indices,
             fps=fps)
    print(f"\nSaved → {out}  ({joint_angles_all.shape})")


if __name__ == "__main__":
    main()
