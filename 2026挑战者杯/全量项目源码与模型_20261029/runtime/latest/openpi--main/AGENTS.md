# Project working agreement

The user has selected Member B's environment as the basis for all future A/B integration.

- Treat `openarm_cloth/` as the production simulator, perception, calibration,
  robot model, gripper, IK and actuator implementation. Its current grasp scene
  is `simulation/v1/droid_cube_scene_perception.xml`, with physical workspace
  configuration in `simulation/v1/task_workspace.json`. Live motion-only scenes
  are derived through `live_motion_scene.py`.
- Use the repository-root `run.ps1` and the configured B Python for normal
  runs and verification. Read `RUNNING.md` for commands and environment setup.
  Keep machine-specific interpreter paths in ignored `.runtime.local.json`.
- Member A contributes human landmark extraction, intent, relative motion and
  gripper signals. Route robot execution through B's controller; do not add two
  independently produced joint-angle commands together.
- Maintain two separate workflows: free-motion reproduction and optional B
  tabletop grasping. Never inject the grasp task into free-motion replay.
  Default `fusion` runs `run_a_reference_on_b.py`; `--reference-mode clean`
  rebuilds pure imitation from A body observations and measured hand poses via
  `clean_motion_reference.py`, excluding historical avoidance joint values.
  Verified RTMW image-plane fallback may hold depth, never use its nonmetric
  model_xyz as robot metres. Unobservable wrists hold independently.
  `--reference-mode legacy` is only an old-result comparison. Both use B's
  robot-only `build_motion_scene`. Default is explicitly KINEMATIC, like A's replay;
  `--execution dynamic` separately tests B's simulated actuators. No task
  perception, grasp assistance, replanning, or safety projection may override
  this mode. Never label kinematic fidelity as physical-actuator validation.
  Default `fusion-replay` uses native MuJoCo 3D controls and the latest successful
  `outputs/a_motion` recording. Keep the old `landmark-fusion`/`outputs/ab_motion`
  and `video-skill`/`outputs/video_fusion` evidence separate. Grasp mode is unchanged.
- Motion-only dynamic tracking uses continuous interpolation/velocity feedforward
  in a separate subclass with unchanged B gains/torque limits; never alter B's
  autonomous grasp controller merely to tune free-motion following.
- B's slide fingers use metres (0 to 0.044), while A's historical hinge-finger
  tools use radians. Convert normalized human closure at the B interface.
  Do not use the historical A model to certify B scene collision safety.
- Old working copies, the dependency bundle and the independent Notebook were
  moved to `D:\openpi\_archive\cleanup_20261021`. Keep archives out of production
  imports. Do not restore older simulator code over the current B environment.
- Preserve B's existing cameras, calibration, physics and validated grasp path
  when integrating A. Record and verify intentional changes separately.
- Read `openarm_cloth/docs/ab_unified_direction_20261019.md` for current
  capability boundaries and the remaining integration work. Documentation of
  a proposed relative-motion interface is not evidence of its implementation.
- Keep personal videos, detector weights, generated datasets and local outputs
  local unless the user explicitly requests sharing them. Preserve historical
  validation results and source hashes; save new results separately.
- Multi-shape tabletop grasping is an independent optional B mode:
  `tabletop` / `tabletop-ui` / `tabletop-check`. Read
  `openarm_cloth/docs/tabletop_merge_20261020_CN.md` for scope and commands.
  Keep its generated box/cylinder/rim model out of the frozen red-cube and
  motion-only scenes. Do not pass fixture class/size/pose into perception or IK.
  This geometric RGB-D recognizer is not a trained semantic policy, and does
  not imply multi-object clutter grasp planning or an A-demonstration task policy.
- Start with `README.md`, `PROJECT_STATUS_20261021.md` and `RUNNING.md` for the
  current local state. Historical docs are evidence, not current entry points.
  Preserve failed/cancelled user runs as well as successful fixed benchmarks.
