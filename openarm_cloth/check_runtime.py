"""Check B scenes, shared imports and model assets without opening a camera."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

from member_b_scene import ROOT, SCENE, build_scene
from live_motion_scene import build_motion_scene
from bimanual_teleop import BimanualController
from generic_object_scene import build_object_model
from collision_safety import DEFAULT_XML


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-models', action='store_true',
                        help='Create the actual MediaPipe CPU detectors; no camera input.')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    checkout = ROOT.parent
    if not str(checkout).isascii():
        raise RuntimeError('Use an ASCII-only checkout path for Windows MuJoCo.')
    model, _ = build_scene()
    motion, _ = build_motion_scene()
    tabletop = build_object_model()
    packages = {name: importlib.metadata.version(name) for name in (
        'mujoco', 'numpy', 'scipy', 'opencv-python', 'mediapipe', 'glfw', 'Pillow')}
    imports = {}
    for name in ('member_b_scene', 'live_motion_scene', 'human_teleop_core',
                 'retarget', 'hand_association', 'rrt_connect_planner', 'collision_safety',
                 'tabletop_perception', 'tabletop_fixture', 'generic_object_scene',
                 'generic_grasp_planner', 'demo_tabletop_objects'):
        path = Path(importlib.import_module(name).__file__).resolve()
        if not path.is_relative_to(checkout):
            raise RuntimeError(f'Stale import {name}: {path}')
        imports[name] = path.relative_to(checkout).as_posix()
    models = checkout / 'pose2arm-main' / 'models'
    assets = [models / name for name in (
        'hand_landmarker.task', 'pose_landmarker_lite.task', 'pose_landmarker_full.task',
        'pose_landmarker_heavy.task')]
    missing = [str(path) for path in assets if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing MediaPipe assets. Run setup_models.py: ' + ', '.join(missing))
    if args.check_models:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        for path in assets:
            options = python.BaseOptions(model_asset_path=str(path), delegate=python.BaseOptions.Delegate.CPU)
            if path.name.startswith('hand_'):
                detector = vision.HandLandmarker.create_from_options(
                    vision.HandLandmarkerOptions(base_options=options))
            else:
                detector = vision.PoseLandmarker.create_from_options(
                    vision.PoseLandmarkerOptions(base_options=options))
            detector.close()
    report = dict(python=sys.executable, python_version=sys.version.split()[0],
                  packages=packages, checkout=str(checkout), scene=str(SCENE),
                  grasp_scene=dict(nq=model.nq, nu=model.nu),
                  motion_scene=dict(nq=motion.nq, nu=motion.nu),
                  tabletop_scene=dict(nq=tabletop.nq, nu=tabletop.nu,
                                      physical_boundary_geoms=sum(
                                          tabletop.geom(i).name.startswith('tabletop_rim_')
                                          for i in range(tabletop.ngeom))),
                  legacy_A_diagnostic_model=str(DEFAULT_XML), imports=imports,
                  assets={path.name: path.stat().st_size for path in assets},
                  detectors_created=args.check_models, verdict='PASS')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
