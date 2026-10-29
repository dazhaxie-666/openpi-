"""Recorded-state viewing or independent B-actuator action replay (not new control)."""
import argparse
from contextlib import nullcontext
from datetime import datetime
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from member_b_actuators import ActuatorController
from validate_human_task import validate
from single_arm_runtime import recorded_step_durations


def replay(path, execution='states', check=False):
    path = Path(path)
    quality = validate(path)
    if not quality['recording_valid']:
        raise ValueError('Invalid recording: ' + '; '.join(quality['errors']))
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['mujoco_version'] != mujoco.__version__:
        raise ValueError('Use the recording MuJoCo version to load its MJB snapshot')
    if execution not in ('states', 'actions'):
        raise ValueError('Unknown replay execution mode')
    model = mujoco.MjModel.from_binary_path(str(path / manifest['model_file']))
    data = mujoco.MjData(model)
    state_spec = manifest['physics_state_spec']
    with np.load(path / 'episode.npz', allow_pickle=False) as blob:
        states = blob['physics_state'].copy()
        qpos, qvel, actions = blob['qpos'].copy(), blob['qvel'].copy(), blob['action_target'].copy()
        applied_ctrl = blob['ctrl'].copy()
        scene_resets=(blob['shared_scene_reset'].astype(bool).copy()
                      if 'shared_scene_reset' in blob.files else np.zeros(len(states),dtype=bool))
        durations=recorded_step_durations(blob,manifest,len(states)).copy()
    if len(states) < 2:
        raise ValueError('At least two frames are needed to verify replay')
    mujoco.mj_setState(model, data, states[0], state_spec)
    mujoco.mj_forward(model, data)
    # Forward reconstructs derived fields but may update the solver warmstart.
    # Restore the recorded integration state after reconstructing those fields.
    mujoco.mj_setState(model, data, states[0], state_spec)
    control = ActuatorController(model, data)
    steps=np.rint(durations/model.opt.timestep).astype(int)
    if not np.allclose(steps*model.opt.timestep,durations,atol=1e-9,rtol=0):
        raise ValueError('Control interval does not match model physics')
    flags = dict(paused=False, quit=False)
    def key(code):
        if code == 32:
            flags['paused'] = not flags['paused']
        elif code in (81, 113, 27):
            flags['quit'] = True
    context = nullcontext(None)
    if not check:
        from mujoco import viewer as mjviewer
        context = mjviewer.launch_passive(model, data, key_callback=key)
    max_qpos = max_qvel = 0.
    compared = 0
    with context as viewer:
        if viewer is not None:
            viewer.cam.lookat[:] = [.43, 0, .46]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.2, 150, -30
        for index, state in enumerate(states):
            if flags['quit'] or viewer is not None and not viewer.is_running():
                break
            while flags['paused'] and viewer is not None and viewer.is_running() and not flags['quit']:
                viewer.sync()
                time.sleep(.03)
            if execution == 'states':
                mujoco.mj_setState(model, data, state, state_spec)
                mujoco.mj_forward(model, data)
                max_qpos = max(max_qpos, float(np.max(np.abs(data.qpos - qpos[index]))))
                max_qvel = max(max_qvel, float(np.max(np.abs(data.qvel - qvel[index]))))
            else:
                if scene_resets[index]:
                    # A collection reset is an environment boundary, not an
                    # actuator action. Restore its recorded initial state,
                    # then continue physical actuator replay within the trial.
                    mujoco.mj_setState(model,data,state,state_spec);mujoco.mj_forward(model,data)
                    mujoco.mj_setState(model,data,state,state_spec)
                    control=ActuatorController(model,data)
                max_qpos = max(max_qpos, float(np.max(np.abs(data.qpos - qpos[index]))))
                max_qvel = max(max_qvel, float(np.max(np.abs(data.qvel - qvel[index]))))
                action = actions[index]
                control.set_targets(dict(left=action[:7], right=action[9:16]),
                                    dict(left=float(action[7]), right=float(action[16])))
                for substep in range(steps[index]):
                    if index == 0 and substep == 0:
                        # The first command was computed with B's preceding-step
                        # gravity cache. Recomputing it after mj_forward changes
                        # that one command and can amplify drift at contacts.
                        data.ctrl[:] = applied_ctrl[0]
                    else:
                        control.apply()
                    mujoco.mj_step(model, data)
            compared += 1
            if viewer is not None:
                viewer.sync()
                time.sleep(float(durations[index]))
    complete = compared == len(states)
    return dict(verdict='PASS' if complete and max_qpos < 1e-5 and max_qvel < 1e-4 else 'FAIL' if complete else 'CANCELLED',
                execution=execution, frames=compared, max_qpos_error=max_qpos, max_qvel_error=max_qvel,
                physical_actuator_replay=execution == 'actions',
                environment_resets_applied=int(np.count_nonzero(scene_resets)) if execution=='actions' else 0,
                synthetic_input=manifest['synthetic_input'], human_validation_performed=False,
                scope='same-model local replay consistency; recorded environment resets delimit actuator-replayed trials; not a new human test or cross-machine determinism guarantee')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--execution', choices=('states', 'actions'), default='states')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    report = replay(args.recording, args.execution, args.check)
    destination = args.recording / ('replay_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['verdict'] in ('PASS', 'CANCELLED') else 1)


if __name__ == '__main__':
    main()
