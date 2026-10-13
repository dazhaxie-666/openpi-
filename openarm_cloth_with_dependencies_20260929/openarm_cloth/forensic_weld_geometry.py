"""Offline geometric forensics for the red-cube grasp penetration report.

Two analysis modes per report:

A) weld-composed: place the cube at the pose recorded in the grasp report
   (interpreted as cube-in-hand-frame) at the home configuration, then measure
   exact MuJoCo distances between the cube and every geom rigidly attached to
   the hand assembly.  These distances are invariant to arm configuration.

B) final-qpos truth: if the report contains result.final_qpos, restore the
   exact joint state, compute the TRUE cube-in-hand / cube-in-TCP poses, match
   them against the recorded weld data, and measure cube-vs-ALL-geom distances.

Usage:
    python forensic_weld_geometry.py report1.json [report2.json ...]
"""

from __future__ import annotations

import itertools
import json
import sys

import mujoco
import numpy as np

BASE = r"D:\openpi\openarm_cloth_with_dependencies_20260929\openarm_cloth"
XML = BASE + r"\simulation\v1\droid_cube_scene.xml"
HAND_BODY = "openarm_left_hand"
CUBE_BODY = "droid_object"
CUBE_HALF = 0.0288
CANDIDATE_FRAMES = ("openarm_left_hand", "openarm_left_hand_tcp", "openarm_left_link8", "openarm_left_link7")


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_conj(a: np.ndarray) -> np.ndarray:
    return np.array([a[0], -a[1], -a[2], -a[3]])


def quat_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    d = abs(float(np.dot(a, b)))
    return float(np.degrees(2.0 * np.arccos(min(1.0, d))))


def quat2mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rigid_with_hand_bodies(model, hand_bid: int) -> set[int]:
    rigid = set()
    for b in range(model.nbody):
        p = b
        while p != 0:
            if p == hand_bid:
                rigid.add(b)
                break
            p = int(model.body_parentid[p])
    rigid.add(hand_bid)
    cur = hand_bid
    while True:
        par = int(model.body_parentid[cur])
        if par == 0 or int(model.body_jntnum[cur]) != 0:
            break
        rigid.add(par)
        cur = par
    return rigid


def geom_rows(model, data, cube_gid: int, body_filter: set[int] | None) -> list[tuple]:
    rows = []
    fromto = np.zeros(6)
    for g in range(model.ngeom):
        if g == cube_gid:
            continue
        if body_filter is not None and int(model.geom_bodyid[g]) not in body_filter:
            continue
        dist = float(mujoco.mj_geomDistance(model, data, cube_gid, g, 0.5, fromto))
        rows.append((dist, g))
    return rows


def print_rows(model, data, rows: list[tuple], gname, bname, limit: int = 22) -> None:
    def gname_of(i):
        return gname(i)

    rows = sorted(rows, key=lambda r: r[0])
    print(f"{'dist mm':>9}  {'ct':>2} {'grp':>3}  {'geom':40}  {'body':26} world pos mm")
    for dist, g in rows[:limit]:
        ct = int(model.geom_contype[g])
        grp = int(model.geom_group[g])
        wp = data.geom_xpos[g] * 1000
        flag = ""
        if dist < -0.002:
            flag = "  <-- DEEP PENETRATION"
        elif dist < 0.002:
            flag = "  <-- <2mm"
        print(f"{dist * 1000:9.2f}  {ct:2d} {grp:3d}  {gname_of(g):40}  {bname(int(model.geom_bodyid[g])):26} "
              f"[{wp[0]:6.1f},{wp[1]:6.1f},{wp[2]:6.1f}]{flag}")


def main() -> None:
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)

    hand_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
    cube_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY)
    cube_gid = next(g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == cube_bid)

    def gname(i: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom{i}"

    def bname(i: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}"

    rigid = rigid_with_hand_bodies(model, hand_bid)
    print(f"rigid-with-hand bodies: {sorted(bname(b) for b in rigid)}")
    print(f"model: {model.ngeom} geoms, mujoco {mujoco.__version__}")

    for report_path in sys.argv[1:]:
        rep = json.load(open(report_path, encoding="utf-8"))
        result = rep.get("result", {})
        plan = rep["grasp_plan"]
        weld = plan["grasp_weld"]
        rec_p = np.array(weld["relative_position_hand_frame"], dtype=float)
        rec_q = np.array(weld["relative_quaternion_hand_frame"], dtype=float)
        sel_q = float(plan.get("selected_close_q", 0.03))
        descent = plan.get("descent_min_height_m")

        print("=" * 100)
        print(f"report: {report_path}")
        print(f"descent_min_height={descent}  selected_close_q={sel_q:.4f}")
        print(f"recorded weld rel pos [mm] = {np.round(rec_p * 1000, 1).tolist()}  quat(wxyz)={np.round(rec_q, 4).tolist()}")

        if "final_qpos" in result:
            # ---- mode B: true final state ----
            mujoco.mj_resetData(model, data)
            data.qpos[:] = np.array(result["final_qpos"], dtype=float)
            mujoco.mj_forward(model, data)
            cp = data.xpos[cube_bid].copy()
            cq = data.xquat[cube_bid].copy()
            print(f"TRUE final cube world pos [mm] = {np.round(cp * 1000, 1).tolist()}")
            print(f"TRUE final cube quat (wxyz)    = {np.round(cq, 4).tolist()}")
            print("-- recorded weld vs TRUE relative poses at final state --")
            for cand in CANDIDATE_FRAMES:
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cand)
                bp = data.xpos[bid].copy()
                bq = data.xquat[bid].copy()
                rel_p = quat2mat(bq).T @ (cp - bp)
                rel_q = quat_mul(quat_conj(bq), cq)
                dpos = float(np.linalg.norm(rel_p - rec_p)) * 1000
                dq = quat_angle_deg(rel_q, rec_q)
                print(f"  cube-in-{cand:26s} pos={np.round(rel_p * 1000, 1).tolist()} "
                      f"quat={np.round(rel_q, 4).tolist()} | dpos_vs_recorded={dpos:6.1f}mm  dquat={dq:6.1f}deg")
            rows = geom_rows(model, data, cube_gid, None)
            print("-- TRUE final state: cube vs ALL geoms (closest first) --")
            print_rows(model, data, rows, gname, bname)

        # ---- mode A: weld-composed at home (regression check of the recorded rel) ----
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        hp = data.xpos[hand_bid].copy()
        hq = data.xquat[hand_bid].copy()
        Rh = quat2mat(hq)
        cube_pos = hp + Rh @ rec_p
        cube_quat = quat_mul(hq, rec_q)
        jadr = int(model.jnt_qposadr[int(model.body_jntadr[cube_bid])])
        data.qpos[jadr : jadr + 3] = cube_pos
        data.qpos[jadr + 3 : jadr + 7] = cube_quat
        for name in ("openarm_left_finger_joint1", "openarm_left_finger_joint2"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[int(model.jnt_qposadr[jid])] = sel_q
        mujoco.mj_forward(model, data)
        Rc = quat2mat(rec_q)
        corners = rec_p[:, None] + Rc @ np.array(list(itertools.product([-CUBE_HALF, CUBE_HALF], repeat=3))).T
        for ax, nm in enumerate("xyz"):
            print(f"  composed cube {nm} extent in hand frame: [{corners[ax].min() * 1000:7.1f}, {corners[ax].max() * 1000:7.1f}] mm")
        rows = geom_rows(model, data, cube_gid, rigid)
        print("-- WELD-ENFORCED target state: cube vs hand-assembly geoms --")
        print_rows(model, data, rows, gname, bname)
        finger_rows = [r for r in rows if "finger" in gname(r[1]).lower()]
        print(f"self-check finger contact distances: {[round(r[0] * 1000, 3) for r in finger_rows]} mm")
        print()


if __name__ == "__main__":
    main()
