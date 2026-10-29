"""Shared Isaac Lab ArticulationCfg for the OpenArm bimanual robot.

Imported by both the RL training environment (rl/env.py) and the live
Isaac Sim demo (demo_isaac_gui.py) so joint limits, stiffness, and damping
stay in one place.
"""
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_USD = str(_REPO_ROOT / "rl" / "assets" / "openarm_bimanual.usd")

# Actuator limits taken from the OpenArm v2.0 datasheet.
# velocity_limit_sim units: rad/s  |  effort_limit_sim units: N·m
OPENARM_BI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=LOCAL_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=10.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # Stage 1: off
            solver_position_iteration_count=10,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "openarm_left_joint.*":  0.0,
            "openarm_right_joint.*": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "openarm_left_joint[1-7]",
                "openarm_right_joint[1-7]",
            ],
            velocity_limit_sim={
                "openarm_left_joint[1-2]":  16.755,
                "openarm_right_joint[1-2]": 16.755,
                "openarm_left_joint[3-4]":  5.4454,
                "openarm_right_joint[3-4]": 5.4454,
                "openarm_left_joint[5-7]":  20.944,
                "openarm_right_joint[5-7]": 20.944,
            },
            effort_limit_sim={
                "openarm_left_joint[1-2]":  40.0,
                "openarm_right_joint[1-2]": 40.0,
                "openarm_left_joint[3-4]":  27.0,
                "openarm_right_joint[3-4]": 27.0,
                "openarm_left_joint[5-7]":  7.0,
                "openarm_right_joint[5-7]": 7.0,
            },
            stiffness=400.0,
            damping=80.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
