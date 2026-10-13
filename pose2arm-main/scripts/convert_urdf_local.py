"""Convert URDF to USD using the locally cached URDF importer extension.

Isaac Lab's built-in convert_urdf.py pins to version 2.4.31 for Isaac Sim 5.1,
but the installed cache has 2.3.10. This script explicitly enables the available
version instead.

Usage:
    source ~/isaac_venv/bin/activate
    python3 scripts/convert_urdf_local.py /tmp/openarm_bimanual.urdf rl/assets/openarm_bimanual.usd
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert URDF to USD (local extension version).")
parser.add_argument("input",  type=str, help="Input URDF file path.")
parser.add_argument("output", type=str, help="Output USD file path.")
parser.add_argument("--fix-base", action="store_true", default=True)
parser.add_argument("--merge-joints", action="store_true", default=False)
parser.add_argument("--joint-stiffness", type=float, default=400.0)
parser.add_argument("--joint-damping",   type=float, default=80.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher    = AppLauncher(args_cli)
simulation_app  = app_launcher.app

# ── After Isaac Sim is up ──────────────────────────────────────────────────────
import os

def main():
    urdf_path = str(Path(args_cli.input).resolve())
    usd_path  = str(Path(args_cli.output).resolve())
    usd_dir   = str(Path(usd_path).parent)
    usd_name  = Path(usd_path).name

    os.makedirs(usd_dir, exist_ok=True)

    import omni.kit.app
    import omni.kit.commands

    # Enable the locally cached URDF importer (version 2.3.10 rather than 2.4.31)
    mgr = omni.kit.app.get_app().get_extension_manager()
    ext_id = "isaacsim.asset.importer.urdf"
    if not mgr.is_extension_enabled(ext_id):
        print(f"[INFO] Enabling extension: {ext_id}")
        mgr.set_extension_enabled_immediate(ext_id, True)

    from isaacsim.asset.importer.urdf import _urdf

    # Build import config using the v2.3.10 API
    _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints      = args_cli.merge_joints
    import_config.fix_base                = args_cli.fix_base
    import_config.make_default_prim       = True
    import_config.import_inertia_tensor   = True
    import_config.distance_scale          = 1.0
    import_config.default_drive_type      = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.default_drive_strength          = args_cli.joint_stiffness
    import_config.default_position_drive_damping  = args_cli.joint_damping

    print(f"[INFO] Converting {urdf_path}")
    print(f"[INFO]         → {usd_path}")

    omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=import_config,
        dest_path=usd_path,
    )
    print("[INFO] Conversion complete.")


if __name__ == "__main__":
    main()
    simulation_app.close()
