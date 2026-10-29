"""Desktop launcher: open the interactive grasp demo without a console."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import subprocess
from datetime import datetime
import traceback


def main():
    root = Path(__file__).resolve().parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = root / "outputs" / datetime.now().strftime("%Y%m%d")
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / f"desktop_demo_{stamp}.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        try:
            # File associations may select a different Python. Delegate actual
            # simulation to the same B environment used by both CLI launchers.
            powershell = Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
            result = subprocess.run([
                str(powershell), '-NoProfile', '-ExecutionPolicy', 'Bypass',
                '-File', str(root.parent / 'run.ps1'), '-Mode', 'grasp',
            ], cwd=root, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
            if result.returncode:
                log.write(f"Simulation exited: {result.returncode}\n")
                log.flush()
                ctypes.windll.user32.MessageBoxW(
                    None, f"仿真运行失败，请查看日志：\n{log_path}", "抓取仿真", 0x10
                )
        except Exception:
            traceback.print_exc(file=log)
            log.flush()
            ctypes.windll.user32.MessageBoxW(
                None, f"仿真运行出现异常，请查看日志：\n{log_path}", "抓取仿真", 0x10
            )


if __name__ == "__main__":
    main()
