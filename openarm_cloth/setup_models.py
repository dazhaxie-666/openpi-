"""Install official MediaPipe task assets into the shared A/B models directory."""
from pathlib import Path
import urllib.request

MODEL_ROOT = Path(__file__).resolve().parent.parent / 'pose2arm-main' / 'models'
BASE = 'https://storage.googleapis.com/mediapipe-models/'
ASSETS = {
    'hand_landmarker.task': 'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    'pose_landmarker_lite.task': 'pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
    'pose_landmarker_full.task': 'pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
    'pose_landmarker_heavy.task': 'pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
}


def main():
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    for name, resource in ASSETS.items():
        target = MODEL_ROOT / name
        if target.is_file() and target.stat().st_size:
            print(f'Exists: {target}')
            continue
        temporary = target.with_suffix('.task.download')
        try:
            print(f'Download: {BASE + resource}', flush=True)
            with urllib.request.urlopen(BASE + resource, timeout=120) as response, temporary.open('wb') as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'Installed: {target} ({target.stat().st_size} bytes)', flush=True)


if __name__ == '__main__':
    main()
