"""Install official MediaPipe task assets into the shared A/B models directory."""
from pathlib import Path
import argparse
import io
import urllib.request
import zipfile

MODEL_ROOT = Path(__file__).resolve().parent.parent / 'pose2arm-main' / 'models'
BASE = 'https://storage.googleapis.com/mediapipe-models/'
ASSETS = {
    'hand_landmarker.task': 'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    'pose_landmarker_lite.task': 'pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
    'pose_landmarker_full.task': 'pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
    'pose_landmarker_heavy.task': 'pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rtmpose',action='store_true',help='Also install the official 2D arm model used by single-arm')
    args=parser.parse_args()
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
    if args.rtmpose:
        target=MODEL_ROOT/'rtmpose_body_s/end2end.onnx'
        if target.is_file():
            print(f'Exists: {target}')
        else:
            url='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip'
            print('Download: '+url,flush=True)
            with urllib.request.urlopen(url,timeout=120) as response:payload=response.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                names=[n for n in archive.namelist() if n.endswith('/end2end.onnx')]
                if len(names)!=1:raise ValueError('Official model archive has ambiguous ONNX asset')
                content=archive.read(names[0])
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_suffix('.download')
            temporary.write_bytes(content);temporary.replace(target)
        print(f'Installed: {target} ({target.stat().st_size} bytes)', flush=True)


if __name__ == '__main__':
    main()
