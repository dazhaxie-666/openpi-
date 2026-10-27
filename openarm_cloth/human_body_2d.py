"""Single-person RTMPose image observations and nominal relative Y/Z input.

No model-3D coordinates are fabricated. X is unavailable (zero placeholder at
this interface) and must be supplied by the independent palm-depth adapter.
Official model/preprocessing: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
"""
from pathlib import Path
import hashlib
import time
import cv2
import numpy as np

MODEL=Path(__file__).resolve().parent.parent/'pose2arm-main/models/rtmpose_body_s/end2end.onnx'
COCO_TO_MP={0:0,1:2,2:5,3:7,4:8,5:11,6:12,7:13,8:14,9:15,10:16,11:23,12:24,13:25,14:26,15:27,16:28}


def map_coco_pose(xy,score,image_size):
    result=np.full((33,4),np.nan);result[:,3]=0.
    for coco,mp in COCO_TO_MP.items():
        result[mp,:2]=np.asarray(xy[coco])/image_size
        result[mp,3]=np.clip(score[coco],0.,1.)
    return result


def image_plane_wrist(pose,side='right',image_size=(1280,720),nominal_span=.4,min_confidence=.6):
    """Upright camera-relative control units, not calibrated XYZ in metres."""
    p=None if pose is None else np.asarray(pose)
    shoulder,wrist=(11,15) if side=='left' else (12,16)
    ids=[11,12,wrist]
    if p is None or p.shape!=(33,4):return None
    if not np.isfinite(p[ids][:,[0,1,3]]).all() or np.min(p[ids,3])<min_confidence:return None
    if np.any(p[ids,:2]<0) or np.any(p[ids,:2]>1):return None
    wh=np.asarray(image_size);span=np.linalg.norm((p[11,:2]-p[12,:2])*wh)
    if not 35.*wh[1]/720.<span<.9*wh[0]:return None
    delta=(p[wrist,:2]-p[shoulder,:2])*wh/span*nominal_span
    return np.array([0.,delta[0],-delta[1]])


def same_person_image(reference,pose,image_size=(1280,720)):
    """Two current-image face estimates anchor the single expected operator."""
    if reference is None or pose is None:return False
    a=np.asarray(reference);b=np.asarray(pose)
    if a.shape!=(33,4) or b.shape!=(33,4):return False
    if not np.isfinite(a[0,[0,1,3]]).all() or not np.isfinite(b[[0,11,12]][:,[0,1,3]]).all():return False
    if a[0,3]<.5 or b[0,3]<.6:return False
    if np.any(a[0,:2]<0) or np.any(a[0,:2]>1):return False
    wh=np.asarray(image_size);span=np.linalg.norm((b[11,:2]-b[12,:2])*wh)
    return bool(np.linalg.norm((a[0,:2]-b[0,:2])*wh)<=max(35.*wh[1]/720.,.25*span))


class ImageBodyDetector:
    def __init__(self,path=MODEL):
        self.path=Path(path)
        if not self.path.is_file():raise FileNotFoundError('RTMPose asset missing; run setup_models.py --rtmpose')
        self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.net=cv2.dnn.readNetFromONNX(str(self.path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self,bgr,context=None):
        started=time.perf_counter()
        h,w=bgr.shape[:2];left,right=0.,float(w)
        p=None if context is None else np.asarray(context)
        if p is not None and p.shape==(33,4):
            ids=np.array([0,7,8,11,12,15,16]);visible=np.isfinite(p[ids][:,[0,1,3]]).all(axis=1)&(p[ids,3]>=.6)&np.all((p[ids,:2]>=0)&(p[ids,:2]<=1),axis=1)
            kept=ids[visible]
            if 0 in kept and (15 in kept or 16 in kept) and len(kept)>=3:
                low,high=np.min(p[kept,0])*w,np.max(p[kept,0])*w
                width=min(float(w),max(.8*w,1.25*(high-low)))
                left=float(np.clip((low+high-width)/2,0,w-width));right=left+width
        center=np.array([(left+right)/2,h/2]);scale=np.array([right-left,h],float)*1.25
        if scale[0]>scale[1]*.75:scale[1]=scale[0]/.75
        else:scale[0]=scale[1]*.75
        warp=np.array([[192/scale[0],0,96-center[0]*192/scale[0]],
                       [0,256/scale[1],128-center[1]*256/scale[1]]],np.float32)
        image=cv2.warpAffine(bgr,warp,(192,256))[:,:,::-1].astype(np.float32)
        image=(image-[123.675,116.28,103.53])/[58.395,57.12,57.375]
        self.net.setInput(np.ascontiguousarray(image.transpose(2,0,1)[None],dtype=np.float32))
        sx,sy=self.net.forward(self.net.getUnconnectedOutLayersNames())
        xy=np.stack((sx[0].argmax(axis=1),sy[0].argmax(axis=1)),axis=1)/2
        score=np.minimum(sx[0].max(axis=1),sy[0].max(axis=1))
        xy=xy/[192,256]*scale+center-scale/2
        return dict(norm=map_coco_pose(xy,score,(w,h)),score=score,ms=(time.perf_counter()-started)*1000)
