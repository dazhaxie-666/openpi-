"""Independent palm-scaled pinch signal, personal calibration, no auto-release."""
import hashlib
import json
from pathlib import Path
import numpy as np
from live_reference_conditioning import AdaptiveReferenceFilter

DEFAULT_CALIBRATION=Path(__file__).resolve().parent/'calibration'/'personal_motion'/'pinch.json'


def pinch_features(hand,norm=None,image_size=(1280,720)):
    h=None if hand is None else np.asarray(hand,dtype=float)
    if h is None or h.shape!=(21,3) or not np.isfinite(h).all():return None
    scales=[np.linalg.norm(h[5]-h[17]),1.3*np.linalg.norm(h[9]-h[0])]
    scales=[v for v in scales if .015<v<.2]
    if not scales:return None
    scale=float(np.median(scales));distance=np.linalg.norm(h[4]-h[8])
    if distance>.25:return None
    r2=None
    n=None if norm is None else np.asarray(norm,dtype=float)
    if n is not None and n.shape==(21,3) and np.isfinite(n).all():
        p=n[:,:2]*np.asarray(image_size)
        width=np.linalg.norm(p[5]-p[17]);length=np.linalg.norm(p[9]-p[0])
        # Side-on palms have an unobservable projected width, not a tiny hand.
        if width>5 and length>5 and width/max(length,1e-6)>.18:
            r2=float(np.linalg.norm(p[4]-p[8])/np.median([width,1.3*length]))
    return dict(r3=float(distance/scale),r2=r2,palm_scale_3d=scale)


class PinchCalibration:
    def __init__(self,payload=None):
        self.trials={'closed':[],'open':[]};self.profile=None;self.reason='UNCALIBRATED';self.calibration_id='uncalibrated'
        if payload:
            self.trials=payload.get('trials',self.trials)
            self.fit()

    def add_trial(self,label,samples):
        if label not in self.trials:raise ValueError('unknown calibration label')
        valid=[s for s in samples if s is not None and np.isfinite(s['r3'])
               and (s.get('r2') is None or np.isfinite(s['r2']))]
        if len(valid)<5:self.reason='TOO_FEW_VALID_CALIBRATION_SAMPLES';return False
        values=np.array([s['r3'] for s in valid])
        if np.ptp(np.percentile(values,[5,95]))>.3:
            self.reason='CALIBRATION_HAND_NOT_STABLE';return False
        self.trials[label].append([{k:(None if v is None else float(v)) for k,v in s.items()} for s in valid])
        self.fit();return True

    def fit(self):
        self.profile=None;self.calibration_id='uncalibrated'
        if min(len(self.trials[k]) for k in self.trials)<10:
            self.reason='NEED_10_CLOSED_AND_10_OPEN_TRIALS';return False
        profile={}
        for field in ('r3','r2'):
            groups={label:[v[field] for trial in trials for v in trial if v.get(field) is not None] for label,trials in self.trials.items()}
            if min(map(len,groups.values()))<30:
                if field=='r3':self.reason='INSUFFICIENT_3D_CALIBRATION';return False
                continue
            a=float(np.percentile(groups['closed'],95));b=float(np.percentile(groups['open'],5))
            if field=='r3' and np.percentile(groups['closed'],5)>np.percentile(groups['open'],95):
                self.reason='OPEN_CLOSED_LABELS_REVERSED';return False
            noise=max(float(np.std([np.median([v[field] for v in trial if v.get(field) is not None]) for trial in self.trials[label] if any(v.get(field) is not None for v in trial)])) for label in self.trials)
            if b-a<=max(.05,6*noise):
                if field=='r3':self.reason='OPEN_CLOSED_DISTRIBUTIONS_OVERLAP';return False
                continue
            profile[field]=dict(closed=a,open=b,close_threshold=a+.25*(b-a),open_threshold=a+.75*(b-a))
        self.profile=profile;self.reason=''
        self.calibration_id=hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest()[:16]
        return True

    def export(self):
        return dict(version=1,trials=self.trials,profile=self.profile,calibration_id=self.calibration_id,reason=self.reason)

    def summary(self):
        return dict(closed_trials=len(self.trials['closed']),open_trials=len(self.trials['open']),
                    calibrated=self.profile is not None,reason=self.reason,calibration_id=self.calibration_id)


class HumanPinchControl:
    def __init__(self,mode='continuous',calibration=None):
        if mode not in ('continuous','binary','hold'):raise ValueError('unknown gripper mode')
        self.mode=mode;self.calibration=calibration or PinchCalibration();self.aperture=.044
        self.filter=AdaptiveReferenceFilter(3.,1.5);self.last_stamp=None;self.raw=None;self.candidate=None;self.candidate_since=None;self.candidate_count=0
        self.closed=False;self.state='LOST';self.reason='PINCH_MISSING';self.last_accepted=None;self.features=None

    def update(self,features,stamp):
        if self.last_stamp is not None and stamp<=self.last_stamp:
            self.reason='NON_MONOTONIC_SOURCE';return self.aperture
        dt=.05 if self.last_stamp is None else stamp-self.last_stamp;self.last_stamp=stamp;self.features=features
        if (features is not None and (not np.isfinite(features.get('r3',np.nan)) or features['r3']<0
            or (features.get('r2') is not None and not np.isfinite(features['r2'])))):
            features=None
        if features is None or dt>.25:
            self.state='HOLD' if self.last_accepted is not None else 'LOST';self.reason='PINCH_MISSING' if features is None else 'SOURCE_GAP'
            self.candidate=self.candidate_since=None;self.candidate_count=0;self.filter.stamp=None;self.raw=None
            return self.aperture
        profile=self.calibration.profile or dict(r3=dict(closed=.15,open=1.2))
        openness={}
        for key,endpoints in profile.items():
            ratio=features.get(key)
            if ratio is not None:openness[key]=float(np.clip((ratio-endpoints['closed'])/(endpoints['open']-endpoints['closed']),0.,1.))
        if 'r3' not in openness:return self.aperture
        if 'r2' in openness and abs(openness['r2']-openness['r3'])>.4:
            self.state='HOLD';self.reason='PINCH_2D_3D_CONFLICT';self.candidate=None;self.candidate_count=0
            return self.aperture
        wanted=openness['r3']
        if self.raw is not None and abs(features['r3']-self.raw)>.35+12*min(dt,.1):
            self.raw=features['r3'];self.state='REACQUIRING';self.reason='PINCH_TEMPORAL_OUTLIER';self.candidate=None;self.candidate_count=0
            return self.aperture
        self.raw=features['r3']
        candidate='closed' if wanted<.25 else 'open' if wanted>.75 else 'middle'
        if candidate!=self.candidate:
            self.candidate=candidate;self.candidate_since=stamp;self.candidate_count=1
        else:self.candidate_count+=1
        if self.candidate_count<2 or stamp-self.candidate_since<.03:
            self.state='REACQUIRING';self.reason='PINCH_CONFIRMING';return self.aperture
        self.state='TRACKING';self.reason='' if self.calibration.profile else 'UNCALIBRATED_ENDPOINTS'
        self.last_accepted=stamp
        if candidate=='closed':self.closed=True
        elif candidate=='open':self.closed=False
        if self.mode=='binary':self.aperture=0. if self.closed else .044
        elif self.mode=='continuous':self.aperture=float(self.filter.update(wanted,stamp))*.044
        return self.aperture


class TaskHandControl:
    """Body-independent right-hand channel for Cartesian direct tasks.

    Same pinch/calibration implementation as full-arm following; input clocks
    and accepted hand identity stay explicit. No detector score substitutes
    for a measured thumb/index ratio.
    """
    def __init__(self,mode='continuous',calibration_path=None,initial_aperture=.044):
        from motion_observation import SourceClock
        self.profiles=load_personal_calibration(calibration_path)
        self.calibration=PinchCalibration(self.profiles.get('right'))
        self.control=HumanPinchControl(mode,self.calibration)
        self.control.aperture=float(initial_aperture);self.control.closed=initial_aperture<.022
        self.clock=SourceClock();self.capture=None;self.seq=-1;self.stamp=None

    def begin_pinch_calibration(self,label):
        if label not in ('closed','open'):raise ValueError('unknown pinch label')
        self.capture=dict(label=label,start=None,samples=[])

    def export_calibration(self):
        return dict(self.profiles,right=self.calibration.export())

    def missing(self,reason='HAND_MISSING'):
        c=self.control;c.state='HOLD' if c.last_accepted is not None else 'LOST';c.reason=reason
        c.candidate=c.candidate_since=None;c.candidate_count=0;c.features=None;c.filter.stamp=None;c.raw=None
        return self.snapshot()

    def process(self,obs,now_s,source_clock=False,image_size=(1280,720)):
        if obs is None:return self.missing()
        if not source_clock and not 0<=now_s-obs['read_end']/1e9<=.35:return self.missing('STALE_HAND_SOURCE')
        arm=obs.get('arms',{}).get('right',{})
        features=pinch_features(arm.get('hand_raw'),arm.get('hand_norm'),image_size) if arm.get('hand_valid') else None
        return self.process_features(obs['seq'],obs['read_end']/1e9,features)

    def process_features(self,seq,stamp,features):
        accepted,_,_=self.clock.accept(seq,stamp)
        if not accepted:return self.snapshot()
        self.seq,self.stamp=int(seq),float(stamp)
        self.control.update(features,stamp)
        if self.capture is not None:
            if self.capture['start'] is None:self.capture['start']=stamp+.35
            elapsed=stamp-self.capture['start']
            if 0<=elapsed<=.75 and features is not None:self.capture['samples'].append(features)
            if elapsed>.75:
                self.calibration.add_trial(self.capture['label'],self.capture['samples']);self.capture=None
        return self.snapshot()

    def snapshot(self):
        c=self.control;intent_valid=c.state=='TRACKING';valid=intent_valid and c.mode!='hold'
        endpoints=(self.calibration.profile or dict(r3=dict(closed=.15,open=1.2)))['r3']
        ratio=None if not intent_valid or c.features is None else .15+1.05*float(np.clip(
            (c.features['r3']-endpoints['closed'])/(endpoints['open']-endpoints['closed']),0.,1.))
        return dict(seq=self.seq,capture_s=self.stamp,aperture=c.aperture,valid=valid,intent_valid=intent_valid,state=c.state,reason=c.reason,
                    features=c.features,intent_ratio=ratio,calibration=self.calibration.summary(),
                    calibration_collecting=None if self.capture is None else self.capture['label'])


def load_personal_calibration(path=None):
    target=Path(path or DEFAULT_CALIBRATION)
    if not target.exists():return {}
    payload=json.loads(target.read_text(encoding='utf-8'))
    if payload.get('version')!=1:raise ValueError('unsupported personal calibration version')
    return payload.get('sides',{})


def save_personal_calibration(path,sides):
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(version=1,scope='personal development calibration; no 99% accuracy claim',sides=sides)
    target.write_text(json.dumps(payload,indent=2),encoding='utf-8')


def calibration_status(summary):
    """Trial counts alone never mean a fitted personal mapping exists."""
    if summary.get('calibrated'):return 'READY - personal endpoints active'
    reason=summary.get('reason') or 'NO_PERSONAL_PROFILE'
    if reason=='OPEN_CLOSED_LABELS_REVERSED':return 'NOT READY: LABELS REVERSED - G=PINCH, O=SPREAD'
    return 'NOT READY: '+reason
