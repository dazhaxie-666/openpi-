"""Relative palm-projection input for a frontal, fixed RGB camera.

Inspired by OAT Sec III-B (2D/3D matching), with local reference rebasing.
This is not calibrated range: log depth ratios drive a configurable relative
displacement. A fixed palm template removes per-frame model-size changes.
https://eprints.whiterose.ac.uk/id/eprint/238695/1/ICRA26_0903_FI.pdf
"""
import numpy as np
import cv2
from live_reference_conditioning import AdaptiveReferenceFilter

PALM=np.array([0,5,9,13,17])


def palm_geometry(hand):
    h=np.asarray(hand,dtype=float)
    if h.shape!=(21,3) or not np.isfinite(h).all():return None
    x=h[5]-h[17];width=np.linalg.norm(x)
    if not .015<=width<=.18:return None
    x=x/width;y=h[9]-h[0];y=y-x*np.dot(x,y);length=np.linalg.norm(y)
    if length<.015:return None
    y/=length;rotation=np.column_stack((x,y,np.cross(x,y)))
    canonical=(h[PALM]-h[0])@rotation
    shape=np.r_[np.linalg.norm(h[4]-h[8])/width,
                np.linalg.norm(h[[8,12,16,20]]-h[0],axis=1)/width]
    return canonical,rotation,shape


def palm_translation(points,pixels,width,height):
    """Robust camera-aligned 2D/3D translation; focal length is nominal only."""
    fx=fy=float(width);p=np.asarray(points);uv=np.asarray(pixels)
    u=uv[:,0]-width/2;v=uv[:,1]-height/2
    a=np.zeros((len(p)*2,3));a[::2,0]=fx;a[1::2,1]=fy;a[::2,2]=-u;a[1::2,2]=-v
    b=np.empty(len(p)*2);b[::2]=u*p[:,2]-fx*p[:,0];b[1::2]=v*p[:,2]-fy*p[:,1]
    t,_,rank,_=np.linalg.lstsq(a,b,rcond=None)
    if rank<3 or not .08<t[2]<4:return None
    for _ in range(5):
        xyz=p+t;z=xyz[:,2]
        if z.min()<=.03:return None
        residual=(xyz[:,:2]/z[:,None]*[fx,fy]+[width/2,height/2]-uv).reshape(-1)
        jac=np.zeros((len(p)*2,3));jac[::2,0]=fx/z;jac[1::2,1]=fy/z
        jac[::2,2]=-fx*xyz[:,0]/z**2;jac[1::2,2]=-fy*xyz[:,1]/z**2
        weights=np.minimum(1.,4./np.maximum(abs(residual),1e-9));jw=jac*weights[:,None]
        delta=np.linalg.solve(jac.T@jw+1e-6*np.eye(3),-jw.T@residual)
        t+=np.clip(delta,-.05,.05)
        if np.linalg.norm(delta)<1e-7:break
    xyz=p+t
    if not .08<t[2]<4 or xyz[:,2].min()<=.03:return None
    predicted=xyz[:,:2]/xyz[:,2,None]*[fx,fy]+[width/2,height/2]
    return float(t[2]),float(np.sqrt(np.mean((predicted-uv)**2)))


def palm_image_similarity(reference,current):
    """Positive uniform scale after translation/in-plane rotation, not range."""
    a=np.asarray(reference,float);b=np.asarray(current,float)
    a=a-a.mean(0);b=b-b.mean(0)
    singular=np.linalg.svd(a,compute_uv=False)
    if singular[0]<1e-9 or singular[-1]/singular[0]<.15:return None
    u,_,v=np.linalg.svd(a.T@b)
    rotation=u@np.diag([1.,np.linalg.det(u@v)])@v
    scale=float(np.sum((a@rotation)*b)/np.sum(a*a))
    error=float(np.linalg.norm(scale*a@rotation-b)/max(np.linalg.norm(b),1e-9))
    return (scale,error) if .6<=scale<=1.7 and error<=.06 else None


def planar_palm_translation(template,pixels,width,height,prior_z=None):
    """Fit BOTH rotation and translation from current pixels, not model rotation.

    IPPE assumes a locally planar palm; this remains relative, nominal-camera
    input. Reject edge-on fits, negative depth, and implausible image geometry.
    """
    points=np.ascontiguousarray(template,dtype=np.float64).copy();points[:,2]=0.
    uv=np.ascontiguousarray(pixels,dtype=np.float64)
    if np.linalg.svd(points[:,:2]-points[:,:2].mean(0),compute_uv=False)[-1]<.005:return None
    camera=np.array([[float(width),0,width/2],[0,float(width),height/2],[0,0,1.]])
    try:
        result=cv2.solvePnPGeneric(points,uv,camera,None,flags=cv2.SOLVEPNP_IPPE)
        candidates=[]
        if result[0]:
            for rotation,translation in zip(result[1],result[2]):
                rotation,translation=cv2.solvePnPRefineLM(points,uv,camera,None,rotation,translation)
                matrix=cv2.Rodrigues(rotation)[0];t=translation.ravel()
                xyz=points@matrix.T+t
                ray_cos=abs(float(np.dot(matrix[:,2],t/max(np.linalg.norm(t),1e-9))))
                if not np.isfinite(xyz).all() or xyz[:,2].min()<=.08 or not .08<t[2]<4 or ray_cos<=.2:continue
                projected=cv2.projectPoints(points,rotation,translation,camera,None)[0].reshape(-1,2)
                error=float(np.sqrt(np.mean((projected-uv)**2)))
                candidates.append((error,float(t[2])))
    except cv2.error:return None
    if not candidates:return None
    best=min(candidates)
    # Plane pose has two solutions. Prefer continuity only among nearly equal
    # image fits; never choose a much worse fit just to keep a prior depth.
    if prior_z is not None and np.isfinite(prior_z) and prior_z>0:
        close=[c for c in candidates if c[0]<=best[0]+.5]
        best=min(close,key=lambda c:abs(np.log(c[1]/prior_z)))
    return best[1],best[0]


class RelativePalmDepth:
    def __init__(self,sensitivity=.15,solver='world'):
        if not np.isfinite(sensitivity) or sensitivity<=0:raise ValueError('palm depth sensitivity must be positive and finite')
        self.sensitivity=float(sensitivity);self.position=0.;self.valid=False;self.reason='DEPTH_WAIT_HAND'
        if solver not in ('world','planar'):raise ValueError('unknown palm solver')
        self.solver=solver;self.last_projection_z=None
        self.stamp=None;self.last_good_stamp=None;self.template=None;self.samples=[];self.reference=None;self.pending=None
        self.identity_stamp=None;self.last_hand_stamp=None;self.hand_gap=False
        self.image_reference=None
        self.rebase=True
        self.uncertain_samples=[]
        self.camera_z=np.nan;self.reprojection_error=np.nan
        self.filter=AdaptiveReferenceFilter(min_cutoff=3.,beta=3.)

    def snapshot(self):
        return dict(position=self.position,valid=self.valid,reason=self.reason,
                    camera_z_nominal=self.camera_z,reprojection_error_px=self.reprojection_error)

    def hold(self,reason,rebase=True,preserve_reference=False,capture_s=None):
        if capture_s is not None and np.isfinite(capture_s) and (self.stamp is None or capture_s>self.stamp):self.stamp=float(capture_s)
        self.valid=False;self.reason=reason;self.pending=None
        if rebase or self.last_good_stamp is None or not preserve_reference and self.stamp-self.last_good_stamp>.25:self.rebase=True
        self.camera_z=self.reprojection_error=np.nan
        if self.rebase:self.filter.stamp=None
        return self.snapshot()

    def update(self,hand,normalized,capture_s,image_size=(1280,720),gripper=None,identity_continuous=False):
        # Gripper aperture is intentionally NOT a depth-validity gate. It can
        # change while an independently observable palm translates normally.
        if not np.isfinite(capture_s):return self.hold('DEPTH_BAD_CLOCK')
        if self.stamp is not None and capture_s<=self.stamp:return self.snapshot()
        if self.stamp is not None and capture_s-self.stamp>.35:self.rebase=True
        self.stamp=float(capture_s)
        if identity_continuous:self.identity_stamp=float(capture_s)
        geometry=None if hand is None else palm_geometry(hand)
        n=None if normalized is None else np.asarray(normalized,dtype=float)
        if hand is None or geometry is None and (self.solver=='world' or self.template is None) or n is None or n.shape!=(21,3) or not np.isfinite(n).all():
            self.hand_gap=True
            bridge=identity_continuous and self.last_hand_stamp is not None and capture_s-self.last_hand_stamp<=.8
            return self.hold('DEPTH_HAND_MISSING',rebase=False,preserve_reference=bridge)
        wh=np.asarray(image_size,dtype=float);uv=n[PALM,:2]*wh
        if np.any(n[PALM,:2]<0) or np.any(n[PALM,:2]>1):return self.hold('DEPTH_PALM_CLIPPED',rebase=False)
        span=float(np.max(np.ptp(uv,axis=0)))
        if span<18.*wh[1]/720.:return self.hold('DEPTH_PALM_TOO_SMALL',rebase=False)
        if self.hand_gap and not identity_continuous:self.rebase=True
        self.hand_gap=False;self.last_hand_stamp=float(capture_s)
        canonical,rotation,_=geometry if geometry is not None else (None,None,None)
        if self.template is None:
            self.samples.append(canonical)
            if len(self.samples)<5:return self.hold('DEPTH_CALIBRATING_PALM')
            self.template=np.median(self.samples,axis=0)
        if self.solver=='planar':
            fitted=planar_palm_translation(self.template,uv,*wh,prior_z=self.last_projection_z)
            if fitted is None:return self.hold('DEPTH_PLANAR_UNOBSERVABLE',rebase=False,preserve_reference=True)
            z,error=fitted
            if error>max(3.,min(10.,.12*span)):
                return self.hold('DEPTH_REPROJECTION_UNCERTAIN',rebase=False,preserve_reference=True)
            return self.accept_projection(z,error,uv,capture_s,'DEPTH_PLANAR_TRACKING')
        # Compare shape after removing model scale. Model-size jitter is never
        # interpreted as physical travel or silently copied into the template.
        scale=np.linalg.norm(canonical)/max(np.linalg.norm(self.template),1e-9)
        distortion=np.linalg.norm(canonical/scale-self.template)/max(np.linalg.norm(self.template),1e-9)
        def uncertain(reason):
            # Geometry can adapt to a stable new palm shape, but the physical
            # size and reference must survive while hand identity is observed.
            self.uncertain_samples.append(canonical)
            self.uncertain_samples=self.uncertain_samples[-5:]
            if len(self.uncertain_samples)==5:
                normalized_shapes=[v/max(np.linalg.norm(v),1e-9) for v in self.uncertain_samples]
                if np.max(np.linalg.norm(np.asarray(normalized_shapes)-np.mean(normalized_shapes,axis=0),axis=(1,2)))<.08:
                    candidate=np.median(self.uncertain_samples,axis=0)
                    self.template=candidate*(np.linalg.norm(self.template)/max(np.linalg.norm(candidate),1e-9))
                    self.uncertain_samples=[]
            return self.hold(reason,rebase=False,preserve_reference=True)
        if distortion>.22:return uncertain('DEPTH_PALM_SHAPE_UNCERTAIN')
        result=palm_translation(self.template@rotation.T,uv,*wh)
        if result is None:return self.hold('DEPTH_PROJECTION_FAILED',rebase=False,preserve_reference=True)
        z,error=result;reason='DEPTH_TRACKING'
        if error>max(3.,min(10.,.12*span)):
            previous_pending=self.pending
            held=uncertain('DEPTH_REPROJECTION_UNCERTAIN')
            anchor=self.image_reference
            # Only an observed, near-isotropic image change can bridge a brief
            # 3D-model mismatch. Never chain fallback references or fill a lost
            # hand. A reliable projection must renew this 0.75-second lease.
            if self.rebase or anchor is None or not 0<=capture_s-anchor[2]<=.75:return held
            similarity=palm_image_similarity(anchor[0],uv)
            if similarity is None:return held
            z=anchor[1]/similarity[0];reason='DEPTH_SCALE_TRACKING'
            self.pending=previous_pending
        else:self.uncertain_samples=[]
        return self.accept_projection(z,error,uv,capture_s,reason)

    def accept_projection(self,z,error,uv,capture_s,reason):
        self.camera_z=z;self.reprojection_error=error
        if self.rebase or self.reference is None:
            self.reference=(z,self.position);self.rebase=False
            self.filter.stamp=None;self.filter.update(self.position,capture_s)
            self.last_good_stamp=float(capture_s)
            self.image_reference=(uv.copy(),z,float(capture_s))
            self.last_projection_z=z
            self.valid=True;self.reason='DEPTH_TRACKING';return self.snapshot()
        log_change=float(np.log(z/self.reference[0]))
        wanted=self.reference[1]-self.sensitivity*log_change
        if abs(wanted-self.position)>self.sensitivity*.30:
            if self.pending is None or abs(z-self.pending)/z>.08:
                self.pending=z;self.valid=False;self.reason='DEPTH_JUMP_CONFIRMING';return self.snapshot()
        self.pending=None
        self.position=float(self.filter.update(wanted,capture_s))
        self.last_good_stamp=float(capture_s)
        self.last_projection_z=z
        if reason=='DEPTH_TRACKING':self.image_reference=(uv.copy(),z,float(capture_s))
        self.valid=True;self.reason=reason
        return self.snapshot()
