"""Conservative grasp REQUEST only; does not alter tracking or release signals."""
import numpy as np


def grasp_request_ratio(features,profile=None):
    """Both visible 2-D separation and model 3-D distance must indicate pinch."""
    if not features:return None
    profile=profile or dict(r3=dict(closed=.15,open=1.2),r2=dict(closed=.15,open=1.2))
    values=[]
    for key in ('r3','r2'):
        measured=features.get(key);endpoints=profile.get(key)
        if measured is None or not np.isfinite(measured) or endpoints is None:return None
        span=endpoints['open']-endpoints['closed']
        if not np.isfinite(span) or span<=0:return None
        values.append(float(np.clip((measured-endpoints['closed'])/span,0.,1.)))
    return .15+1.05*max(values)
