"""Readable status panel beside the unoccluded human camera preview."""
import textwrap
import cv2
import numpy as np


def compose_task_preview(camera,lines,stage,paused=False,start_requested=False,direct=True):
    canvas=np.full((720,1440,3),22,np.uint8)
    canvas[:540,:960]=cv2.resize(camera,(960,540))
    cv2.line(canvas,(960,0),(960,720),(80,80,80),2)
    if paused:
        title='PAUSED - SPACE TO RESUME'
    elif stage=='ready':
        title=('START REQUESTED - SHOW WRIST' if direct else 'START REQUESTED - SHOW OPEN HAND') if start_requested else 'READY - SPACE / START TO BEGIN'
    else:title=stage.upper()+' - C TO RECENTER'
    cv2.putText(canvas,title,(16,590),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,230,255),2,cv2.LINE_AA)
    for i,line in enumerate((('Wrist drives arm; missing fingers hold gripper.' if direct else
                             'Assisted mode requires measured hand gestures.'),
                             'SPACE start/pause | C recenter | R restart vision | Q save/exit')):
        cv2.putText(canvas,line,(16,632+i*30),cv2.FONT_HERSHEY_SIMPLEX,.65,(235,235,235),1,cv2.LINE_AA)
    wrapped=[]
    for line in lines:wrapped.extend(textwrap.wrap(str(line),width=43) or [''])
    # The usual task diagnostics fit at this size. Fit extra wrapped lines
    # inside the panel rather than allowing the calibration status to vanish.
    spacing=min(28.,680/max(1,len(wrapped)))
    scale=min(.62,spacing/43.)
    for i,line in enumerate(wrapped):
        cv2.putText(canvas,line,(977,30+round(i*spacing)),cv2.FONT_HERSHEY_SIMPLEX,
                    scale,(245,245,245),1,cv2.LINE_AA)
    return canvas
