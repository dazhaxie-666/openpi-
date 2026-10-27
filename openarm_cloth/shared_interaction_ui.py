"""Large operator cues derived from the same shared task entry conditions."""
from functools import lru_cache
from pathlib import Path
import numpy as np


def shared_interaction_hint(task,feedback,clock,paused=False):
    def cue(code,title,english,detail='',theme='wait',progress=0.):
        return dict(code=code,title=title,english=english,detail=detail,theme=theme,progress=float(progress))
    if paused:return cue('paused','已暂停，按空格继续','PAUSED - SPACE TO RESUME')
    if task.stage in task.terminal:return cue('finished','本轮已结束，记录已保留','SESSION ENDED')
    if task.stage=='ready':return cue('start','尚未开始：按空格或点击开始','SPACE / START')
    if task.phase=='manual':
        status=task.grasp_readiness(feedback,clock)
        if not status['fresh']:return cue('target_hidden','方块暂不可见：向上移手，机械臂随手抬起','TARGET HIDDEN - HAND CONTROLS UPWARD MOTION',
            '手移动多少就跟随多少；视野恢复后，进入范围直接捏合抓取')
        distance=status['distance_m']*1000
        if not status['in_range']:return cue('approach','继续移动右手，靠近方块','MOVE WRIST CLOSER',f'距方块表面约 {distance:.0f} 毫米；接近门限 75 毫米')
        if not status['height_clear']:return cue('raise_for_grasp','请向上移手一点，留出抓取空间','MOVE HAND UP FOR GRASP CLEARANCE')
        if not task.hand_intent_valid or task.request_ratio is None:return cue('show_hand','已进入范围：请让拇指和食指清楚入镜','IN RANGE - SHOW THUMB AND INDEX')
        if task.request_close_since is not None:
            progress=float(np.clip((task.request_stamp-task.request_close_since)/.20,0.,1.))
            return cue('pinch_confirming','正在确认捏合，请保持片刻','CONFIRMING NEW PINCH',
                '二维与三维指尖距离均需确认闭合约 0.2 秒','auto',progress)
        return cue('grasp_ready','可以抓取：现在捏合拇指和食指','READY TO GRASP - PINCH NOW',
            f'距目标表面约 {distance:.0f} 毫米；捏合后自动规划、闭合和提起','ready')
    if task.phase in ('transport','holding_payload'):
        if task.phase=='holding_payload':return cue('payload_hold','夹持反馈待确认：保持不动，张手可请求放下','PAYLOAD HOLD - OPEN TO PLACE',theme='hold')
        if task.placement_blocked_origin is not None:
            return cue('place_blocked','此处无法放下：已恢复搬运，请移手换个位置','PLACE BLOCKED - MOVE WRIST TO ANOTHER PLACE',
                '夹爪继续保持；移动至少 2 厘米后，可再次张手确认放下','hold')
        if not task.hand_intent_valid:return cue('carry_hand_missing','正在保持夹持：请让拇指和食指入镜','GRIP HELD - SHOW HAND TO PLACE')
        progress=(0. if task.open_since is None or task.open_stamp is None else
            float(np.clip((task.open_stamp-task.open_since)/.35,0.,1.)))
        if task.open_since is not None:return cue('release_confirming','正在确认张手，请稍作保持','OPEN DETECTED - HOLD TO CONFIRM',
            f'放下确认 {progress:.0%}；确认后先下降到桌面，再打开夹爪','auto',progress)
        return cue('carry_ready','已抓牢：移手搬运，张手保持即可放下','CARRY BY WRIST - OPEN TO PLACE',
            '无需再次捏合；张手需连续确认约 0.35 秒','ready')
    if task.phase in ('lower','release','retract','verify'):
        titles=dict(lower='张手已收到：正在下降到桌面',release='正在打开夹爪，放下方块',
            retract='夹爪已打开：机械臂正在退出',verify='正在确认方块已放稳')
        return cue('placing',titles[task.phase],'PLACING - '+task.phase.upper(),theme='auto')
    if task.phase=='restore_view':return cue('human_view_up','视野恢复中：机械臂跟随你向上移手','HUMAN-CONTROLLED UPWARD MOTION',
        '不会自动升到固定高度；右手保持入镜，移动手腕控制抬起幅度')
    if task.phase=='reacquire':
        return cue('restoring_view','正在张爪上退，恢复目标视野','RESTORING VIEW - KEEP CLEAR',theme='auto')
    if task.phase=='recover':return cue('retry','本次未完成：张手调整后可重试','OPEN HAND, ADJUST AND RETRY',theme='hold')
    names=dict(plan='规划抓取路径',transit='移动到安全接近位',approach='自动精确对准',descend='接近方块',close='闭合并确认夹持',lift='夹持确认，正在提起')
    return cue('grasping','正在'+names.get(task.phase,'执行抓取'),'ROBOT GRASP - '+task.phase.upper(),theme='auto')


@lru_cache(maxsize=1)
def _fonts():
    from PIL import ImageFont
    regular=Path('C:/Windows/Fonts/msyh.ttc');bold=Path('C:/Windows/Fonts/msyhbd.ttc')
    if regular.exists():return ImageFont.truetype(str(bold if bold.exists() else regular),30),ImageFont.truetype(str(regular),18),True
    return ImageFont.load_default(),ImageFont.load_default(),False


@lru_cache(maxsize=48)
def _hint_sprite(title,english,detail,theme):
    from PIL import Image,ImageDraw
    palettes={'ready':((20,92,48),(64,230,132)),'wait':((85,60,19),(250,193,60)),
        'auto':((22,58,100),(93,179,255)),'hold':((92,31,30),(255,127,105))}
    background,border=palettes.get(theme,palettes['wait'])
    image=Image.new('RGB',(928,88),background);draw=ImageDraw.Draw(image)
    draw.rounded_rectangle((0,0,927,87),radius=10,outline=border,width=2)
    title_font,detail_font,chinese=_fonts()
    draw.text((16,8),title if chinese else english,font=title_font,fill=(250,250,250))
    draw.text((16,53),detail if chinese else english,font=detail_font,fill=(225,235,235))
    return np.asarray(image)[:,:,::-1].copy()


def draw_shared_interaction_hint(canvas,hint):
    # Entirely below the 960x540 camera image; raw landmarks remain visible.
    canvas[548:636,16:944]=_hint_sprite(hint['title'],hint['english'],hint.get('detail',''),hint.get('theme','wait'))
    progress=float(hint.get('progress',0.))
    if progress>0.:
        canvas[628:632,32:928]=(70,70,70)
        canvas[628:632,32:32+round(896*min(1.,progress))]=(110,230,110)
