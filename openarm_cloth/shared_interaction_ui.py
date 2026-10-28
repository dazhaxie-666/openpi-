"""Large operator cues derived from the same shared task entry conditions."""
from functools import lru_cache
from pathlib import Path
import numpy as np


def shared_interaction_hint(task,feedback,clock,paused=False):
    def cue(code,title,english,detail='',theme='wait',progress=0.):
        return dict(code=code,title=title,english=english,detail=detail,theme=theme,progress=float(progress))
    trial_limit=int(getattr(task,'trial_limit',0) or 0)
    active_trial=max(1,int(getattr(task,'trial_id',0) or 0))
    next_trial=min(trial_limit,max(1,int(getattr(task,'trial_id',0) or 0)+1)) if trial_limit else 0
    def numbered(title,index=None):
        if not trial_limit:return title
        return f'第 {active_trial if index is None else index}/{trial_limit} 段'+title
    if paused:return cue('paused','已暂停，按空格继续','PAUSED - SPACE TO RESUME')
    if task.stage in task.terminal:return cue('finished','本轮已结束，记录已保留','SESSION ENDED')
    if task.stage=='ready':return cue('start',f'准备采集 {trial_limit} 段：按空格或点击开始' if trial_limit else
        '尚未开始：按空格或点击开始','SPACE / START')
    if task.phase=='session_complete':
        return cue('session_complete',numbered('已结束，正在保存'),'ALL TRIALS COMPLETE - SAVING',
            '请保持窗口开启，保存完成后程序会自动退出','ready')
    if task.phase=='reset':
        remaining=max(0.,float(getattr(task,'reset_until',clock) or clock)-clock)
        status=(getattr(task,'last_trial_result',None) or {}).get('status','completed')
        status_cn={'completed':'完成','failed':'失败','cancelled':'取消','interrupted':'中断'}.get(status,status)
        return cue('trial_reset',numbered(f'已结束（{status_cn}）：现在是重置时间'),'INTER-TRIAL RESET',
            f'机械臂和方块已自动复位；还剩 {remaining:.1f} 秒，请调整手的位置，期间不会开始下一段','hold',
            0. if getattr(task,'inter_trial_reset_s',0)<=0 else 1.-remaining/task.inter_trial_reset_s)
    if task.phase=='manual':
        status=task.grasp_readiness(feedback,clock)
        prefix=(f'第 {next_trial}/{trial_limit} 段准备：' if trial_limit else '')
        if not status['fresh']:return cue('target_hidden',prefix+'方块暂不可见：向上移手，机械臂随手抬起','TARGET HIDDEN - HAND CONTROLS UPWARD MOTION',
            '手移动多少就跟随多少；视野恢复后，进入范围直接捏合抓取')
        distance=status['distance_m']*1000
        if not status['in_range']:return cue('approach',prefix+'继续移动右手，靠近方块','MOVE WRIST CLOSER',f'距方块表面约 {distance:.0f} 毫米；接近门限 75 毫米')
        if not status['height_clear']:return cue('raise_for_grasp',prefix+'请向上移手一点，留出抓取空间','MOVE HAND UP FOR GRASP CLEARANCE')
        if not task.hand_intent_valid or task.request_ratio is None:return cue('show_hand',prefix+'已进入范围：请让拇指和食指清楚入镜','IN RANGE - SHOW THUMB AND INDEX')
        if task.request_close_since is not None:
            progress=float(np.clip((task.request_stamp-task.request_close_since)/.20,0.,1.))
            return cue('pinch_confirming',prefix+'正在确认捏合，请保持片刻','CONFIRMING NEW PINCH',
                '二维与三维指尖距离均需确认闭合约 0.2 秒','auto',progress)
        return cue('grasp_ready',prefix+'现在捏合，捏合确认后正式开始采集','READY TO GRASP - PINCH NOW',
            f'距目标表面约 {distance:.0f} 毫米；捏合后本段开始，自动规划、闭合和提起','ready')
    if task.phase in ('transport','holding_payload'):
        if getattr(task,'collection_task_profile','pickplace')=='hold':
            required=float(getattr(task,'collection_hold_seconds',3.))
            remaining=max(0.,required-(clock-float(getattr(task,'phase_since',clock)))) if not getattr(task,'collection_hold_complete',False) else 0.
            if remaining>0.:
                return cue('payload_timed_hold',numbered(f'采集中：保持方块不动，还剩 {remaining:.1f} 秒'),
                    'HOLD PAYLOAD STILL', '机械臂会自动保持；暂时不要移动手腕或张手', 'auto',
                    0. if required<=0 else 1.-remaining/required)
            if not getattr(task,'collection_release_rearmed',True):
                return cue('wait_new_open',numbered('采集中：夹爪持续闭合，等待一次新的张手'),
                    'GRIP LATCHED - NEW OPEN REQUIRED',
                    '如果已经提前张手，请先合手或放松，再重新张手；方块会一直保持夹紧','hold')
            progress=(0. if task.open_since is None or task.open_stamp is None else
                float(np.clip((task.open_stamp-task.open_since)/.35,0.,1.)))
            if task.open_since is not None:
                return cue('source_release_confirming','保持完成：正在确认张手','HOLD COMPLETE - CONFIRMING OPEN',
                    f'张手确认 {progress:.0%}；随后自动原位下降并放下方块','auto',progress)
            return cue('hold_complete',numbered('采集中：保持完成，请张手原位放下'),
                'HOLD COMPLETE - OPEN TO RETURN', '手腕保持不动；连续张手约 0.35 秒，系统会自动下降到原位置','ready')
        if task.phase=='holding_payload':return cue('payload_hold','夹持反馈待确认：保持不动，张手可请求放下','PAYLOAD HOLD - OPEN TO PLACE',theme='hold')
        if task.placement_blocked_origin is not None:
            return cue('place_blocked','此处无法放下：已恢复搬运，请移手换个位置','PLACE BLOCKED - MOVE WRIST TO ANOTHER PLACE',
                '夹爪继续保持；移动至少 2 厘米后，可再次张手确认放下','hold')
        if not task.hand_intent_valid:return cue('carry_hand_missing','正在保持夹持：请让拇指和食指入镜','GRIP HELD - SHOW HAND TO PLACE')
        progress=(0. if task.open_since is None or task.open_stamp is None else
            float(np.clip((task.open_stamp-task.open_since)/.35,0.,1.)))
        if task.open_since is not None:return cue('release_confirming','正在确认张手，请稍作保持','OPEN DETECTED - HOLD TO CONFIRM',
            f'放下确认 {progress:.0%}；确认后先下降到桌面，再打开夹爪','auto',progress)
        return cue('carry_ready',numbered('采集中：已抓牢，请移到绿色目标区上方再张手'),'CARRY TO GREEN - OPEN TO PLACE',
            '先确认方块位于绿色圆区上方；无需再次捏合，张手连续约 0.35 秒后放下','ready')
    if task.phase in ('lower','release','retract','verify'):
        source_return=getattr(task,'collection_task_profile','pickplace')=='hold'
        titles=dict(lower='张手已收到：正在原位下降到桌面' if source_return else '张手已收到：正在下降到桌面',release='正在打开夹爪，放下方块',
            retract='夹爪已打开：机械臂正在退出',verify='正在确认方块已放稳')
        return cue('placing',numbered('采集中：'+titles[task.phase]),'PLACING - '+task.phase.upper(),theme='auto')
    if task.phase=='restore_view':return cue('human_view_up','视野恢复中：机械臂跟随你向上移手','HUMAN-CONTROLLED UPWARD MOTION',
        '不会自动升到固定高度；右手保持入镜，移动手腕控制抬起幅度')
    if task.phase=='reacquire':
        return cue('restoring_view','正在张爪上退，恢复目标视野','RESTORING VIEW - KEEP CLEAR',theme='auto')
    if task.phase=='recover':return cue('retry','本次未完成：张手调整后可重试','OPEN HAND, ADJUST AND RETRY',theme='hold')
    names=dict(plan='规划抓取路径',transit='移动到安全接近位',approach='自动精确对准',descend='接近方块',close='闭合并确认夹持',lift='夹持确认，正在提起')
    return cue('grasping',numbered('采集中：正在'+names.get(task.phase,'执行抓取')),'ROBOT GRASP - '+task.phase.upper(),theme='auto')


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
