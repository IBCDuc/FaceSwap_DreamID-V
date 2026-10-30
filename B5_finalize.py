import os
from pathlib import Path
import json,cv2,numpy as np,subprocess,time,hashlib,shutil
from PIL import Image,ImageDraw,ImageFont
B=Path(os.environ.get('FS_ROOT','/data/faceswap/experiments'))/'dreamidv';R=B/'B_5min_run01';D=R/'deliverables';D.mkdir(exist_ok=True);C=B.parent/'checkpoints/CP01_before_hair_AB_20261030'
subprocess.run(['python3',str(C/'restore.py')],check=True)
manifest=json.loads((R/'manifest.json').read_text());windows=manifest['windows'];assert all((R/'windows'/f"{w['index']:03d}"/'done.json').exists() for w in windows)
ff=os.environ.get('FS_FFMPEG','/usr/bin/ffmpeg');fp=str(Path(ff).with_name('ffprobe'))
def read(path,start,count):
    cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,start);frames=[]
    for i in range(count):
        ok,f=cap.read();assert ok,(path,start,count,i);frames.append(f)
    cap.release();return frames
def writer(path,width,header=False):
    return subprocess.Popen([ff,'-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{548 if header else 512}','-r','15','-i','pipe:0','-i',str(R/'inputs/audio.m4a'),'-map','0:v:0','-map','1:a:0','-c:v','libx264','-crf','18' if not header else '20','-preset','fast','-threads','6','-pix_fmt','yuv420p','-c:a','copy','-t','300','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
(R/'status.json').write_text(json.dumps({'stage':'assembling_and_verifying','updated_unix':time.time()},indent=2))
main=writer(D/'B_5min.mp4',512);comparison=writer(D/'comparison_5min.mp4',1024,True);count=0;join_metrics=[];previous=None;sample_ids=set(range(0,4500,225));sample_ids|={max(0,w['start']+w['keep_from']+k) for w in windows for k in [-1,0,1]};sample_frames={};sample_orig={}
def emit(frame,original,index,is_join=False):
    global count,previous
    assert index==count,(index,count);main.stdin.write(frame.tobytes())
    panel=np.zeros((548,1024,3),np.uint8);panel[36:,:512]=original;panel[36:,512:]=frame
    cv2.putText(panel,'Original head-following crop',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA);cv2.putText(panel,'DreamID-V B / long-video trial',(520,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA);comparison.stdin.write(panel.tobytes())
    if index in sample_ids:sample_frames[index]=frame.copy();sample_orig[index]=original.copy()
    if is_join and previous is not None:join_metrics.append({'frame':index,'seconds':index/15,'output_mean_abs_frame_change':float(np.abs(frame.astype(float)-previous.astype(float)).mean())})
    previous=frame.copy();count+=1
for seg in manifest['segments']:
    if not seg['face_present']:
        cap=cv2.VideoCapture(str(R/'inputs/crop.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,seg['start'])
        for i in range(seg['start'],seg['end']):
            ok,f=cap.read();assert ok;emit(f,f,i)
        cap.release();continue
    for w in [x for x in windows if x['segment_start']==seg['start']]:
        raw=read(R/'windows'/f"{w['index']:03d}"/'raw.mp4',0,73);n=w['end']-w['start'];orig=read(R/'inputs/crop.mp4',w['start'],n)
        for j in range(w['keep_from'],n):emit(raw[j],orig[j],w['start']+j,j==w['keep_from'] and w['overlap']>0)
assert count==4500,count
for proc in [main,comparison]:proc.stdin.close();assert proc.wait()==0
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15);ids=list(range(0,4500,225));canvas=Image.new('RGB',(1024,5*278),'#121820');draw=ImageDraw.Draw(canvas)
for j,i in enumerate(ids):
    xx=j%4*256;yy=j//4*278;canvas.paste(Image.fromarray(cv2.cvtColor(sample_frames[i],cv2.COLOR_BGR2RGB)).resize((256,256)),(xx,yy+22));draw.text((xx+4,yy+3),f'B | {i/15:.0f}s',font=font,fill='white')
canvas.save(D/'timeline.jpg',quality=93)
# Save the largest detected temporal joins for targeted visual QA.
top=sorted(join_metrics,key=lambda x:x['output_mean_abs_frame_change'],reverse=True)[:8];join_canvas=Image.new('RGB',(960,len(top)*346),'#121820');jd=ImageDraw.Draw(join_canvas)
for row,item in enumerate(top):
    center=item['frame']
    for col,idx in enumerate([center-1,center,center+1]):
        frame=sample_frames[idx];yy=row*346;join_canvas.paste(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((320,320)),(col*320,yy+26));jd.text((col*320+4,yy+4),f'{idx/15:.2f}s | join {center/15:.2f}s',font=font,fill='white')
join_canvas.save(D/'join_QA.jpg',quality=94)
verification={'expected_frames':4500,'fps':15,'duration_seconds':300,'windows':len(windows),'join_diagnostics':join_metrics,'note':'Frame difference is a QA locator, not a naturalness score. Long-video temporal seams, identity drift, hair and lip motion need visual review.','outputs':{}}
for name in ['B_5min.mp4','comparison_5min.mp4']:
    info=json.loads(subprocess.check_output([fp,'-v','error','-show_streams','-show_format','-of','json',str(D/name)]));v=next(s for s in info['streams'] if s['codec_type']=='video');assert int(v['nb_frames'])==4500 and v['r_frame_rate']=='15/1';assert abs(float(v['duration'])-300)<.01;assert any(s['codec_type']=='audio' for s in info['streams'])
    cap=cv2.VideoCapture(str(D/name));n=0
    while cap.read()[0]:n+=1
    cap.release();assert n==4500,n
    verification['outputs'][name]={'frames':n,'fps':15,'size':[v['width'],v['height']],'duration':v['duration'],'audio':True,'bytes':(D/name).stat().st_size}
(D/'verification.json').write_text(json.dumps(verification,indent=2));(D/'manifest.json').write_text(json.dumps(manifest,indent=2))
shutil.copyfile(R/'preparation.json',D/'preparation.json');shutil.copyfile(R/'tracking_QA.jpg',D/'tracking_QA.jpg')
shutil.copyfile(R/'runtime.json',D/'runtime.json')
protocol={'baseline_checkpoint':C.name,'base_B_settings':str(B/'hair_AB/protocol.json'),'long_video_adaptations':['15 FPS retained from source instead of 24 FPS short test','Smoothed head-following crop at 512x512','73-frame windows with stride 64','Previous 9 generated frames used as known context and first three temporal latents locked','Slides/no-face segments preserved from original via letterbox; generation skipped','Output windows joined chronologically; no face pasteback, output crossfade or enhancer'],'model_weights_modified':False,'CP01_unchanged':True,'quality_status':'Pending visual review'}
(D/'protocol.json').write_text(json.dumps(protocol,indent=2));(R/'status.json').write_text(json.dumps({'stage':'complete','output':str(D/'B_5min.mp4'),'frames':4500,'duration':300,'updated_unix':time.time()},indent=2));print('B5_COMPLETE',verification['outputs'],flush=True)
