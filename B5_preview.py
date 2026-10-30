import os
from pathlib import Path
import json,cv2,numpy as np,subprocess,sys
B=Path(os.environ.get('FS_ROOT','/data/faceswap/experiments'))/'dreamidv';R=B/'B_5min_run01';D=R/'previews';D.mkdir(exist_ok=True)
seconds=int(sys.argv[1]) if len(sys.argv)>1 else 15
manifest=json.loads((R/'manifest.json').read_text());want=seconds*15
needed=[w for w in manifest['windows'] if w['start']+w['keep_from']<want]
if not all((R/'windows'/f"{w['index']:03d}"/'done.json').exists() for w in needed):print('PREVIEW_WAITING');raise SystemExit(2)
ff=os.environ.get('FS_FFMPEG','/usr/bin/ffmpeg')
def read(path,start,count):
    cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,start);out=[]
    for _ in range(count):
        ok,f=cap.read();assert ok;out.append(f)
    cap.release();return out
frames=[]
for seg in manifest['segments']:
    if seg['start']>=want:break
    if not seg['face_present']:frames.extend(read(R/'inputs/crop.mp4',seg['start'],min(want,seg['end'])-seg['start']))
    else:
        for w in [x for x in needed if x['segment_start']==seg['start']]:
            raw=read(R/'windows'/f"{w['index']:03d}"/'raw.mp4',0,73);frames.extend(raw[w['keep_from']:min(w['end'],want)-w['start']])
assert len(frames)==want,len(frames)
def encode(name,comparison=False):
    proc=subprocess.Popen([ff,'-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1024x548' if comparison else '512x512','-r','15','-i','pipe:0','-i',str(R/'inputs/audio.m4a'),'-map','0:v:0','-map','1:a:0','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','copy','-t',str(seconds),'-movflags','+faststart',str(D/name)],stdin=subprocess.PIPE)
    original=read(R/'inputs/crop.mp4',0,want) if comparison else None
    for i,f in enumerate(frames):
        if comparison:
            panel=np.zeros((548,1024,3),np.uint8);panel[36:,:512]=original[i];panel[36:,512:]=f
            for x,label in [(8,'Original'),(520,'B long-video preview')]:cv2.putText(panel,label,(x,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
            f=panel
        proc.stdin.write(f.tobytes())
    proc.stdin.close();assert proc.wait()==0
encode(f'preview_{seconds}s.mp4');encode(f'comparison_{seconds}s.mp4',True);print('PREVIEW_READY',seconds,flush=True)
