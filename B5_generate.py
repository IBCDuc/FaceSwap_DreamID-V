import os
from pathlib import Path
import ast,copy,json,logging,types,time,hashlib,cv2,torch,numpy as np,subprocess
import torch.nn.functional as F
import dreamidv_wan.wan_swapface as wm
from dreamidv_wan.configs import WAN_CONFIGS,SIZE_CONFIGS
from dreamidv_wan.utils.utils import cache_video
B=Path(os.environ.get('FS_ROOT','/data/faceswap/experiments'))/'dreamidv';R=B/'B_5min_run01';C=B.parent/'checkpoints/CP01_before_hair_AB_20261030';W=R/'windows';W.mkdir(exist_ok=True)
ff=os.environ.get('FS_FFMPEG','/usr/bin/ffmpeg')
def read(path,start=0,count=73):
    cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,start);out=[]
    for _ in range(count):
        ok,f=cap.read()
        if not ok:break
        out.append(f)
    cap.release();assert len(out)==count,(path,start,count,len(out));return out
def write(path,frames):
    proc=subprocess.Popen([ff,'-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','512x512','-r','15','-i','pipe:0','-an','-c:v','libx264','-crf','0','-threads','3','-pix_fmt','yuv420p',str(path)],stdin=subprocess.PIPE)
    for f in frames:proc.stdin.write(f.tobytes())
    proc.stdin.close();assert proc.wait()==0
class CachedText:
    def __init__(self,encoder):self.encoder=encoder;self.model=encoder.model;self.cache={}
    def __call__(self,texts,device):
        key=(tuple(texts),str(device))
        if key not in self.cache:self.cache[key]=self.encoder(texts,device)
        return self.cache[key]
def custom():
    from anchors import custom_generate
    return custom_generate()

def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s');manifest=json.loads((R/'manifest.json').read_text());windows=manifest['windows']
    pending=[w for w in windows if not (W/f"{w['index']:03d}"/'done.json').exists()]
    if not pending:print('ALL_WINDOWS_ALREADY_COMPLETE',flush=True);return
    free,_=torch.cuda.mem_get_info();assert free>26*1024**3,free
    (R/'runtime.json').write_text(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'attention':'Frozen CP01 SDPA implementation','source_reference':str(B/'hair_AB/inputs/source_full_head.png'),'weights':str(C/'weights'),'checkpoint':C.name,'output_postprocessing':'Video-window assembly and audio mux only; no face pasteback or output crossfade'},indent=2))
    cfg=WAN_CONFIGS['swapface'];cfg.sample_fps=15
    pipeline=wm.DreamIDV(config=cfg,checkpoint_dir=str(C/'weights'),dreamidv_ckpt=str(C/'weights/dreamidv.pth'),device_id=0,rank=0,t5_cpu=True)
    pipeline.text_encoder=CachedText(pipeline.text_encoder);pipeline.generate=types.MethodType(custom(),pipeline)
    # Pose projection is a deterministic function of an unchanged input across
    # all 40 conditional/unconditional calls in a window. Cache it exactly.
    original_pose_forward=pipeline.model.project_pose.forward;pose_cache={};cache_check={'verified':False}
    def cached_pose_forward(x):
        key=(tuple(x.shape),str(x.dtype),str(x.device),x.data_ptr())
        if pose_cache.get('key')==key:return pose_cache['value']
        value=original_pose_forward(x)
        if not cache_check['verified']:
            check=original_pose_forward(x);assert torch.equal(value[0],check[0]),'Pose cache not bit-exact';cache_check['verified']=True
            print('POSE_PROJECTION_CACHE_BIT_EXACT',flush=True)
        pose_cache.update(key=key,value=value);return value
    pipeline.model.project_pose.forward=cached_pose_forward
    for win in windows:
        index=win['index'];out=W/f'{index:03d}';out.mkdir(exist_ok=True)
        if (out/'done.json').exists():continue
        pose_cache.clear()
        start=win['start'];length=win['end']-start;overlap=win['overlap'];begin=time.monotonic()
        state={'stage':'generating','window':index+1,'windows':len(windows),'input_seconds':[start/15,win['end']/15],'completed_windows':index,'updated_unix':time.time()}
        (R/'status.json').write_text(json.dumps(state,indent=2));print('WINDOW_START',json.dumps(state),flush=True)
        frames=read(R/'inputs/crop.mp4',start,length);poses=read(R/'inputs/pose.mp4',start,length);masks=read(R/'inputs/mask.mp4',start,length)
        while len(frames)<73:frames.append(frames[-1]);poses.append(poses[-1]);masks.append(masks[-1])
        if overlap:
            prev=W/f'{index-1:03d}'/'raw.mp4';assert prev.exists();prior=read(prev);frames[:overlap]=prior[-overlap:]
        write(out/'known_input.mp4',frames);write(out/'pose.mp4',poses);write(out/'mask.mp4',masks)
        arr=np.stack([f.max(2).astype(np.float32)/255 for f in masks]);spatial=F.interpolate(torch.tensor(arr[:,None],device='cuda'),size=(64,64),mode='area')[:,0]
        temporal=[spatial[0]]+[spatial[i:i+4].amax(0) for i in range(1,73,4)];edit=torch.stack(temporal)[None,None]
        if overlap:edit[:,:,:1+(overlap-1)//4]=0
        pipeline._edit_latent=edit;pipeline._anchor_stats=[]
        paths=[out/'known_input.mp4',out/'mask.mp4',B/'hair_AB/inputs/source_full_head.png',out/'pose.mp4']
        with torch.no_grad():
            generated=pipeline.generate('chang face',list(map(str,paths)),size=SIZE_CONFIGS['832*480'],frame_num=73,shift=5.0,sampling_steps=20,guide_scale_img=4.0,seed=42,offload_model=True)
        assert torch.isfinite(generated).all()
        cache_video(tensor=generated[None],save_file=str(out/'raw.mp4'),fps=15,nrow=1,normalize=True,value_range=(-1,1))
        decoded=read(out/'raw.mp4');assert len(decoded)==73 and len(pipeline._anchor_stats)==20
        assert all(s['protected_latent_max_error']==0 for s in pipeline._anchor_stats)
        info={**win,'generated_frames':73,'elapsed_seconds':time.monotonic()-begin,'anchor_checks':pipeline._anchor_stats,'pose_projection_cache_verified_exact':cache_check['verified'],'overlap_method':'Previous generated last 9 frames inserted into known input; entire first 3 temporal latents locked during denoising. No output crossfade.'}
        tmp=out/'done.tmp';tmp.write_text(json.dumps(info,indent=2));tmp.replace(out/'done.json')
        state.update(completed_windows=index+1,elapsed_last_window=info['elapsed_seconds'],updated_unix=time.time());(R/'status.json').write_text(json.dumps(state,indent=2));print('WINDOW_COMPLETE',index+1,len(windows),round(info['elapsed_seconds'],1),flush=True)
        del generated,decoded,frames,poses,masks,spatial,temporal,edit;torch.cuda.empty_cache()
    (R/'status.json').write_text(json.dumps({'stage':'generation_complete','windows':len(windows),'completed_windows':len(windows),'updated_unix':time.time()},indent=2));print('B5_GENERATION_COMPLETE',flush=True)
