import os
from pathlib import Path
import urllib.request,json,concurrent.futures,hashlib,shutil,time,threading
B=Path(os.environ.get('FS_ROOT','/data/faceswap/experiments'))/'dreamidv';M=B/'weights';M.mkdir(exist_ok=True);C=B/'download_parts';C.mkdir(exist_ok=True)
items=[]
for repo,meta,wanted in [('XuGuo699/DreamID-V','DreamID-V_files.json',['dreamidv.pth','dw-ll_ucoco_384.onnx','yolox_l.onnx']),('Wan-AI/Wan2.1-T2V-1.3B','Wan2.1-T2V-1.3B_files.json',['Wan2.1_VAE.pth','models_t5_umt5-xxl-enc-bf16.pth','config.json','google/umt5-xxl/special_tokens_map.json','google/umt5-xxl/spiece.model','google/umt5-xxl/tokenizer.json','google/umt5-xxl/tokenizer_config.json'])]:
    data=json.loads((B/meta).read_text())
    for entry in data['siblings']:
        if entry['rfilename'] in wanted:items.append((repo,data['sha'],entry))
chunk=128*1024*1024;tasks=[]
for repo,rev,e in items:
    name=e['rfilename'];size=e['size'];dest=M/name;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest.stat().st_size==size:continue
    for start in range(0,size,chunk):tasks.append((repo,rev,name,size,start,min(start+chunk,size)-1))
lock=threading.Lock();done=0
def fetch(t):
    global done
    repo,rev,name,size,start,end=t;p=C/(name.replace('/','_')+f'.{start}')
    if not(p.exists() and p.stat().st_size==end-start+1):
        for attempt in range(4):
            try:
                url=f'https://huggingface.co/{repo}/resolve/{rev}/{name}?download=true&chunk={start}'
                offset=p.stat().st_size if p.exists() else 0
                req=urllib.request.Request(url+f'&resume={offset}',headers={'Range':f'bytes={start+offset}-{end}'})
                with urllib.request.urlopen(req,timeout=120) as response:
                    if size>chunk:assert response.status==206 and response.headers.get('Content-Range','').startswith(f'bytes {start+offset}-{end}/'),dict(response.headers)
                    with p.open('ab' if offset else 'wb') as f:shutil.copyfileobj(response,f,1024*1024)
                assert p.stat().st_size==end-start+1
                break
            except Exception:
                if attempt==3:raise
                time.sleep(2)
    with lock:
        done+=1;print('CHUNK',done,'/',len(tasks),name,start,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:list(pool.map(fetch,tasks))
records=[]
for repo,rev,e in items:
    name=e['rfilename'];size=e['size'];dest=M/name
    if not dest.exists():
        with dest.open('wb') as out:
            for start in range(0,size,chunk):
                with (C/(name.replace('/','_')+f'.{start}')).open('rb') as f:shutil.copyfileobj(f,out,1024*1024)
    h=hashlib.sha256()
    with dest.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    expected=e.get('lfs',{}).get('sha256');digest=h.hexdigest()
    if expected:assert digest==expected,(name,digest,expected)
    records.append({'file':name,'bytes':size,'sha256':digest,'repo':repo,'revision':rev})
    print('FILE_VERIFIED',name,flush=True)
    for start in range(0,size,chunk):(C/(name.replace('/','_')+f'.{start}')).unlink(missing_ok=True)
pose=B/'repo/pose/models';pose.mkdir(exist_ok=True)
for name in ['dw-ll_ucoco_384.onnx','yolox_l.onnx']:
    if not(pose/name).exists():(pose/name).symlink_to(M/name)
(B/'weights.json').write_text(json.dumps(records,indent=2));print('WEIGHTS_READY',flush=True)
