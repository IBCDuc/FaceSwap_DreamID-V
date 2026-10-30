import os
from pathlib import Path
import json,subprocess,time,traceback
P=Path(os.environ.get('FS_ROOT','/data/faceswap/experiments'));R=P/'dreamidv/B_5min_run01';SCRIPT=Path(__file__).resolve().parent
try:
    if not (R/'preparation.json').exists():
        with (R/'prepare.log').open('a') as log:subprocess.run(['python3',str(SCRIPT/'B5_run.py'),'--prepare'],stdout=log,stderr=subprocess.STDOUT,check=True)
    while True:
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        completed=len(list((R/'windows').glob('*/done.json')))
        if completed>=len(json.loads((R/'manifest.json').read_text())['windows']):break
        if free<26*1024:
            (R/'status.json').write_text(json.dumps({'stage':'waiting_for_gpu','completed_windows':completed,'windows':64,'free_vram_MiB':free,'required_free_vram_MiB':26*1024,'reason':'Shared GPU occupied; completed windows retained. No other process is interrupted.','updated_unix':time.time()},indent=2));time.sleep(15);continue
        with (R/'generation.log').open('a') as log:result=subprocess.run(['python3',str(SCRIPT/'B5_run.py')],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode==0:break
        tail=(R/'generation.log').read_text(errors='replace')[-10000:]
        if 'out of memory' in tail.lower() or 'OutOfMemoryError' in tail:
            (R/'status.json').write_text(json.dumps({'stage':'waiting_for_gpu','completed_windows':len(list((R/'windows').glob('*/done.json'))),'windows':64,'reason':'GPU allocation changed during generation; automatic resume waits for sufficient free VRAM.','updated_unix':time.time()},indent=2));time.sleep(20);continue
        raise RuntimeError('Generation failed; inspect generation.log')
    with (R/'finalize.log').open('a') as log:subprocess.run([str(P/'dreamidv/.venv/bin/python'),str(SCRIPT/'B5_finalize.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
except Exception as e:
    traceback.print_exc();(R/'status.json').write_text(json.dumps({'stage':'failed','error':str(e),'updated_unix':time.time(),'resume':'python3 '+str(P/'B5_job.py')},indent=2));raise
