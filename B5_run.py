import os
from pathlib import Path
import os,sys,subprocess,ast
B=Path(os.environ.get('FS_ROOT','/data/faceswap/experiments'))/'dreamidv';R=B/'B_5min_run01';C=B.parent/'checkpoints/CP01_before_hair_AB_20261030'
if '--inside' not in sys.argv:
    env=os.environ.copy()
    for k,v in {'HOME':R/'home','TMPDIR':R/'tmp','HF_HOME':R/'cache/hf','TORCH_HOME':R/'cache/torch','XDG_CACHE_HOME':R/'cache','CUDA_CACHE_PATH':R/'cache/cuda','TRITON_CACHE_DIR':R/'cache/triton'}.items():Path(v).mkdir(parents=True,exist_ok=True);env[k]=str(v)
    import glob
    env['LD_LIBRARY_PATH']=':'.join(glob.glob(str(B/'.venv/lib/python3.12/site-packages/nvidia/*/lib')))

    env.update(OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    sys.exit(subprocess.run([str(B/'.venv/bin/python'),__file__,'--inside',*sys.argv[1:]],env=env,cwd=C/'snapshot/dreamidv/repo').returncode)
repo=C/'snapshot/dreamidv/repo';sys.path[:0]=[str(repo),str(repo/'pose')]
import torch,torch.nn.functional as F
torch.set_num_threads(4)
if '--prepare' in sys.argv:
    import B5_prepare;B5_prepare.main();sys.exit(0)
import dreamidv_wan.modules.model as mm
from attention import sdpa
mm.flash_attention=sdpa
import B5_generate;B5_generate.main()
