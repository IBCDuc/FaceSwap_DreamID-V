import os
from pathlib import Path
import json,time
p=Path(os.environ.get('FS_ROOT','/data/faceswap/experiments'))/'dreamidv/B_5min_run01/status.json';last=None
while True:
    try:
        state=json.loads(p.read_text());key=(state.get('stage'),state.get('completed_windows'),state.get('window'))
        if key!=last:print(json.dumps(state),flush=True);last=key
        if state.get('stage') in ['complete','failed']:break
    except (OSError,ValueError):pass
    time.sleep(5)
