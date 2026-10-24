import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
from exobalancelab.hjb_transport import source_hashes, triple_reference

root = Path.cwd()
out = root / 'logs/transport_v3/coarse_reference'
out.mkdir()
sources = source_hashes()
for name in ['benchmark_transport.py', 'hjb_analytic_reference.py']:
    sources[name] = hashlib.sha256((root / 'examples' / name).read_bytes()).hexdigest()
for name in sources:
    source = root / ('examples' if name in ['benchmark_transport.py', 'hjb_analytic_reference.py'] else 'exobalancelab') / name
    shutil.copy2(source, out / name)
start = time.perf_counter()
axes, value = triple_reference(449, .01)
field = out / 'triple_n449_dt0.01.npz'
np.savez_compressed(field, position=axes[0], velocity=axes[1], value=value.min(axis=2))
manifest = dict(sources=sources, field=field.name, sha256=hashlib.sha256(field.read_bytes()).hexdigest(), resolution=449, time_step_s=.01, duration_s=.5, wall_s=time.perf_counter()-start, allocator='bfc')
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
shutil.copy2(__file__, out / 'record_coarse_reference.py')
print(json.dumps(manifest, indent=2), flush=True)
