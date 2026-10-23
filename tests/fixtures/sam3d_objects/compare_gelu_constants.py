"""Storage-level round-to-nearest-even BF16 normalization of frozen probe output."""
import hashlib,json
from pathlib import Path
import numpy as np
p=Path(__import__('sys').argv[1]); blob=p.read_bytes(); n=int.from_bytes(blob[:8],'little'); header=json.loads(blob[8:8+n]); offset=8+n
values={}
for name in ('gelu.x','gelu.out','gelu.f32_constants'):
    entry=header[name]; lo,hi=entry['data_offsets']; raw=blob[offset+lo:offset+hi]
    values[name]=(np.frombuffer(raw,'<u2').astype(np.uint32)<<16).view(np.float32) if entry['dtype']=='BF16' else np.frombuffer(raw,'<f4')
bits=values['gelu.f32_constants'].view(np.uint32)
rounded=((bits+0x7fff+((bits>>16)&1))>>16).astype(np.uint16)
actual=(rounded.astype(np.uint32)<<16).view(np.float32); expected=values['gelu.out']
print(json.dumps({'fixture_sha256':hashlib.sha256(blob).hexdigest(),'algorithm':'BF16 RNE: (f32bits + 0x7fff + ((f32bits >> 16) & 1)) >> 16','mismatches':int(np.count_nonzero(actual!=expected)),'elements':int(expected.size),'maximum_absolute_error':float(np.max(np.abs(actual-expected)))}))
