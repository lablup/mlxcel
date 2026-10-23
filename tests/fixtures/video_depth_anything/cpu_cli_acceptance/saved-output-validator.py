import argparse,hashlib,json,pathlib,time
import numpy as np
from PIL import Image
from safetensors.numpy import load_file
p=argparse.ArgumentParser();p.add_argument('kind',choices=['relative','metric']);p.add_argument('device',choices=['cpu','gpu']);p.add_argument('frames',type=int);a=p.parse_args()
out=pathlib.Path(f'/home/devops/Development/wt-epic-2382-issue-2352/target/epic-2382-vda-reference/cli-{a.device}-{a.kind}-{a.frames}')
receipt=json.loads((out/'capture-receipt.json').read_text());assert (out/'exit.status').read_text().strip()=='0'
log=(out/'cli.log').read_text();assert f'Runtime device: {a.device.upper()}' in log
if a.device=='cpu':assert 'Running on the CPU because MLXCEL_DEVICE=cpu' in log
else:assert 'gfx1151' in log
arrays=load_file(out/'result.safetensors');assert set(arrays)=={'depth'};depth=arrays['depth'];assert depth.shape==(a.frames,96,128) and depth.dtype==np.float32 and np.isfinite(depth).all()
meta=json.loads((out/'meta.json').read_text());assert meta['metric']==(a.kind=='metric');assert meta['units']==('metres' if a.kind=='metric' else 'relative_disparity');assert meta['frames']==a.frames
previews=sorted((out/'depth').glob('*.png'));assert [f.name for f in previews]==[f'{i:06}.png' for i in range(a.frames)]
for path in previews:
 data=path.read_bytes();assert data[:8]==b'\x89PNG\r\n\x1a\n' and data[24:26]==bytes([16,0])
 with Image.open(path) as im:assert im.size==(128,96)
receipt.update(exit=0,validated_unix=time.time(),runtime_device_log_verified=True,output_sha256=hashlib.sha256((out/'result.safetensors').read_bytes()).hexdigest(),shape=list(depth.shape),dtype=str(depth.dtype),previews=len(previews),metadata=meta,validation_note='Independent saved-output validator; original wrapper display-case assertion corrected without rerunning inference')
(out/'capture-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');(out/'validation.status').write_text('0\n');print(json.dumps({'kind':a.kind,'device':a.device,'shape':list(depth.shape),'previews':len(previews),'units':meta['units'],'output_sha256':receipt['output_sha256']}))
