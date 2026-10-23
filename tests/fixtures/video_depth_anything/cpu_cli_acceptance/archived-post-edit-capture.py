import argparse, hashlib, io, json, os, pathlib, struct, subprocess, time
import numpy as np
from PIL import Image
from safetensors.numpy import load_file
p=argparse.ArgumentParser(); p.add_argument('kind',choices=['relative','metric']); p.add_argument('--device',choices=['cpu','gpu'],required=True); p.add_argument('--frames',type=int,choices=[40,64],required=True); p.add_argument('--preflight-only',action='store_true'); a=p.parse_args()
w=pathlib.Path('/home/devops/Development/wt-epic-2382-issue-2352'); cache=w/'target/epic-2382-vda-reference'; fixtures=w/'tests/fixtures/video_depth_anything'; ff=pathlib.Path('/tmp/mlxcel-ffmpeg/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg'); binary=w/'target/test-fast/mlxcel'
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
 return h.hexdigest()
select='+'.join(f'eq(n\\,{i})' for i in range(a.frames))
cmd=[str(ff),'-hide_banner','-loglevel','error','-i',str(fixtures/'davis_64.mp4'),'-vf',f"select='{select}',setpts=N/FRAME_RATE/TB",'-fps_mode','vfr','-f','image2pipe','-vcodec','png','-']
raw=subprocess.check_output(cmd); pos=0; decoded=[]
while pos<len(raw):
 start=pos; assert raw[pos:pos+8]==b'\x89PNG\r\n\x1a\n'; pos+=8
 while True:
  n=struct.unpack('>I',raw[pos:pos+4])[0]; kind=raw[pos+4:pos+8]; pos+=12+n
  if kind==b'IEND': break
 decoded.append(np.asarray(Image.open(io.BytesIO(raw[start:pos])).convert('RGB')))
frames=np.stack(decoded); assert frames.shape==(a.frames,96,128,3)
contract=json.loads((fixtures/'reference-contract.json').read_text()); pixels=hashlib.sha256(frames[:40].tobytes()).hexdigest(); assert pixels==contract['decoded_rgb_sha256'],(pixels,contract)
print('INPUT_EXACT',frames.shape,pixels,flush=True)
if a.preflight_only: raise SystemExit(0)
out=cache/f'cli-{a.device}-{a.kind}-{a.frames}'; out.mkdir(exist_ok=False)
checkpoint=cache/'checkpoints'/a.kind
command=[str(binary),'predict','--model',str(checkpoint),'--video',str(fixtures/'davis_64.mp4'),'--max-frames',str(a.frames),'--output',str(out)]
env=os.environ.copy(); env['MLXCEL_DEVICE']=a.device; env['MLX_ENABLE_TF32']='0'; env['PATH']=str(ff.parent)+':'+env['PATH']
source={str(f.relative_to(w)):digest(f) for f in sorted((w/'src/vision/dense/video_depth_anything').glob('*.rs'))}
for rel in ['src/commands/predict.rs','src/multimodal/video.rs','src/vision/encoders/dinov2.rs']: source[rel]=digest(w/rel)
receipt={'command':command,'device':a.device,'tf32':'0','binary_sha256':digest(binary),'source_sha256':source,'reviewed_source_manifest_sha256':digest('/tmp/epic2382-vda-final-source.sha256'),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip(),'checkpoint_sha256':{f.name:digest(f) for f in sorted(checkpoint.iterdir()) if f.is_file()},'video_sha256':digest(fixtures/'davis_64.mp4'),'decoded_rgb_sha256':hashlib.sha256(frames.tobytes()).hexdigest(),'first40_rgb_sha256':pixels,'decoder_command':cmd,'decoder_sha256':digest(ff),'options':{'max_frames':a.frames,'target_fps':None},'state':'unverified capture','started_unix':time.time()}
(out/'capture-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
with open(out/'cli.log','w') as log: rc=subprocess.run(command,cwd=w,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
(out/'exit.status').write_text(str(rc)+'\n'); receipt['exit']=rc; receipt['finished_unix']=time.time()
if rc==0:
 runtime_log=(out/'cli.log').read_text(); assert f'Runtime device: {a.device.upper()}' in runtime_log
 if a.device=='cpu': assert 'Running on the CPU because MLXCEL_DEVICE=cpu' in runtime_log
 receipt['runtime_device_log_verified']=True
 tensors=load_file(out/'result.safetensors'); assert set(tensors)=={'depth'}; depth=tensors['depth']; assert depth.shape==(a.frames,96,128) and depth.dtype==np.float32 and np.isfinite(depth).all()
 meta=json.loads((out/'meta.json').read_text()); assert meta['metric']==(a.kind=='metric'); assert meta['units']==('metres' if a.kind=='metric' else 'relative_disparity'); assert meta['frames']==a.frames
 previews=sorted((out/'depth').glob('*.png')); assert [f.name for f in previews]==[f'{i:06}.png' for i in range(a.frames)]
 for preview in previews:
  png=preview.read_bytes(); assert png[:8]==b'\x89PNG\r\n\x1a\n' and png[24:26]==bytes([16,0])
  with Image.open(preview) as im: assert im.size==(128,96)
 receipt.update(output_sha256=digest(out/'result.safetensors'),shape=list(depth.shape),dtype=str(depth.dtype),previews=len(previews),metadata=meta)
(out/'capture-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt),flush=True)
raise SystemExit(rc)
