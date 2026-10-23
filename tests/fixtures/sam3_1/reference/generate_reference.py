import mlx.core as mx, mlx.nn as nn, sys
sys.path.insert(0,'/tmp/epic2382-sam3-reference/lib/python3.12/site-packages')
import ast,types,inspect,json,time,hashlib,os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
from mlx.utils import tree_flatten
SCRIPT=Path(__file__).resolve().parent
W=Path(os.environ['MLXCEL_REPO']); O=Path(os.environ['SAM31_REFERENCE_OUTPUT']); P=Path(os.environ['SAM31_CHECKPOINT'])
O.mkdir(parents=True,exist_ok=True)
root=Path(os.environ['SAM31_UPSTREAM'])/'mlx_vlm'
for name,path in [('mlx_vlm',root),('mlx_vlm.models',root/'models'),('mlx_vlm.models.sam3',root/'models/sam3'),('mlx_vlm.models.sam3_1',root/'models/sam3_1')]:
 m=types.ModuleType(name);m.__path__=[str(path)];sys.modules[name]=m
base=types.ModuleType('mlx_vlm.models.base');base.inspect=inspect;base.dataclass=dataclass
for node in ast.parse(SCRIPT/'base_snapshot.py'.read_text()).body:
 if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in ['BaseModelConfig','install_auto_processor_patch']:exec(compile(ast.Module(body=[node],type_ignores=[]),'pinned-base.py','exec'),base.__dict__)
sys.modules[base.__name__]=base
stream=types.ModuleType('mlx_vlm._stream_cleanup');exec(SCRIPT/'stream_cleanup.py'.read_text(),stream.__dict__);sys.modules[stream.__name__]=stream
kernels=types.ModuleType('mlx_vlm.models.kernels');exec(SCRIPT/'upstream_kernels.py'.read_text(),kernels.__dict__);sys.modules[kernels.__name__]=kernels
interp=types.ModuleType('mlx_vlm.models.interpolate');exec(SCRIPT/'upstream_interpolate.py'.read_text(),interp.__dict__);sys.modules[interp.__name__]=interp
from mlx_vlm.models.sam3_1.config import ModelConfig
from mlx_vlm.models.sam3_1.sam3_1 import Model
from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor
from mlx_vlm.models.sam3.generate import Sam3Predictor
from mlx_vlm.models.sam3_1.generate import predict_multi,_get_backbone_features,_init_tracker_state,_masks_to_detection
from transformers import AutoTokenizer
mx.set_default_device(mx.gpu)
assert 'python-exact-core' in mx.__file__,mx.__file__
maps=Path('/proc/self/maps').read_text();assert 'python-exact-core/libmlx' in maps
(O/'runtime-maps.txt').write_text(maps)
print('Runtime',mx.__file__,mx.__version__,mx.default_device(),flush=True)
model=Model(ModelConfig.from_dict(json.loads((P/'config.json').read_text())))
weights=model.sanitize(mx.load(str(P/'model.safetensors')));model.load_weights(list(weights.items()),strict=True);mx.eval(model.parameters())
params={k:{'dtype':str(v.dtype),'shape':list(v.shape)} for k,v in tree_flatten(model.parameters())}
(O/'loaded-parameter-dtypes.json').write_text(json.dumps(params,indent=2)+'\n')
print('Loaded dtype counts',{dt:sum(v['dtype']==dt for v in params.values()) for dt in set(v['dtype'] for v in params.values())},flush=True)
processor=Sam3Processor.from_pretrained(str(P));processor._tokenizer=AutoTokenizer.from_pretrained(str(P));predictor=Sam3Predictor(model,processor,score_threshold=.3)
exec(SCRIPT/'capture_hooks.py'.read_text())
cat=Image.open(os.environ['SAM31_CAT_IMAGE']).convert('RGB')
r=predict_multi(predictor,cat,['a cat'],score_threshold=.3)
np.savez_compressed(O/'cats-reference.npz',boxes=r.boxes,scores=r.scores,masks=r.masks)
print('CATS',r.boxes.tolist(),r.scores.tolist(),r.masks.sum(axis=(1,2)).tolist(),flush=True)
frames=sorted(Path(os.environ['SAM31_VIDEO_FRAMES']).glob('*.png'));assert len(frames)==3
outputs=[];state=None;calls=0
for i,path in enumerate(frames):
 image=Image.open(path).convert('RGB');inputs=processor.preprocess_image(image);pixels=mx.array(inputs['pixel_values']);print('INPUT',i,str(pixels.dtype),pixels.shape,flush=True)
 if i==0:
  r=predict_multi(predictor,image,['a car'],score_threshold=.15)
  assert len(r.scores)>=3,('need genuine >=3 detected cars',len(r.scores))
  n=len(r.scores);features=_get_backbone_features(model,pixels);state=_init_tracker_state(model,features,list(r.masks))
 else:
  features=_get_backbone_features(model,pixels);out=model.track_step(state,features,i);calls+=1
  print('TRACK_DTYPES',i,{k:str(v.dtype) for k,v in out.items() if hasattr(v,'dtype')},flush=True)
  np.savez_compressed(O/f'video-raw-{i}.npz', **{k:np.array(out[k].astype(mx.float32)) for k in ['pred_masks_high_res','object_score_logits']})
  r=_masks_to_detection(out,n,image.size)
 assert r.masks.shape==(n,image.height,image.width),r.masks.shape
 assert r.boxes.shape==(n,4) and r.scores.shape==(n,)
 assert np.isfinite(r.boxes).all() and np.isfinite(r.scores).all()
 np.savez_compressed(O/f'video-frame-{i}.npz',boxes=r.boxes,scores=r.scores,masks=r.masks,object_ids=np.arange(1,n+1))
 print('FRAME',i,'scores',r.scores.tolist(),'foreground',r.masks.sum(axis=(1,2)).tolist(),flush=True)
 outputs.append({'frame':i,'scores':r.scores.tolist(),'foreground':r.masks.sum(axis=(1,2)).tolist()})
assert calls==len(frames)-1
(O/'video-summary.json').write_text(json.dumps({'track_step_calls':calls,'frames':outputs},indent=2)+'\n')
print('COMPLETE',flush=True)
