"""Bounded exact-native BF16 graph probe; no model parameters loaded."""
import argparse, hashlib, importlib.util, json, math, sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(); p.add_argument('--fixtures',type=Path,required=True); p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
sys.path.insert(0,str(a.fixtures))
from provenance import verify_source_runtime,sha
import mlx.core as mx
import mlx.nn as nn
mx.set_default_device(mx.gpu)
runtime=verify_source_runtime(a.source,mx)
spec=importlib.util.spec_from_file_location('objects_layers',a.source/'mlx_vlm/models/sam3d_objects/layers.py'); layers=importlib.util.module_from_spec(spec); spec.loader.exec_module(layers)
a.output.mkdir(parents=True,exist_ok=False)
rng=np.random.default_rng(2381); arrays={}; counts={}
for trial in range(32):
 x=mx.array(rng.normal(0,2,(1,2,2,8)).astype(np.float32),mx.bfloat16)
 s=mx.array(rng.normal(0,1,(1,1,2,8)).astype(np.float32),mx.bfloat16)
 t=mx.array(rng.normal(0,1,(1,1,2,8)).astype(np.float32),mx.bfloat16)
 g=mx.array(rng.normal(1,.5,(2,8)).astype(np.float32),mx.bfloat16)
 xf=x.astype(mx.float32)
 cases={'modulate':({'x':x,'scale':s,'shift':t},layers.modulate(x,s,t),x*(1+s)+t), 'residual':({'x':x,'h':s,'gate':t},layers.gated_residual(x,s,t),x+s*t), 'head_norm':({'x':x,'gamma':g},layers._head_norm(x,g),(xf*mx.rsqrt(mx.maximum(mx.sum(xf*xf,axis=-1,keepdims=True),1e-24))*g*math.sqrt(8)).astype(x.dtype))}
 for name,(inputs,out,eager) in cases.items():
  mx.eval(out,eager); n=int(mx.sum(out!=eager).item())
  if name not in counts or n>counts[name]:
   counts[name]=n
   arrays.update({name+'.'+k:v for k,v in {**inputs,'out':out,'eager':eager}.items()})
 if all(counts.get(k,0)>0 for k in cases): break
x=mx.array(np.linspace(-5,5,8192,dtype=np.float32),mx.bfloat16)
y=nn.gelu(x)
z=(x*mx.array(.5,mx.float32))*(1+mx.erf(x*mx.array(1/math.sqrt(2),mx.float32)))
mx.eval(y,z); counts['gelu_f32_constants']=int(mx.sum(y!=z).item())
arrays.update({'gelu.x':x,'gelu.out':y,'gelu.f32_constants':z})
mx.save_safetensors(str(a.output/'layers.safetensors'),arrays)
receipt={'generator_sha256':sha(Path(__file__)),'runtime':runtime,'device':str(mx.default_device()),'dtype':'bfloat16','differences':counts,'output_sha256':sha(a.output/'layers.safetensors'),'shapes':{k:list(v.shape) for k,v in arrays.items()}}
(a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt),flush=True)
