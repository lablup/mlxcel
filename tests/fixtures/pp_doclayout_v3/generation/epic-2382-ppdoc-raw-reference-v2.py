import argparse,ast,dataclasses,hashlib,importlib,inspect,json,os,pathlib,resource,sys,time,types
P=pathlib.Path
ROOT=P('/home/devops/Development/wt-epic-2382-issue-2412/target/epic-2382-ppdoc-reference')
RUNTIME=P('/home/devops/Development/mlxcel/.cache/epic-2382/python-exact-core')
CHECKPOINT_ROOT=P('/home/devops/Development/mlxcel/.cache/epic-2382/preflight-ppdoc-checkpoints')
PINS={'paddle':'e4f489dc536556fc1be02b973ba06756e0d4a2ac','hashnuke':'cd50996ad0c7e35a7e20c27480f99cec1822cf7b'}
def require(ok,message):
 if not ok:raise RuntimeError(message)
def digest(path):
 h=hashlib.sha256()
 with P(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def authenticated_json(path,expected):
 require(digest(path)==expected,'manifest/receipt identity changed '+str(path))
 return json.loads(P(path).read_text())
def verify_files(manifest,root,expected):
 names=[]
 for entry in manifest['files']:
  path=P(entry['path']);require(path.resolve().is_relative_to(root.resolve()),'manifest path escapes root')
  names.append(str(path.relative_to(root)))
  require(digest(path)==entry['sha256'],'file changed '+str(path))
 require(len(names)==len(set(names)) and set(names)==set(expected),'manifest file inventory mismatch')

def namespace(name,path):
 m=types.ModuleType(name);m.__path__=[str(path)];sys.modules[name]=m;return m
p=argparse.ArgumentParser();p.add_argument('kind',choices=PINS);p.add_argument('--device',choices=['cpu','gpu'],default='gpu');args=p.parse_args()
require(os.environ.get('MLX_ENABLE_TF32')=='0','TF32 must explicitly be disabled')
require(digest(RUNTIME/'libmlx.so')=='9e38e9292550ea8eec9eb1877d06b72f5155ff382ac0e4ec1054818e15276094','native identity changed')
require(digest(RUNTIME/'python/mlx/core.cpython-312-x86_64-linux-gnu.so')=='f8ec0f363bc6eab1cda644a2579ae43aa663939f30ace531d5fb8cc33c7090cc','binding identity changed')
source_manifest=authenticated_json(ROOT/'upstream-manifest.json','ef120ba06b56e27f54accc9ccd1710fe16b62c177b40c257d16495dce2c9059c')
require(source_manifest['revision']=='4319ef2c837b7226b758c3dd69edda51b7079785','source revision')
verify_files(source_manifest,ROOT/'upstream',['LICENSE', 'mlx_vlm/models/base.py', 'mlx_vlm/models/falcon_ocr/README.md', 'mlx_vlm/models/falcon_ocr/__init__.py', 'mlx_vlm/models/falcon_ocr/config.py', 'mlx_vlm/models/falcon_ocr/falcon_ocr.py', 'mlx_vlm/models/falcon_ocr/language.py', 'mlx_vlm/models/falcon_ocr/layout.py', 'mlx_vlm/models/falcon_ocr/processing_falcon_ocr.py', 'mlx_vlm/models/pp_doclayout_v3/README.md', 'mlx_vlm/models/pp_doclayout_v3/__init__.py', 'mlx_vlm/models/pp_doclayout_v3/backbone.py', 'mlx_vlm/models/pp_doclayout_v3/config.py', 'mlx_vlm/models/pp_doclayout_v3/convert.py', 'mlx_vlm/models/pp_doclayout_v3/decoder.py', 'mlx_vlm/models/pp_doclayout_v3/encoder.py', 'mlx_vlm/models/pp_doclayout_v3/pp_doclayout_v3.py', 'mlx_vlm/models/rt_detr_v2/README.md', 'mlx_vlm/models/rt_detr_v2/__init__.py', 'mlx_vlm/models/rt_detr_v2/config.py', 'mlx_vlm/models/rt_detr_v2/convert.py', 'mlx_vlm/models/rt_detr_v2/generate.py', 'mlx_vlm/models/rt_detr_v2/language.py', 'mlx_vlm/models/rt_detr_v2/processing_rt_detr_v2.py', 'mlx_vlm/models/rt_detr_v2/rt_detr_v2.py', 'mlx_vlm/models/rt_detr_v2/transformer.py', 'mlx_vlm/models/rt_detr_v2/vision.py', 'mlx_vlm/utils.py', 'pyproject.toml', 'mlx_vlm/models/kernels.py'])
checkpoint=CHECKPOINT_ROOT/args.kind/PINS[args.kind]
checkpoint_pins={'paddle':'6e929ff81c52d99730a9ca106dd760deae5b41e5f99bd3ff499f5e46dd220155','hashnuke':'c92c30d4d63925fc79ac84a0834cb51b2c02653777482fe19ef54825d6c239c3'}
checkpoint_manifest=authenticated_json(checkpoint/'manifest.json',checkpoint_pins[args.kind])
require(checkpoint_manifest['revision']==PINS[args.kind],'checkpoint revision')
verify_files(checkpoint_manifest,checkpoint,['.gitattributes','README.md','config.json','model.safetensors']+(['inference.yml','preprocessor_config.json'] if args.kind=='paddle' else []))
processor_receipt=authenticated_json(ROOT/'hf-processor-receipt.json','520e5cc3a7a4cdcc5b0a6522a57f88dc0a81ff42e2558487d2d93d28dd2667e8')
page_manifest=authenticated_json(ROOT/'bert-page/manifest.json','b83a6915ad152cc348d45296ef5a85e006e582bc95efa24fdf23cb28a4737a44')
require(digest('/tmp/epic-2382-ppdoc-hf-processor-v1.py')=='56a174dc48ee5f33d07889c8ec340f798d03b2b78dfd30f0cce6c63b808cf308','processor helper changed')
require(digest(processor_receipt['processor_source'])==processor_receipt['processor_source_sha256'],'processor implementation changed')
require(processor_receipt['device']=='cpu' and processor_receipt['shape']==[1,800,800,3] and processor_receipt['dtype']=='float32','processor identity/options')
require(digest(ROOT/'bert-page/two_column_page.png')=='375cc65b7975ddacd9d43a390fb3e7b8973abd7b47ecb7a09cc591bd17e1e6ac','page image changed')
wrapper=P('/tmp/epic-2382-ppdoc-run-raw-v2.sh');launch=P('/tmp/epic-2382-ppdoc-raw-v2-launch.sha256')
launch_entries={}
for line in launch.read_text().splitlines():
 sha,path=line.split(maxsplit=1);require(path not in launch_entries,'duplicate launch path');launch_entries[path]=sha
require(set(launch_entries)=={str(wrapper),str(P(__file__))},'launch inventory')
for path,sha in launch_entries.items():require(digest(path)==sha,'launch artifact changed')
input_file=ROOT/'hf-processor-input.safetensors';require(digest(input_file)=='0023b22357d4fb93a6c6a721c6386df75e44d85eba9020b804c5a2e0fe9b7faf','processor input changed')
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from safetensors.numpy import load_file,save_file
binding=RUNTIME/'python/mlx/core.cpython-312-x86_64-linux-gnu.so'
require(P(mx.__file__).resolve()==binding.resolve() and digest(mx.__file__)=='f8ec0f363bc6eab1cda644a2579ae43aa663939f30ace531d5fb8cc33c7090cc','wrong imported binding')
mapped={line.split()[-1] for line in P('/proc/self/maps').read_text().splitlines() if 'libmlx.so' in line}
require(mapped=={str(RUNTIME/'libmlx.so')},'wrong mapped native runtime')
output_dir=ROOT/'raw-references';out=output_dir/f'{args.kind}_{args.device}_raw.safetensors'
require(not out.exists() and not out.with_suffix('.provenance.json').exists(),'refuse existing output/sidecar')
mx.set_default_device(mx.gpu if args.device=='gpu' else mx.cpu)
require(mx.default_device()==mx.Device(mx.gpu if args.device=='gpu' else mx.cpu,0),'resolved device differs')
source=ROOT/'upstream/mlx_vlm';namespace('mlx_vlm',source);namespace('mlx_vlm.models',source/'models');namespace('mlx_vlm.models.rt_detr_v2',source/'models/rt_detr_v2')
# Execute the exact upstream BaseModelConfig definition only. This avoids importing unrelated language-model caches; no model/math definition is replaced.
base_path=source/'models/base.py';tree=ast.parse(base_path.read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BaseModelConfig');base=types.ModuleType('mlx_vlm.models.base');base.__dict__.update({'dataclass':dataclasses.dataclass,'inspect':inspect});exec(compile(ast.Module(body=[node],type_ignores=[]),str(base_path),'exec'),base.__dict__);sys.modules[base.__name__]=base
from mlx_vlm.models.pp_doclayout_v3.config import ModelConfig
from mlx_vlm.models.pp_doclayout_v3.pp_doclayout_v3 import Model
config=ModelConfig.from_dict(json.loads((checkpoint/'config.json').read_text()));model=Model(config);weights=Model.sanitize(mx.load(str(checkpoint/'model.safetensors')));model.load_weights(list(weights.items()),strict=True);model.eval()
loaded={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in tree_flatten(model.parameters())}
require(len(loaded)==857,'unexpected loaded parameter count');require(all(v['dtype']=='mlx.core.float32' for v in loaded.values()),'actual loaded dtype is not uniformly F32')
mx.eval(model.parameters());print('LOADED',args.kind,args.device,len(loaded),'float32',flush=True)
inputs=load_file(input_file);require(set(inputs)=={'decoded_rgb_u8','pixel_values_nhwc'},'input keys');x=inputs['pixel_values_nhwc'];require(x.shape==(1,800,800,3) and x.dtype==np.float32 and np.isfinite(x).all(),'input tensor')
decoded=inputs['decoded_rgb_u8'];require(decoded.shape==(1684,1191,3) and decoded.dtype==np.uint8 and hashlib.sha256(decoded.tobytes()).hexdigest()==processor_receipt['decoded_sha256'],'decoded RGB identity')
require(hashlib.sha256(x.tobytes()).hexdigest()==processor_receipt['processed_sha256'],'processed tensor identity')
start=time.monotonic();outputs=model(mx.array(x));require(set(outputs)=={'logits','pred_boxes','order_logits'},'raw output keys');mx.eval(outputs);arrays={k:np.array(v) for k,v in outputs.items()};expected={'logits':(1,300,25),'pred_boxes':(1,300,4),'order_logits':(1,300,300)}
for k,v in arrays.items():require(v.shape==expected[k] and v.dtype==np.float32 and np.isfinite(v).all(),'invalid output '+k)
output_dir=ROOT/'raw-references';output_dir.mkdir(exist_ok=True);out=output_dir/f'{args.kind}_{args.device}_raw.safetensors';require(not out.exists(),'refuse overwrite existing output');save_file(arrays,out,metadata={'upstream_revision':source_manifest['revision'],'device':args.device,'checkpoint_dtype':'float32'})
maps=P('/proc/self/maps').read_text();require(str(RUNTIME/'libmlx.so') in maps,'wrong mapped native runtime');modules={k:getattr(v,'__file__',None) for k,v in sys.modules.items() if k.startswith(('mlx.','mlx_vlm.'))};require(str(RUNTIME/'python') in str(mx.__file__),'wrong binding module')
receipt={'processor_receipt_sha256':digest(ROOT/'hf-processor-receipt.json'),'processor_helper_sha256':digest('/tmp/epic-2382-ppdoc-hf-processor-v1.py'),'page_manifest_sha256':digest(ROOT/'bert-page/manifest.json'),'processor_receipt':processor_receipt,'decoded_sha256':hashlib.sha256(decoded.tobytes()).hexdigest(),'wrapper_sha256':digest(wrapper),'launch_manifest_sha256':digest(launch),'kind':args.kind,'checkpoint_revision':PINS[args.kind],'checkpoint_manifest_sha256':digest(checkpoint/'manifest.json'),'upstream_revision':source_manifest['revision'],'upstream_manifest_sha256':digest(ROOT/'upstream-manifest.json'),'source_import_note':'Exact upstream BaseModelConfig AST executed; namespace package setup skips unrelated framework entrypoint; all model and kernel files unmodified.','generator_sha256':digest(__file__),'requested_device':args.device,'resolved_device':str(mx.default_device()),'tf32':'0','input_sha256':digest(input_file),'processed_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'native_sha256':digest(RUNTIME/'libmlx.so'),'binding_sha256':digest(mx.__file__),'module_paths':modules,'loaded_parameters':loaded,'outputs':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'finite':bool(np.isfinite(v).all()),'min':float(v.min()),'max':float(v.max())} for k,v in arrays.items()},'output_sha256':digest(out),'seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
(out.with_suffix('.provenance.json')).write_text(json.dumps(receipt,indent=2)+'\n');print('PASS',out,receipt['seconds'],receipt['output_sha256'],flush=True)
