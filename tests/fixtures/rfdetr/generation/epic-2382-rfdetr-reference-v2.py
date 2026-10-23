"""Authentic RF-DETR public predictor capture; model sources remain unchanged."""
import argparse, ast, dataclasses, hashlib, inspect, json, os, pathlib, resource, sys, time, types
P = pathlib.Path
ROOT = P('/home/devops/Development/wt-epic-2382-issue-2431/target/epic-2382-rfdetr-reference')
RUNTIME = P('/home/devops/Development/mlxcel/.cache/epic-2382/python-exact-core-layernorm-v1')
IMAGE = ROOT.parents[1] / 'tests/fixtures/sam3/cats.png'
PINS = {'base': ('mlx-community/rfdetr-base-fp32', 'eddb85a12c5996074fac2f2e769ca5e947bd80d3', 487, 498, 560, 300), 'seg': ('mlx-community/rfdetr-seg-small-fp32', 'a7a2375b4ca7b2ef0427af91b55d8176ca647b35', 544, 559, 384, 100)}
def require(ok, message):
    if not ok: raise RuntimeError(message)
def digest(path):
    h = hashlib.sha256()
    with P(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()
def authenticated(path, value):
    require(digest(path) == value, 'identity changed: '+str(path))
    return json.loads(P(path).read_text())
def namespace(name, path):
    module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module
parser=argparse.ArgumentParser();parser.add_argument('kind',choices=PINS);parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
require(os.environ.get('MLX_ENABLE_TF32') == '0', 'TF32 must be disabled')
source=authenticated(ROOT/'upstream-manifest.json','d9853592f09e0d4faca0b976a75dfa40959a75328d9890ed49f23dc0da948c71')
require(source['revision']=='4319ef2c837b7226b758c3dd69edda51b7079785','source revision')
for name, value in source['files'].items():
    path=ROOT/'upstream'/name;require(path.resolve().is_relative_to((ROOT/'upstream').resolve()),'source path');require(digest(path)==value,'source changed '+name)
require({str(p.relative_to(ROOT/'upstream')) for p in (ROOT/'upstream').rglob('*') if p.is_file()} == set(source['files']),'source inventory')
bundles=authenticated(ROOT/'checkpoint-download.json','feb9cdc21fc655afdccb957fa1dcce519aea122a4c37f99fab23b47495838b46')
repo, revision, checkpoint_count, parameter_count, resolution, queries = PINS[args.kind]
bundle=next(b for b in bundles['bundles'] if b['repository']==repo);require(bundle['revision']==revision,'checkpoint revision');checkpoint=P(bundle['canonical_path'])
for entry in bundle['files']:
    path=checkpoint/entry['path'];require(path.resolve().is_relative_to(checkpoint.resolve()),'checkpoint path');require(path.stat().st_size==entry['bytes'] and digest(path)==entry['sha256'],'checkpoint file '+entry['path'])
identity=authenticated(RUNTIME/'identity.json','a34c99a7509385a0bc7f45081f7d0b215fae6ac65f738005ae6d469ef086a793')
require(digest(RUNTIME/'libmlx.so')=='c1d8203b257364271e1ea6298705dd3bb0c82a7f06634f5578a209883601a9fc','native runtime')
binding=RUNTIME/'python/mlx/core.cpython-312-x86_64-linux-gnu.so'
require(digest(binding)=='f8ec0f363bc6eab1cda644a2579ae43aa663939f30ace531d5fb8cc33c7090cc','binding runtime')
require(digest(IMAGE)=='2a7ba590d525fe31f5627c73b2e83a5b63b3cba3840fb10ae9d726187aca6769','fixed committed image')
wrapper=P('/tmp/epic-2382-rfdetr-reference-v2.sh');launch=P('/tmp/epic-2382-rfdetr-reference-v2-launch.sha256')
entries={}
for line in launch.read_text().splitlines():
    value,path=line.split(maxsplit=1);require(path not in entries,'duplicate launch entry');entries[path]=value
require(set(entries)=={str(wrapper),str(P(__file__))},'launch inventory')
for path,value in entries.items():require(digest(path)==value,'launch artifact changed')
import numpy as np
import PIL, cv2, transformers, safetensors
from PIL import Image
import mlx.core as mx
from mlx.utils import tree_flatten
from safetensors.numpy import save_file
require(P(mx.__file__).resolve()==binding.resolve(),'actual binding path')
require({line.split()[-1] for line in P('/proc/self/maps').read_text().splitlines() if 'libmlx.so' in line}=={str(RUNTIME/'libmlx.so')},'actual mapped native')
mx.set_default_device(mx.cpu if args.preflight else mx.gpu)
source_root=ROOT/'upstream/mlx_vlm'
namespace('mlx_vlm',source_root);namespace('mlx_vlm.models',source_root/'models')
# Execute exact upstream definitions, including its actual Transformers registration.
base_path=source_root/'models/base.py';tree=ast.parse(base_path.read_text());names={'BaseModelConfig','check_array_shape','install_auto_processor_patch'}
nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
require({n.name for n in nodes}==names,'base definition closure')
base=types.ModuleType('mlx_vlm.models.base');base.__dict__.update({'dataclass':dataclasses.dataclass,'inspect':inspect});exec(compile(ast.Module(body=nodes,type_ignores=[]),str(base_path),'exec'),base.__dict__);sys.modules[base.__name__]=base
from mlx_vlm.models.rfdetr.config import ModelConfig
from mlx_vlm.models.rfdetr.rfdetr import Model
from mlx_vlm.models.rfdetr.processing_rfdetr import RFDETRProcessor
from mlx_vlm.models.rfdetr.generate import RFDETRPredictor, postprocess
config=ModelConfig.from_dict(json.loads((checkpoint/'config.json').read_text()));processor=RFDETRProcessor.from_pretrained(checkpoint)
image=Image.open(IMAGE).convert('RGB');processed=processor.preprocess_image(image)
x=processed['pixel_values'];require(x.shape==(1,resolution,resolution,3) and x.dtype==np.float32 and np.isfinite(x).all(),'actual processor tensor')
require(processed['original_size']==(image.height,image.width) and image.width!=image.height,'non-square image')
modules={m.__name__:{'version':getattr(m,'__version__','unknown'),'path':str(m.__file__),'sha256':digest(m.__file__)} for m in [np,PIL,cv2,transformers,safetensors]}
cv_native=next(P(cv2.__file__).parent.glob('cv2*.so'));modules['cv2']['native_sha256']=digest(cv_native)
common={'kind':args.kind,'checkpoint':bundle,'source_manifest_sha256':digest(ROOT/'upstream-manifest.json'),'runtime_identity_sha256':digest(RUNTIME/'identity.json'),'native_sha256':digest(RUNTIME/'libmlx.so'),'binding_sha256':digest(binding),'python_path':sys.executable,'python_sha256':digest(sys.executable),'dependencies':modules,'image_sha256':digest(IMAGE),'image_size':[image.height,image.width],'processed_shape':list(x.shape),'processed_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'generator_sha256':digest(__file__),'wrapper_sha256':digest(wrapper),'launch_sha256':digest(launch),'tf32':'0','upstream_revision':source['revision'],'trust_scope':'Existing task Python environment and staged read-only dependencies; exact source definitions selected from base.py avoid unrelated cache imports; no model function replaced.'}
if args.preflight:
    out=ROOT/f'{args.kind}-preflight-v2.json';require(not out.exists(),'preflight output exists');out.write_text(json.dumps(common,indent=2)+'\n');print('PREFLIGHT PASS',args.kind,x.shape,modules,flush=True);sys.exit(0)
output=ROOT/'references-v2'/args.kind;require(not output.exists(),'reference output exists')
require(mx.default_device()==mx.Device(mx.gpu,0),'actual device')
model=Model(config);original_weights=mx.load(str(checkpoint/'model.safetensors'));require(len(original_weights)==checkpoint_count,'checkpoint tensor count');weights=Model.sanitize(original_weights);require(len(weights)==parameter_count,'sanitized parameter count');model.load_weights(list(weights.items()),strict=True);model.eval()
loaded={name:{'shape':list(value.shape),'dtype':str(value.dtype)} for name,value in tree_flatten(model.parameters())}
require(len(loaded)==parameter_count and all(v['dtype']=='mlx.core.float32' for v in loaded.values()),'actual loaded F32 parameters')
mx.eval(model.parameters());print('LOADED',args.kind,len(loaded),mx.default_device(),'F32',flush=True)
observed={}
def observe(frame,event,arg):
    if event=='return' and frame.f_code is Model.__call__.__code__ and frame.f_locals.get('self') is model:observed['raw']=arg
    if event=='return' and frame.f_code is RFDETRProcessor.preprocess_image.__code__ and frame.f_locals.get('self') is processor:observed['processed']=arg
    if event=='return' and frame.f_code is postprocess.__code__:
        observed['query_indices']=frame.f_locals['final_query_idx'].copy()
        observed['postprocess_inputs']={k:frame.f_locals[k].copy() for k in ['pred_logits','pred_boxes']+(['pred_masks'] if config.segmentation else [])}
predictor=RFDETRPredictor(model,processor,score_threshold=0.5,nms_threshold=0.5)
start=time.monotonic();sys.setprofile(observe)
try: result=predictor.predict(image)
finally: sys.setprofile(None)
require(set(observed)=={'raw','processed','query_indices','postprocess_inputs'},'observer inventory')
require(observed['processed']['pixel_values'].tobytes()==x.tobytes(),'processor used exact input')
keys={'pred_logits','pred_boxes'}|({'pred_masks'} if config.segmentation else set());require(set(observed['raw'])==keys,'exact output keys')
raw={k:np.array(mx.contiguous(v)) for k,v in observed['raw'].items()}
shapes={'pred_logits':(1,queries,91),'pred_boxes':(1,queries,4)}
if config.segmentation:shapes['pred_masks']=(1,queries,resolution//4,resolution//4)
for name,value in raw.items():
    require(value.shape==shapes[name] and value.dtype==np.float32 and np.isfinite(value).all(),'raw tensor '+name)
    require(value.tobytes()==observed['postprocess_inputs'][name].tobytes(),'public predictor consumed same raw '+name)
n=len(result.scores);require(n>0 and result.boxes.shape==(n,4) and result.labels.shape==(n,) and len(result.class_names)==n,'nonempty complete detections')
require(np.isfinite(result.boxes).all() and np.isfinite(result.scores).all() and np.all(result.scores>0.5),'finite thresholded detections')
require(len(observed['query_indices'])==n,'query mapping')
if config.segmentation:
    require(result.masks.shape==(n,image.height,image.width) and result.masks.dtype==np.uint8 and set(np.unique(result.masks)).issubset({0,1}),'source resolution binary masks')
    require(all(0<int(m.sum())<m.size for m in result.masks),'nonempty masks')
else:require(result.masks is None,'base masks absent')
output.mkdir(parents=True);save_file({'pixel_values':x},output/'input.safetensors');save_file(raw,output/'raw.safetensors')
records=[]
for i in range(n):
    entry={'query_index':int(observed['query_indices'][i]),'label':int(result.labels[i]),'class_name':result.class_names[i],'score':float(result.scores[i]),'box':[float(v) for v in result.boxes[i]]}
    if config.segmentation:
        name=f'mask-{i:03}.png';Image.fromarray(result.masks[i]*255).save(output/name);entry.update(mask=name,mask_sha256=digest(output/name),foreground_pixels=int(result.masks[i].sum()))
    records.append(entry)
(output/'detections.json').write_text(json.dumps({'threshold':0.5,'nms_iou':0.5,'num_select':processor.num_select,'source_size':[image.height,image.width],'detections':records},indent=2)+'\n')
common.update(actual_device=str(mx.default_device()),loaded_parameters=loaded,checkpoint_tensor_count=checkpoint_count,parameter_count=len(loaded),outputs={k:{'shape':list(v.shape),'dtype':str(v.dtype),'finite':bool(np.isfinite(v).all())} for k,v in raw.items()},capture_note='Unchanged public RFDETRPredictor.predict with read-only CPython return observers; raw logical bytes exactly equal actual postprocess inputs.',seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,files={p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()})
(output/'receipt.json').write_text(json.dumps(common,indent=2)+'\n');print('PASS',args.kind,n,'detections',common['seconds'],'seconds',flush=True)
