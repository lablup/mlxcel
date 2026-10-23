import hashlib, importlib.util, json, pathlib, types
import numpy as np
import torch
from safetensors.numpy import load_file
from transformers import PPDocLayoutV3ImageProcessor
R=pathlib.Path('/home/devops/Development/wt-epic-2382-issue-2412/target/epic-2382-ppdoc-reference')
C=pathlib.Path('/home/devops/Development/mlxcel/.cache/epic-2382/preflight-ppdoc-checkpoints/paddle/e4f489dc536556fc1be02b973ba06756e0d4a2ac')
def require(ok,message):
 if not ok:raise RuntimeError(message)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
manifest=R/'upstream-manifest.json'
require(sha(manifest)=='ef120ba06b56e27f54accc9ccd1710fe16b62c177b40c257d16495dce2c9059c','source manifest')
source=R/'upstream/mlx_vlm/models/falcon_ocr/layout.py'
entry=next(x for x in json.loads(manifest.read_text())['files'] if x['path']==str(source))
require(sha(source)==entry['sha256'],'Falcon source')
spec=importlib.util.spec_from_file_location('pinned_falcon_layout',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
require(sha(C/'config.json')=='3cf834b91d23a756b1519bce4db42c09e852f3e35c35092dd5a3e253a50c071a','config')
require(sha(C/'preprocessor_config.json')=='519fe0187a43a1ca429e3ad8317bab8700f0d5e8fb3a6e3a0a413ffac078ba42','processor config')
import inspect
require(sha(inspect.getfile(PPDocLayoutV3ImageProcessor))=='5b064fa7383dda12b3550448eae77d4f627a102c25e8b4db25d99e85fba4abc6','processor source')
processor=PPDocLayoutV3ImageProcessor.from_pretrained(str(C),local_files_only=True)
labels={int(k):v for k,v in json.loads((C/'config.json').read_text())['id2label'].items()}
proxy=types.SimpleNamespace(_processor=processor,_id2label=labels)
all_results={}
for kind in ['hashnuke','paddle']:
 path=R/f'raw-references/{kind}_gpu_raw.safetensors';require(sha(path)=='170221e4b16da6abb1d5e2676c3906da61e7a5a61af362bd0b43869b0b24d7d1','raw identity');a=load_file(path)
 provenance=path.with_suffix('.provenance.json');require(sha(provenance)=={'hashnuke':'33b26e752ad25856fa348537dcff8bd25ff894999ebf34a6ce2fdde2f6027e81','paddle':'f747114440d0c5377c0394de8300d9b17c8bc5bf0b26c007740c21925360f508'}[kind],'raw provenance')
 require(set(a)=={'logits','pred_boxes','order_logits'},'output keys')
 out=types.SimpleNamespace(**{k:torch.from_numpy(v) for k,v in a.items()})
 original=mod.LayoutDetector._post_process(proxy,out,torch.tensor([[1684,1191]]),0.3)[0]
 scores,indices=out.logits.sigmoid().flatten(1).topk(300,dim=-1)
 queries=indices//25;classes=indices%25;ranks=processor._get_order_seqs(out.order_logits).gather(1,queries)
 valid=scores[0]>=0.3;order=torch.argsort(ranks[0][valid],stable=True)
 kept=[{'query':int(q),'label':int(c),'query_rank':int(r)} for q,c,r in zip(queries[0][valid][order],classes[0][valid][order],ranks[0][valid][order])]
 require(len(kept)==len(original),'retained count')
 detections=[dict(det,**meta,reading_order=i+1) for i,(det,meta) in enumerate(zip(original,kept))]
 for d in detections:require(d['category']==labels[d['label']],'class mapping')
 centers=out.pred_boxes[0,:,:2];dims=out.pred_boxes[0,:,2:];xyxy=torch.cat([centers-0.5*dims,centers+0.5*dims],dim=-1)*torch.tensor([1191,1684,1191,1684],dtype=torch.float32)
 rust_expected=[{'bbox':xyxy[int(q)].tolist(),'score':float(s),'label':int(c),'class_name':labels[int(c)],'query':int(q),'query_rank':int(r),'reading_order':i+1} for i,(s,q,c,r) in enumerate(zip(scores[0][valid][order],queries[0][valid][order],classes[0][valid][order],ranks[0][valid][order]))]
 geometric=sorted(detections,key=lambda d:(d['bbox'][1],d['bbox'][0],-d['score'],d['label']))
 all_results[kind]={'raw_sha256':sha(path),'upstream_falcon_detections':original,'detections':detections,'rust_expected_detections':rust_expected,'geometric_reading_orders':[d['reading_order'] for d in geometric],'order_differs':geometric!=detections}
 print(kind,json.dumps(all_results[kind],indent=2),flush=True)
out=R/'expected_detections.json';require(not out.exists(),'existing results');out.write_text(json.dumps({'threshold':0.3,'size':[1191,1684],'script_sha256':sha(__file__),'falcon_source_sha256':sha(source),'results':all_results},indent=2)+'\n')
