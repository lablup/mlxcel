import argparse,hashlib,json,sys
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output-stem', type=Path, required=True)
parser.add_argument('--device', choices=['cpu','gpu'], default='gpu')
args=parser.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parent))
from provenance import verify_source_runtime
from upstream_loader import load_upstream
import mlx.core as mx
mx.set_default_device(mx.gpu if args.device=='gpu' else mx.cpu)
source=args.source
runtime=verify_source_runtime(source,mx)
load_upstream(source)
from mlx_vlm.models.interpolate import resize_nearest_nhwc,resize_bilinear_nhwc
x=mx.arange(84,dtype=mx.float32).reshape(1,1,84,1)
y=mx.array([((i*37)%101-50)/17 for i in range(12)],dtype=mx.float32).reshape(1,3,4,1)
arrays={'nearest_input':x,'nearest_output':resize_nearest_nhwc(x,(1,518)),'bilinear_input':y,'bilinear_output':resize_bilinear_nhwc(y,(7,9),antialias=True)}
arrays['bilinear_height_first_output']=resize_bilinear_nhwc(y,(2,3),antialias=True)
arrays['bilinear_tie_output']=resize_bilinear_nhwc(y,(5,12),antialias=True)
mx.eval(*arrays.values())
out=Path(str(args.output_stem)+'.safetensors')
receipt_path=Path(str(args.output_stem)+'.json')
if receipt_path.exists(): raise RuntimeError('receipt exists')
out.parent.mkdir(parents=True,exist_ok=True)
if out.exists(): raise RuntimeError('output exists')
mx.save_safetensors(str(out),arrays)
r={'runtime':runtime,'device':str(mx.default_device()),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'arrays':{k:{'shape':v.shape,'dtype':str(v.dtype),'values':v.reshape(-1).tolist()} for k,v in arrays.items()}}
receipt_path.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='arrays'}))
