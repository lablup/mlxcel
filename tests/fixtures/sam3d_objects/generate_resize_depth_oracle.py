"""Reproduce authenticated helper values without loading checkpoint weights."""
from pathlib import Path
import argparse,json,hashlib
parser=argparse.ArgumentParser(description="Reproduce the tiny authenticated depth resize oracle")
parser.add_argument("--source",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--device",choices=("cpu","gpu"),default="cpu")
args=parser.parse_args()

EXECUTED_HELPERS = {'provenance.py': '062ca141346c59988ead974a2426bdc5912b4b17a309536ad76b15f4ff9f2829', 'provenance.json': 'cfccfb008b154edc6a67ec45ed4233de929f7a0ab74f1aafc679d8019fc83641', 'upstream_loader.py': '27f4d0384fcdcce9a2461e30af84cd6e70cf4d4f7a133f35fc96647fab45956f'}
for _name, _expected in EXECUTED_HELPERS.items():
    if hashlib.sha256((Path(__file__).parent / _name).read_bytes()).hexdigest() != _expected:
        raise ValueError(f"Unapproved reproduction helper: {_name}")

import mlx.core as mx
from provenance import verify_source_runtime
from upstream_loader import load_upstream
source=args.source
runtime=verify_source_runtime(source,mx)
load_upstream(source)
from mlx_vlm.models.interpolate import resize_bicubic_nhwc,resize_bilinear_nhwc
mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
x=mx.arange(18,dtype=mx.float32).reshape(1,3,3,2)/17
b=resize_bicubic_nhwc(x,scale_factor=((4+.1)/3,(6+.1)/3))
y=mx.arange(24,dtype=mx.float32).reshape(1,3,4,2)/23
l=resize_bilinear_nhwc(y,(2,3),antialias=False)
mx.eval(b,l)
p=args.output
if p.exists():
    raise ValueError("Refusing to overwrite an existing oracle")
r={'purpose':'authentic tiny helper oracle, no model acceptance','runtime':runtime,'executed_helpers':EXECUTED_HELPERS,'source_revision':'abd3b2d7b728cb5372d4e092260da325e5ded209','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'bicubic':{'input':'arange(18,F32).reshape(1,3,3,2)/17','scale':[(4+.1)/3,(6+.1)/3],'shape':list(b.shape),'values':b.tolist()},'bilinear':{'input':'arange(24,F32).reshape(1,3,4,2)/23','size':[2,3],'antialias':False,'shape':list(l.shape),'values':l.tolist()}}
p.write_text(json.dumps(r,indent=2)+'\n');print(p,hashlib.sha256(p.read_bytes()).hexdigest())
