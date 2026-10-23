#!/usr/bin/env python3
"""Download revision-pinned real checkpoints, DAVIS frames, and reference source."""
from pathlib import Path
import urllib.request,concurrent.futures,json,hashlib
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--cache',type=Path,required=True)
parser.add_argument('--all-checkpoints',action='store_true')
args=parser.parse_args()
root=args.cache
root.mkdir(parents=True,exist_ok=True)
sha='abd3b2d7b728cb5372d4e092260da325e5ded209';jobs=[]
for model,rev,name in [('Video-Depth-Anything-Small-MLX','a12a442242291eed6b885b196cbbdb51f3bcdb87','relative'),('Metric-Video-Depth-Anything-Small-MLX','0d4ff7a5907b62654d4115d83ce3b2af317649e2','metric')]:
 for f in ['config.json','preprocessor_config.json','model.safetensors']:
  jobs.append((f'https://huggingface.co/mlx-community/{model}/resolve/{rev}/{f}',root/'checkpoints'/name/f))
for i in range(64):jobs.append((f'https://huggingface.co/datasets/AlonzoLeeeooo/DAVIS-Edit/resolve/3c33c3245ce366cf3f533d35f4a3f1c295ab889c/JPEGImages/car-turn/{i:05}.jpg',root/'frames'/f'{i:05}.jpg'))
for f in ['base.py','interpolate.py','kernels.py','dinov2/config.py','dinov2/dinov2.py']+[f'video_depth_anything/{name}.py' for name in ['__init__','config','dpt','generate','motion','processing_video_depth_anything','video_depth_anything']]:jobs.append((f'https://raw.githubusercontent.com/Blaizzy/mlx-vlm/{sha}/mlx_vlm/models/{f}',root/'upstream/mlx_vlm/models'/f))
expected={item['path']:item for item in json.loads((Path(__file__).parent/'source-manifest.json').read_text())}
if args.all_checkpoints:
 matrix=json.loads((Path(__file__).parent/'checkpoint-matrix-manifest.json').read_text())['files']
 existing={str(path.relative_to(root)) for _,path in jobs}
 for item in matrix:
  if item['path'] not in existing:jobs.append((item['url'],root/item['path']))
  expected[item['path']]={k:item[k] for k in ('url','path','bytes','sha256')}
def fetch(job):
 url,p=job;p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():
  with urllib.request.urlopen(url,timeout=120) as src,p.open('wb') as dst:
   while chunk:=src.read(1024*1024):dst.write(chunk)
 record={'url':url,'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 if record != expected[record['path']]:
  raise RuntimeError(f'Cached/downloaded asset checksum mismatch: {p}')
 return record
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 records=list(pool.map(fetch,jobs))
(root/'download-manifest.json').write_text(json.dumps(records,indent=2)+'\n');print('downloaded',len(records),'files',sum(r['bytes'] for r in records),'bytes',flush=True)
