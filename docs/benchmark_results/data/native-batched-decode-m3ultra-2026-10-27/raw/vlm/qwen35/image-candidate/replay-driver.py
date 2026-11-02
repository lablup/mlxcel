import argparse,datetime,hashlib,json,math,os,pathlib,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--binary',required=True);p.add_argument('--source',required=True);p.add_argument('--loaded-metallib',required=True);a=p.parse_args()
root=pathlib.Path('/tmp/epic-2574-measurements');out=root/'candidate-qwen35-vlm-image';out.mkdir(exist_ok=False)
repo=pathlib.Path('/Users/devops/Development/mlxcel');model=pathlib.Path('/Users/devops/.cache/mlxcel/models/mlx-community/Qwen3.5-4B-4bit');image=repo/'tests/fixtures/test_image.png';corpus=repo/'docs/architecture.md'
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
meta={'source_revision':a.source,'binary':{'path':a.binary,'sha256':digest(a.binary)},'loaded_metallib':{'path':a.loaded_metallib,'sha256':digest(a.loaded_metallib)},'image_sha256':digest(image),'config_sha256':digest(model/'config.json'),'corpus_sha256':digest(corpus),'driver_sha256':digest(__file__),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
for rows in (1,4):
 overrides={'MLXCEL_TRACE_BATCH_ROWS':str(rows),'MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE':'0','MLXCEL_TRACE_IMAGE':str(image),'MLXCEL_TRACE_IMAGE_PROMPT':'Describe this image briefly.'}
 env=os.environ.copy();env.update(overrides);cmd=[a.binary,str(model),str(corpus),'64','1','8'];start=time.monotonic()
 with (out/f'trace-b{rows}.out').open('w') as stdout,(out/f'trace-b{rows}.err').open('w') as stderr:r=subprocess.run(cmd,env=env,stdout=stdout,stderr=stderr,timeout=600)
 (out/f'trace-b{rows}.status.json').write_text(json.dumps({'command':cmd,'env':overrides,'returncode':r.returncode,'seconds':time.monotonic()-start,'source_revision':a.source},indent=2)+'\n');r.check_returncode()
 data=[line.split('\t') for line in (out/f'trace-b{rows}.out').read_text().splitlines() if line and not line.startswith('#')];assert len(data)==64 and all(math.isfinite(float(row[3])) for row in data)
for label,left,right in [('baseline-v-b1',root/'baseline-qwen35-vlm-image/trace-b1.out',out/'trace-b1.out'),('b1-v-b4',out/'trace-b1.out',out/'trace-b4.out')]:
 r=subprocess.run(['python3',str(repo/'scripts/compare_logit_traces.py'),str(left),str(right)],capture_output=True,text=True);(out/(label+'.txt')).write_text(r.stdout+r.stderr);print(label,r.returncode,r.stdout,flush=True)
