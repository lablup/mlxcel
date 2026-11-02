import pathlib,subprocess,json,os,time
root=pathlib.Path('/tmp/epic-2574-measurements');source=root/'candidate-ring-mini';out=root/'candidate-ring-mini-unfused-diagnostic';out.mkdir(exist_ok=False)
for rows in [1,4]:
 old=json.loads((source/f'trace-b{rows}.status.json').read_text());cmd=old['command'];env=os.environ.copy();env.update({'MLXCEL_TRACE_BATCH_ROWS':str(rows),'MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE':'0','MLXCEL_FUSED_MOE':'0'});start=time.monotonic()
 with (out/f'trace-b{rows}.out').open('w') as stdout,(out/f'trace-b{rows}.err').open('w') as stderr:r=subprocess.run(cmd,env=env,stdout=stdout,stderr=stderr,timeout=600)
 (out/f'trace-b{rows}.status.json').write_text(json.dumps({'command':cmd,'env':{k:env[k] for k in ['MLXCEL_TRACE_BATCH_ROWS','MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE','MLXCEL_FUSED_MOE']},'returncode':r.returncode,'seconds':time.monotonic()-start,'source_revision':'413e4649ebe9451fc73a7f4d03c492807a4e9148','note':'Diagnostic only, not production performance setting.'},indent=2)+'\n');r.check_returncode()
r=subprocess.run(['python3','/Users/devops/Development/mlxcel/scripts/compare_logit_traces.py',str(out/'trace-b1.out'),str(out/'trace-b4.out')],capture_output=True,text=True);(out/'b1-v-b4.txt').write_text(r.stdout+r.stderr);print(r.stdout)
