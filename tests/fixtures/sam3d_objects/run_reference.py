"""Freeze the reference program and monitor one full, unchanged model run."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def memory():
    return {k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if k in ('MemAvailable','SwapFree','MemTotal')}


def main():
    p=argparse.ArgumentParser()
    for name in ('fixtures','source','bundle','input-dir','run-dir','python','reviewed-manifest'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reviewed-manifest-sha256',required=True)
    p.add_argument('--case',choices=('default','rgba-dropout'),default='default')
    p.add_argument('--reserve-gib',type=float,default=3.0)
    a=p.parse_args()
    if not 1 <= a.reserve_gib <= 8: raise ValueError('Invalid host reserve')
    a.run_dir.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); stopped=False; reason=None; child=None; phase='freeze program'; low_samples=0
    try:
        frozen=a.run_dir/'frozen'; frozen.mkdir()
        phase='authenticate frozen program'
        names=('generate_reference.py','provenance.py','upstream_loader.py','provenance.json','input-provenance.json')
        for name in names: shutil.copyfile(a.fixtures/name,frozen/name)
        shutil.copyfile(Path(__file__),frozen/'run_reference.py')
        hashes={x.name:digest(x) for x in frozen.iterdir()}
        if digest(a.reviewed_manifest)!=a.reviewed_manifest_sha256: raise ValueError('Reviewed manifest hash mismatch')
        reviewed=json.loads(a.reviewed_manifest.read_text())
        if set(reviewed)!=set(names): raise ValueError('Reviewed manifest key set mismatch')
        if any(hashes[n]!=reviewed[n] for n in names): raise ValueError('Frozen program differs from reviewed manifest')
        shutil.copyfile(a.reviewed_manifest,frozen/'reviewed-manifest.json')
        command=[str(a.python),str(frozen/'generate_reference.py'),'--source',str(a.source),'--bundle',str(a.bundle),'--input-dir',str(a.input_dir),'--output',str(a.run_dir/'output'),'--case',a.case,'--device','gpu']
        invocation={'command':command,'frozen_sha256':hashes,'reviewed_manifest_sha256':digest(frozen/'reviewed-manifest.json'),'environment':{k:os.environ.get(k) for k in ('PYTHONPATH','LD_LIBRARY_PATH','MLX_ENABLE_TF32','PYTHONDONTWRITEBYTECODE')},'start_memory':memory(),'reserve_bytes':int(a.reserve_gib*2**30),'global_swap_pressure_available_bytes':8*2**30,'global_swap_loss_bytes':256*2**20,'child_swap_bytes':64*2**20}
        (a.run_dir/'invocation.json').write_text(json.dumps(invocation,indent=2)+'\n')
        if invocation['start_memory']['MemAvailable'] <= invocation['reserve_bytes']: raise RuntimeError('Insufficient available host memory')
        with (a.run_dir/'child.log').open('w') as log, (a.run_dir/'memory.jsonl').open('w') as metrics:
            phase='child execution'
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            (a.run_dir/'child.pid').write_text(str(child.pid)+'\n')
            while child.poll() is None:
                sample={'elapsed_seconds':time.monotonic()-started,**memory()}
                try:
                    status=Path(f'/proc/{child.pid}/status').read_text().splitlines()
                    sample.update({line.split(':')[0]:int(line.split()[1])*1024 for line in status if line.startswith(('VmRSS:','VmSwap:','VmHWM:'))})
                except FileNotFoundError: pass
                metrics.write(json.dumps(sample)+'\n'); metrics.flush()
                low_samples=low_samples+1 if sample['MemAvailable'] < invocation['reserve_bytes'] else 0
                swap_loss=invocation['start_memory']['SwapFree']-sample['SwapFree']
                reason=('two consecutive host MemAvailable samples below recorded reserve' if low_samples>=2 else 'SwapFree loss exceeds 256 MiB with host MemAvailable below 8 GiB' if swap_loss>invocation['global_swap_loss_bytes'] and sample['MemAvailable']<invocation['global_swap_pressure_available_bytes'] else 'child VmSwap exceeds 64 MiB' if sample.get('VmSwap',0)>64*2**20 else None)
                if reason is not None:
                    stopped=True; os.killpg(child.pid,signal.SIGTERM)
                    try: child.wait(timeout=5)
                    except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL)
                    break
                time.sleep(.1)
            code=child.wait()
        (a.run_dir/'child.exit').write_text(str(code)+'\n')
        record={'child_exit':code,'stopped_for_memory':stopped,'stop_reason':reason,'elapsed_seconds':time.monotonic()-started,'end_memory':memory(),'invocation_sha256':digest(a.run_dir/'invocation.json'),'child_log_sha256':digest(a.run_dir/'child.log'),'monitor_sha256':hashes['run_reference.py']}
        (a.run_dir/'resources.json').write_text(json.dumps(record,indent=2)+'\n')
        (a.run_dir/'wrapper.exit').write_text(('0' if code==0 and not stopped else '1')+'\n')
        print(json.dumps(record),flush=True)
        return 0 if code==0 and not stopped else 1
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL); child.wait()
        (a.run_dir/'wrapper.exit').write_text('1\n')
        failure={'phase':phase,'child_started':child is not None,'child_exit':None if child is None else child.returncode,'wrapper_exit':1,'error':str(error),'elapsed_seconds':time.monotonic()-started}
        (a.run_dir/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
        if child is not None: (a.run_dir/'child.exit').write_text(str(child.returncode)+'\n')
        raise

if __name__=='__main__': raise SystemExit(main())
