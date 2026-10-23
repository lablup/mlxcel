import resource, subprocess, sys, time
from pathlib import Path
start=time.monotonic()
script=Path(__file__).with_name('generate_reference.py')
result=subprocess.run([sys.executable,str(script)])
print('RESOURCE elapsed_seconds',time.monotonic()-start,'peak_child_rss_kib',resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,flush=True)
sys.exit(result.returncode)
