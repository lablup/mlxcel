"""Synthetic fail-closed checks; all generated artifacts stay in /tmp."""
from pathlib import Path
import copy
import json
import subprocess
import shutil
import tempfile

ANALYZER = Path(__file__).resolve().with_name('analyze_break_even.py')
REPO = ANALYZER.parents[4]
PIN = subprocess.check_output(['bash', str(REPO / 'scripts/ci/mlx_pinned_commit.sh')], text=True).strip()
root=Path(tempfile.mkdtemp(prefix='issue2264-analyzer-'))
d=root/'new';d.mkdir();old=root/'server-prompt-lookup-gb10-2026-10-11';old.mkdir()
prov={'model_config': {'model_type':'qwen3','hidden_size':2048,'num_hidden_layers':28,'quantization':{'bits':4}}, 'purpose':'final-acceptance','measurement_checkout':'a'*40,'binary_source_commit':'b'*40,'binary_sha256':'c'*64,'model_config_sha256':'d'*64, 'native_mlx_pin':PIN, 'binary':'/tmp/synthetic-server', 'model':'/tmp/synthetic-model'}
(d/'environment.json').write_text(json.dumps(prov));(d/'binary-after.sha256').write_text('c'*64+'  server\n')
def row(style,rep,rate,side,closed=False):
    active=side=='on' and not closed
    return dict(style=style,rep=rep,tokens=256,conc=1,decode_tok_s=[rate],draft_n=10 if active else 0,draft_n_accepted=8 if active else 0,draft_rounds=2 if active else 0,draft_kind='prompt-lookup' if active else None)
def write(path,rows): path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
for r in range(1,6):
    for side in ['on','off']:
        write(d/f'a2-{side}-r{r}.txt',[dict(q=q,completion_tokens=256,decode_s=2.55,draft_n=0,draft_n_accepted=0,draft_rounds=0) for q in range(8)])
        for target in [d,old]:
            write(target/f'ab-{side}-r{r}.txt',[row(style,q,160 if side=='on' else 100,side) for style in ['chat','copy'] for q in range(4)])
for r in range(1,4):
    for side in ['on','off']:
        (d/f'c-{side}-r{r}.txt').write_text(''.join(f'Server: http://localhost\nPrompt: style {style}\n 2 2 0 1.0 1.0 1.0 100.0\n' for style in ['plain','copy']))
        write(d/f'd2-{side}-r{r}.txt',[row(style,q,(ratio if side=='on' else 1)*100,side) for ratio in [2.69,2.79,1.42] for style in ['long-copy','long-chat'] for q in range(3)])
        for storage in ['paged','dense']:
            write(d/f'd-{storage}-{side}-r{r}.txt',[row(style,q,210 if side=='on' else 100,side,closed=storage=='paged') for style in ['long-copy','long-chat'] for q in range(3)])

for output in d.glob('*.txt'):
    label=output.stem
    side='on' if '-on-' in label else 'off'
    argv=['/tmp/synthetic-server','-m','/tmp/synthetic-model']
    if side=='on': argv += ['--draft-kind','prompt-lookup']
    if label.startswith('d-'): argv += ['--decode-storage-backend',label.split('-')[1]]
    elif label.startswith('d2-'): argv += ['--decode-storage-backend','paged']
    output.with_suffix('.arm.json').write_text(json.dumps({'binary':'/tmp/synthetic-server','argv':argv}))

def check(label,passed):
    result=subprocess.run(['python3','-I',str(ANALYZER),str(d)],capture_output=True,text=True)
    assert (result.returncode==0)==passed,(label,result.stdout,result.stderr)
    print(label+': PASS')
check('complete passing matrix',True)
original_directory = d
d = root / 'nested' / 'attempt-3'
shutil.copytree(original_directory, d)
check('nested attempt passing matrix', True)
d = original_directory
p=d/'a2-on-r5.txt';saved=p.read_text();p.unlink();check('missing round',False);p.write_text(saved)
p=d/'environment.json';saved=p.read_text();bad=copy.deepcopy(prov);del bad['model_config'];p.write_text(json.dumps(bad));check('missing model',False);bad=copy.deepcopy(prov);bad['model_config']['hidden_size']=4096;p.write_text(json.dumps(bad));check('wrong model',False);p.write_text(saved)
p=d/'d2-on-r1.txt';saved=p.read_text();p.write_text('\n'.join(saved.splitlines()[:-1]));check('missing long-copy cell',False);p.write_text(saved)
p=d/'a2-on-r1.txt';saved=p.read_text();rows=[json.loads(x) for x in saved.splitlines()];rows[0]['completion_tokens']=0;write(p,rows);check('zero tokens',False);p.write_text(saved)
p=old/'ab-off-r1.txt';saved=p.read_text();rows=[json.loads(x) for x in saved.splitlines()];del rows[4]['decode_tok_s'];write(p,rows);check('missing baseline field',False);p.write_text(saved)
p=d/'c-on-r1.txt';saved=p.read_text();p.write_text(saved.replace('2 2 0','2 1 1'));check('failed concurrency request',False);p.write_text(saved)

p=d/'ab-on-r1.arm.json';saved=p.read_text();bad=json.loads(saved);bad['argv']=bad['argv'][:-2];p.write_text(json.dumps(bad));check('wrong on arm argv',False);p.write_text(saved)
p=d/'ab-off-r1.txt';saved=p.read_text();rows=[json.loads(x) for x in saved.splitlines()];rows[0].update(draft_n=7,draft_n_accepted=7,draft_rounds=1,draft_kind='prompt-lookup');write(p,rows);check('off arm reports verification',False);p.write_text(saved)
p=d/'d-paged-on-r1.txt';saved=p.read_text();rows=[json.loads(x) for x in saved.splitlines()];rows[0].update(draft_n=7,draft_n_accepted=7,draft_rounds=1,draft_kind='prompt-lookup');write(p,rows);check('paged 8192 control verified',False);p.write_text(saved)
p=d/'ab-on-r1.txt';saved=p.read_text();rows=[json.loads(x) for x in saved.splitlines()];rows[4]['draft_kind']='mtp';write(p,rows);check('wrong draft kind',False);p.write_text(saved)
p=d/'environment.json';saved=p.read_text();bad=json.loads(saved);bad['native_mlx_pin']='f'*40;p.write_text(json.dumps(bad));check('wrong native MLX pin',False);del bad['native_mlx_pin'];p.write_text(json.dumps(bad));check('missing native MLX pin',False);p.write_text(saved)

for rate in [99.0,98.9999]:
    for r in range(1,6):
        write(d/f'a2-on-r{r}.txt',[dict(q=q,completion_tokens=256,decode_s=255/rate,draft_n=0,draft_n_accepted=0,draft_rounds=0) for q in range(8)])
    check(f'chat threshold rate {rate}',rate==99.0)
print('Synthetic data: '+str(root))
