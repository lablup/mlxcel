"""Compare authenticated complete Objects outputs without sampling or filtering."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from provenance import canonical, require, sha, validate_result


def load_file(path):
    path=Path(path)
    with path.open('rb') as stream:
        header_size=int.from_bytes(stream.read(8),'little')
        require(2<=header_size<=1024*1024,'Invalid tensor header length')
        header=json.loads(stream.read(header_size))
    start=8+header_size; total=path.stat().st_size-start; intervals=[]; arrays={}
    types={'F32':'<f4','F16':'<f2','BF16':'<u2','I32':'<i4','I64':'<i8','U32':'<u4','U8':'u1','BOOL':'?'}
    for name,entry in header.items():
        if name=='__metadata__': continue
        require(entry['dtype'] in types,'Unsupported tensor dtype')
        shape=entry['shape']; require(all(type(d) is int and d>0 for d in shape),'Invalid tensor shape')
        size=1
        for d in shape: size*=d
        dt=np.dtype(types[entry['dtype']]); lo,hi=entry['data_offsets']
        require(type(lo) is int and type(hi) is int and 0<=lo<hi<=total and hi-lo==size*dt.itemsize,'Invalid tensor storage')
        intervals.append((lo,hi))
        array=np.memmap(path,dtype=dt,mode='r',offset=start+lo,shape=tuple(shape))
        arrays[name]=(array.astype(np.uint32)<<16).view(np.float32) if entry['dtype']=='BF16' else array
    cursor=0
    for lo,hi in sorted(intervals):
        require(lo==cursor,'Tensor gap or overlap'); cursor=hi
    require(cursor==total,'Trailing tensor payload')
    return arrays


def compare(reference,actual):
    require(set(reference)==set(actual),'Output key set mismatch')
    require(np.array_equal(reference['coords'],actual['coords']),'Occupied coordinates differ')
    require(reference['latents'].shape==actual['latents'].shape,'Latent shape mismatch')
    r=reference['latents'].astype(np.float64); a=actual['latents'].astype(np.float64)
    rms=float(np.sqrt(np.mean((a-r)**2))/max(float(np.sqrt(np.mean(r*r))),1e-12))
    rv=reference['mesh.vertices'].astype(np.float64); av=actual['mesh.vertices'].astype(np.float64)
    count_error=abs(len(av)-len(rv))/len(rv)
    scale=float(np.linalg.norm(np.max(rv,axis=0)-np.min(rv,axis=0)))
    require(scale>0 and np.isfinite(scale),'Degenerate reference mesh scale')
    forward=float(np.mean(cKDTree(rv).query(av,workers=1)[0]))
    backward=float(np.mean(cKDTree(av).query(rv,workers=1)[0]))
    normalized=.5*(forward+backward)/scale
    metrics={'latents_relative_rms':rms,'reference_vertices':len(rv),'actual_vertices':len(av),'vertex_count_relative_error':count_error,'actual_to_reference_mean_distance':forward,'reference_to_actual_mean_distance':backward,'reference_aabb_diagonal':scale,'normalized_symmetric_chamfer':normalized}
    failures=[]
    if not rms<.01: failures.append('latents relative RMS >=1e-2')
    if not count_error<=.01: failures.append('mesh vertex count error >1%')
    if not normalized<.001: failures.append('normalized Chamfer >=1e-3')
    return metrics,failures


def main():
    p=argparse.ArgumentParser()
    for n in ('reference','actual','actual-receipt','report'): p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--capture-plan',type=Path,required=True)
    p.add_argument('--capture-plan-sha256',required=True)
    p.add_argument('--validator-manifest',type=Path,required=True)
    p.add_argument('--validator-manifest-sha256',required=True)
    p.add_argument('--reference-receipt-sha256',required=True)
    p.add_argument('--actual-receipt-sha256',required=True)
    a=p.parse_args()
    require(sha(a.validator_manifest)==a.validator_manifest_sha256,'Unapproved validator manifest')
    validators=json.loads(a.validator_manifest.read_text())
    expected_helpers={'compare_reference.py','provenance.py','comparison-contract.json','capture_identity.py','provenance.json'}
    require(set(validators)==expected_helpers,'Validator manifest schema mismatch')
    for name,digest in validators.items(): require(sha(Path(__file__).parent/name)==digest,f'Validator changed: {name}')
    contract=json.loads((Path(__file__).parent/'comparison-contract.json').read_text())
    require(contract['schema_version']==1 and contract['latents']['relative_rms_strict_upper_bound']==.01 and contract['mesh']['vertex_count_relative_error_maximum']==.01 and contract['mesh']['normalized_chamfer_strict_upper_bound']==.001,'Unsupported comparison contract')
    require(sha(a.reference.with_suffix('.json'))==a.reference_receipt_sha256,'Unapproved reference receipt')
    require(sha(a.actual_receipt)==a.actual_receipt_sha256,'Unapproved actual capture receipt')
    ref=json.loads(a.reference.with_suffix('.json').read_text()); actual=json.loads(a.actual_receipt.read_text()); pin=canonical()
    from capture_identity import validate_capture
    attested=validate_capture(a.capture_plan,a.capture_plan_sha256)
    require(actual==attested,'Actual receipt differs from independently verified capture')
    require(a.actual.resolve()==Path(attested['run_dir'])/'output'/'full.safetensors','Actual tensor path is outside the attested capture')
    require(ref['output_sha256']==sha(a.reference),'Reference hash mismatch')
    require(actual['output_sha256']==sha(a.actual),'Actual hash mismatch')
    for key in ('source_revision','source_sha256','checkpoint'): require(ref[key]==pin[key],f'Reference {key} mismatch')
    for key in ('version','native_sha256','binding_sha256'): require(ref['runtime'][key]==pin['runtime'][key],f'Reference runtime {key} mismatch')
    require(ref['tf32']=='0' and ref['device']=='Device(gpu, 0)','Primary reference must use pinned GPU/TF32 policy')
    require(ref['case'] in ('default','rgba-dropout'),'Unknown reference case')
    opts=dict(pin['default_options']); opts['estimate_depth']=ref['case']=='default'
    require(ref['options']==opts,'Reference options mismatch')
    names=['object.png','mask.png'] if ref['case']=='default' else ['object-rgba.png']
    require(ref['input_files']=={n:pin['inputs'][n] for n in names},'Reference input mismatch')
    require(ref.get('execution_api')=='Pinned unchanged Pipeline.generate; only complete event is evaluated by upstream','Reference execution API mismatch')
    require(ref.get('generator_sha256')=='2074c2ccc23cbe3adcee45e523095d539bc4d541d27791f681adf3cd09c76cae','Canonical generator mismatch')
    expected_artifacts={'reference.safetensors','mesh.obj','gaussian.ply'}
    require(set(ref['artifact_sha256'])==expected_artifacts,'Reference artifact set mismatch')
    require(ref['artifact_sha256']['reference.safetensors']==ref['output_sha256'],'Reference output identity differs')
    for name,digest in ref['artifact_sha256'].items():
        require(Path(name).name==name,'Invalid artifact name')
        require(sha(a.reference.parent/name)==digest,f'Reference artifact changed: {name}')
    require(actual['producer']=='mlxcel' and actual['runtime_device']=='GPU' and actual['tf32']=='0','Actual execution identity mismatch')
    for key in ('case','options','input_files','checkpoint'): require(actual[key]==ref[key],f'Actual {key} mismatch')
    for key in ('binary_sha256','source_archive_sha256','invocation_sha256','log_sha256'):
        require(isinstance(actual.get(key),str) and len(actual[key])==64,f'Missing actual capture identity: {key}')
    require(actual['child_exit']==0,'Actual capture did not succeed')
    reference=load_file(str(a.reference)); produced=load_file(str(a.actual))
    for tensors in (reference,produced):
        for name,value in tensors.items():
            require(value.dtype.kind in ('iu' if name in ('coords','mesh.faces') else 'f'),f'Invalid semantic dtype: {name}')
    validate_result(reference,opts['formats'],ref['pointmap_conditioned'])
    validate_result(produced,opts['formats'],actual['pointmap_conditioned'])
    require(ref['pointmap_conditioned']==actual['pointmap_conditioned']==opts['estimate_depth'],'Conditioning route differs')
    metrics,failures=compare(reference,produced)
    report={'reference_sha256':sha(a.reference),'actual_sha256':sha(a.actual),'reference_receipt_sha256':sha(a.reference.with_suffix('.json')),'actual_receipt_sha256':sha(a.actual_receipt),'validator_manifest_sha256':sha(a.validator_manifest),'validator_sha256':validators,'comparator_sha256':sha(Path(__file__)),'metrics':metrics,'failures':failures}
    a.report.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report))
    require(not failures,'Full-array parity failed')

if __name__=='__main__': main()
