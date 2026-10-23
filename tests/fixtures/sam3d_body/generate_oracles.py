# Copyright 2025-2026 Lablup Inc.
# SPDX-License-Identifier: Apache-2.0
"""Small unchanged upstream geometry and OpenCV crop oracles; no model weights."""
from pathlib import Path
import argparse
import hashlib
import json
from types import SimpleNamespace

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--image', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--manifest-sha256', required=True)
a = p.parse_args()
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
if digest(a.manifest) != a.manifest_sha256:
    raise ValueError('Unapproved oracle program')
manifest = json.loads(a.manifest.read_text())
if set(manifest) != {'generate_oracles.py', 'provenance.py', 'provenance.json'}:
    raise ValueError('Oracle helper inventory differs')
for name, value in manifest.items():
    if digest(Path(__file__).parent / name) != value:
        raise ValueError(f'Oracle helper differs: {name}')
from provenance import load_upstream, require, verify_image, verify_source_runtime
import mlx.core as mx
import numpy as np
from safetensors.numpy import save_file
mx.set_default_device(mx.cpu)
runtime = verify_source_runtime(a.source, mx)
load_upstream(a.source)
from pinned_sam3d_body import mhr_utils as u
from pinned_sam3d_body.mhr_body import MHRBodyModel, _scatter_add_1d
from pinned_sam3d_body.batch_prep import get_affine_transform, apply_affine_transform, prepare_image, get_cliff_condition
require(not a.output.exists(), 'Refuse to replace oracles')
out = {}
rot = mx.array([[1,0,0,0,1,0],[1,2,3,4,-2,1],[1,0,0,1,1e-7,0],[0,0,0,0,0,0]], mx.float32)
out['rot6d_input'] = rot
out['rot6d_matrix'] = u.rot6d_to_rotmat(rot)
out['euler_zyx'] = u.rotmat_to_euler_ZYX(out['rot6d_matrix'])
out['euler_xyz'] = u.batch_xyz_from_6d(rot)
q = mx.array([[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[.2,-.3,.1,.92736185]], mx.float32)
out['quaternions'] = q
out['quaternion_matrix'] = u.quat_to_rotmat(q)
out['matrix_quaternion'] = u.rotmat_to_quat(out['quaternion_matrix'])
body = mx.array(np.sin(np.arange(520,dtype=np.float32)*.13).reshape(2,260))
hand = mx.array(np.cos(np.arange(108,dtype=np.float32)*.17).reshape(2,54))
out['body_cont'] = body
out['body_params'] = u.compact_cont_to_model_params_body(body)
out['hand_cont'] = hand
out['hand_params'] = u.compact_cont_to_model_params_hand(hand)
offsets = mx.array([[1,2,3],[-2,.5,1]], mx.float32)
prerot = mx.array([[0,0,.38268343,.9238795],[.25881904,0,0,.9659258]], mx.float32)
parents = mx.array([-1,0], mx.int32)
dofs = mx.array([[.2,-.3,.4,.1,.2,-.3,.5, -.1,.2,.3,-.2,.1,.4,-.25],[-.4,.3,.1,-.3,.4,.2,-.5,.3,-.2,.4,.2,-.3,.1,.75]], mx.float32)
receiver = SimpleNamespace(num_joints=2,joint_translation_offsets=offsets,joint_prerotations=prerot,joint_parents=parents)
state,pos,rotation,scale = MHRBodyModel._forward_kinematics(receiver,dofs)
for name,value in [('fk_dofs',dofs),('fk_offsets',offsets),('fk_prerotations',prerot),('fk_parents',parents),('fk_state',state),('fk_global_pos',pos),('fk_global_rot',rotation),('fk_global_scale',scale)]: out[name]=value
values=mx.array([1e8,1,-1e8,3,-2],mx.float32);indices=mx.array([0,0,0,1,1],mx.int32)
out['scatter_values']=values;out['scatter_indices']=indices;out['scatter_result']=_scatter_add_1d(values,indices,3)
mx.eval(*out.values())
arrays={k:np.asarray(v) for k,v in out.items()}
require(all(np.isfinite(v).all() for v in arrays.values()),'Nonfinite small geometry oracle')
a.output.mkdir(parents=True)
save_file(arrays,str(a.output/'geometry.safetensors'))
image=verify_image(a.image)
synthetic=((np.arange(9*13*3,dtype=np.uint32)*37+11)%256).astype(np.uint8).reshape(9,13,3)
crop_arrays={};cases=[('tiny_full',synthetic,[0,0,13,9],(7,5)),('tiny_partial',synthetic,[-3,-2,10,8],(7,5)),('dancing',image,[732,114,1523,1451],(512,512)),('dancing_full',image,[0,0,2250,1500],(512,512))]
for name,img,bbox,size in cases:
 box=np.array(bbox,np.float32);center=np.array([(box[0]+box[2])/2.,(box[1]+box[3])/2.]);scale=np.array([box[2]-box[0],box[3]-box[1]])
 matrix=get_affine_transform(center,scale,(size[1],size[0]));cropped=apply_affine_transform(img,matrix,(size[1],size[0]));focal=np.float32((img.shape[0]**2+img.shape[1]**2)**.5).item()
 crop_arrays.update({name+'.input':img,name+'.bbox':box,name+'.affine':matrix,name+'.crop':cropped,name+'.normalized':prepare_image(img,bbox,size),name+'.cliff':get_cliff_condition(bbox,img.shape[:2],focal)})
save_file(crop_arrays,str(a.output/'crop.safetensors'))
receipt={'purpose':'tiny geometry and affine crop literals; no full model acceptance','runtime':runtime,'helpers':manifest,'geometry_sha256':digest(a.output/'geometry.safetensors'),'crop_sha256':digest(a.output/'crop.safetensors'),'geometry_shapes':{k:list(v.shape) for k,v in arrays.items()},'crop_shapes':{k:list(v.shape) for k,v in crop_arrays.items()},'fk_receiver':'Two-joint SimpleNamespace supplies only the four fields read by unchanged upstream method; no model constructor or weights.'}
(a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
