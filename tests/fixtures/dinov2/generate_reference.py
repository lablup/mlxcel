#!/usr/bin/env python3
"""Real checkpoint references; see README.md for versions and provenance."""
from pathlib import Path
import argparse
import torch
import transformers
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import AutoModel

parser = argparse.ArgumentParser(description='Regenerate pinned public DINOv2 CPU f32 reference tensors.')
parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent)
parser.add_argument('--cache-dir', type=Path, default=None)
args = parser.parse_args()
out = args.output_dir
out.mkdir(exist_ok=True)
torch.set_num_threads(4)
x = (torch.arange(224 * 224 * 3, dtype=torch.float32).reshape(1,224,224,3) % 256) / 255
mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1)
std = torch.tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1)
normalized = (x.permute(0,3,1,2) - mean) / std
for repo, name, registers, sha in [
    ('facebook/dinov2-small', 'small_reference', 0, 'ed25f3a31f01632728cabb09d1542f84ab7b0056'),
    ('facebook/dinov2-with-registers-small', 'register_small_reference', 4, '0d9846e56b43a21fa46d7f3f5070f0506a5795a9'),
]:
    checkpoint = snapshot_download(repo, revision=sha, allow_patterns=['config.json','model.safetensors','preprocessor_config.json'], cache_dir=args.cache_dir)
    print(repo, sha, checkpoint, flush=True)
    model = AutoModel.from_pretrained(checkpoint, torch_dtype=torch.float32, attn_implementation='eager').eval()
    with torch.inference_mode():
        result = model(pixel_values=normalized)
    tensors = {'pixel_values_nhwc': x.contiguous(), 'last_hidden_state': result.last_hidden_state.contiguous(), 'pooler_output': result.pooler_output.contiguous(), 'hidden_patch_tokens': result.last_hidden_state[:,1+registers:,:].contiguous()}
    metadata = {'model_id':repo, 'revision':sha, 'torch':torch.__version__, 'transformers':transformers.__version__, 'dtype':'float32', 'device':'cpu', 'attention':'eager', 'input':'NHWC arange % 256 / 255; ImageNet normalized before reference inference'}
    save_file({k: v.clone() for k,v in tensors.items()}, str(out / (name+'.safetensors')), metadata=metadata)
    print({k:list(v.shape) for k,v in tensors.items()},flush=True)
