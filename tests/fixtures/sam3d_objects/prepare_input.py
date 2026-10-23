"""Fetch the pinned external example; images remain outside source control."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from PIL import Image, __version__ as pillow_version


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    here=Path(__file__).resolve().parent
    provenance=json.loads((here/'input-provenance.json').read_text())
    if pillow_version != provenance['pillow_version']: raise ValueError('Pinned Pillow version required')
    a.output.mkdir(parents=True,exist_ok=True)
    originals=a.output/'original'; originals.mkdir(exist_ok=True)
    for info in provenance['original']['files']:
        target=originals/Path(info['path']).name
        if not target.exists():
            temporary=target.with_suffix('.download')
            with urllib.request.urlopen(info['url'],timeout=60) as response, temporary.open('wb') as out:
                total=0
                while block:=response.read(1024*1024):
                    total+=len(block)
                    if total>info['size']: raise ValueError('Oversized download')
                    out.write(block)
            if temporary.stat().st_size!=info['size'] or sha(temporary)!=info['sha256']: raise ValueError('Download identity mismatch')
            temporary.rename(target)
        if target.stat().st_size!=info['size'] or sha(target)!=info['sha256']: raise ValueError('Original identity mismatch')
    rgb=Image.open(originals/'image.png').convert('RGB').resize(tuple(provenance['size']),Image.Resampling.LANCZOS)
    mask=Image.open(originals/'0.png').convert('RGBA').getchannel('A').resize(tuple(provenance['size']),Image.Resampling.NEAREST)
    rgba=rgb.convert('RGBA'); rgba.putalpha(mask)
    for name,image in [('object.png',rgb),('mask.png',mask),('object-rgba.png',rgba)]:
        target=a.output/name
        if target.exists():
            if sha(target)!=provenance['files'][name]['sha256']: raise ValueError('Existing canonical input differs')
        else: image.save(target)
        if target.stat().st_size!=provenance['files'][name]['bytes'] or sha(target)!=provenance['files'][name]['sha256']: raise ValueError('Canonical transformation mismatch')
    print(json.dumps({'verified':provenance['files'],'output':str(a.output)}))

if __name__=='__main__': main()
