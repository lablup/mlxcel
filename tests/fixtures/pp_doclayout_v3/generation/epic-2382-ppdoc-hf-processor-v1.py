from pathlib import Path
import json,hashlib,inspect,importlib.metadata
import numpy as np
from PIL import Image
import torch,torchvision,transformers
from transformers import PPDocLayoutV3ImageProcessor
from safetensors.numpy import save_file
root=Path('/home/devops/Development/wt-epic-2382-issue-2412/target/epic-2382-ppdoc-reference');checkpoint=Path('/home/devops/Development/mlxcel/.cache/epic-2382/preflight-ppdoc-checkpoints/paddle/e4f489dc536556fc1be02b973ba06756e0d4a2ac');image=Image.open(root/'bert-page/two_column_page.png').convert('RGB');processor=PPDocLayoutV3ImageProcessor.from_pretrained(checkpoint,local_files_only=True);result=processor(images=image,return_tensors='pt')['pixel_values'];values=result.numpy().transpose(0,2,3,1).copy()
if values.shape!=(1,800,800,3) or values.dtype!=np.float32 or not np.isfinite(values).all():raise RuntimeError('processor output invalid')
out=root/'hf-processor-input.safetensors';save_file({'decoded_rgb_u8':np.asarray(image),'pixel_values_nhwc':values},out)
pil=np.asarray(image.resize((800,800),Image.Resampling.BICUBIC)).astype(np.float32)/255
receipt={'checkpoint':str(checkpoint),'config_sha256':hashlib.sha256((checkpoint/'preprocessor_config.json').read_bytes()).hexdigest(),'device':str(result.device),'shape':list(values.shape),'dtype':str(values.dtype),'decoded_sha256':hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),'processed_sha256':hashlib.sha256(values.tobytes()).hexdigest(),'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'versions':{n:importlib.metadata.version(n) for n in ['torch','torchvision','transformers','Pillow','numpy','safetensors']},'processor_source':inspect.getsourcefile(PPDocLayoutV3ImageProcessor),'processor_source_sha256':hashlib.sha256(Path(inspect.getsourcefile(PPDocLayoutV3ImageProcessor)).read_bytes()).hexdigest(),'pil_max_abs_difference':float(np.max(np.abs(values[0]-pil)))};(root/'hf-processor-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
