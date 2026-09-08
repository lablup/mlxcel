# Registers a stand-in AutoConfig for `nemotron-nas` so AutoTokenizer can resolve
# a config. The mlx conversion of this checkpoint kept config.json's `auto_map`
# pointing at configuration_decilm.DeciLMConfig but shipped no .py files, so
# transformers falls back to a bare PreTrainedConfig and its rope standardization
# then reads a `max_position_embeddings` that is not there. mlx-lm reads the model
# config from config.json directly, so this affects the tokenizer lookup only.
from transformers import AutoConfig, PretrainedConfig
class NemotronNasConfig(PretrainedConfig):
    model_type = "nemotron-nas"
    def __init__(self, max_position_embeddings=131072, **kw):
        self.max_position_embeddings = max_position_embeddings
        super().__init__(**kw)
AutoConfig.register("nemotron-nas", NemotronNasConfig, exist_ok=True)

import time, json, mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.generate import stream_generate
P = 'models/mlx/llama-3_3-nemotron-super-49b-4bit'
model, tok = load(P)
prompt = "Hello, how are you today?"
ids = tok.encode(prompt)
while len(ids) < 512:
    ids = ids + ids
ids = ids[:512]
text = tok.decode(ids)
# warmup
for _ in stream_generate(model, tok, text, max_tokens=20): pass
mx.eval(mx.zeros(1))
t0 = time.perf_counter(); first = None; n = 0
for resp in stream_generate(model, tok, text, max_tokens=128):
    if first is None: first = time.perf_counter()
    n += 1
t1 = time.perf_counter()
prefill_ms = (first - t0) * 1000
decode_ms = (t1 - first) * 1000
print(json.dumps({
  "prompt_tokens": len(ids), "generated": n,
  "prefill_ms": round(prefill_ms, 2), "prefill_tok_s": round(len(ids)/(prefill_ms/1000), 2),
  "decode_ms": round(decode_ms, 2), "decode_tok_s": round((n-1)/(decode_ms/1000), 2),
}))
