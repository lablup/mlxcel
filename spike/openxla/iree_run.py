"""Bonus: execute one decode step through the IREE runtime on the SAME exported
StableHLO and confirm the next-token logits match the JAX/PJRT path. Proves the
exported graph is portable across two independent runtimes (PJRT and IREE)."""
import numpy as np
import jax, jax.numpy as jnp
from iree.runtime import load_vm_flatbuffer_file

import model_jax as M

MODEL = "models/Llama-3.2-1B-Instruct"
MAX_SEQ = 256
Lp = 64


def main():
    cfg = M.load_config(MODEL)
    params = M.load_params(MODEL, cfg)
    prefill, decode_step = M.make_fns(cfg, MAX_SEQ)

    # drive prefill on JAX to get a real cache + first token
    prompt = np.zeros(Lp, np.int32)
    prompt[:5] = np.array([128000, 9906, 11, 1268, 527], np.int32)  # "<bos>Hello, how are"
    real = 5
    logits, kc, vc = jax.jit(prefill)(params, jnp.asarray(prompt), jnp.arange(Lp, dtype=jnp.int32), jnp.int32(real))
    tok0 = int(np.asarray(logits).argmax())

    # one decode step on JAX (PJRT)
    jx_logits, _, _ = jax.jit(decode_step)(params, jnp.int32(tok0), jnp.int32(real), jnp.int32(real), kc, vc)
    jx_logits = np.asarray(jx_logits)

    # same step on IREE
    leaves = jax.tree_util.tree_flatten(params)[0]
    inputs = [np.asarray(x, np.float32) for x in leaves]
    inputs += [np.asarray(tok0, np.int32), np.asarray(real, np.int32), np.asarray(real, np.int32),
               np.asarray(kc, np.float32), np.asarray(vc, np.float32)]
    mod = load_vm_flatbuffer_file("artifacts/decode_step.vmfb", driver="local-task")
    ir_out = mod.main(*inputs)
    ir_logits = np.asarray(ir_out[0].to_host() if hasattr(ir_out[0], "to_host") else ir_out[0])

    print(f"JAX/PJRT next-token argmax : {int(jx_logits.argmax())}")
    print(f"IREE     next-token argmax : {int(ir_logits.argmax())}")
    print(f"argmax match               : {int(jx_logits.argmax()) == int(ir_logits.argmax())}")
    d = np.abs(jx_logits - ir_logits)
    print(f"logit abs diff PJRT vs IREE: max={d.max():.4e} mean={d.mean():.4e}")


if __name__ == "__main__":
    main()
