# Copyright 2025-2026 Lablup Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exact float32 access to a compressed-tensors `nvfp4-pack-quantized` checkpoint.

Out-of-band support for reference harnesses such as
`scripts/laguna_oracle_trace.py`; nothing in the runtime imports it. It reads
every `*.safetensors` shard the way mlxcel's loader does, dequantizes a plane
as `code * (E4M3(scale) / global_scale)` in float32 with no rounding to the
checkpoint dtype, and keeps stacked routed-expert planes compressed until a
module's forward needs them.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
NVFP4_BLOCK = 16


class CheckpointError(Exception):
    """A checkpoint whose tensors do not have the layout this module reads."""


class Checkpoint:
    """Every tensor of every `*.safetensors` shard, by key.

    Shards are globbed the way mlxcel's loader globs them, rather than read
    from `model.safetensors.index.json`, so both arms see the same tensors.
    """

    def __init__(self, model_dir: Path):
        from safetensors import safe_open

        shards = sorted(model_dir.glob("*.safetensors"))
        if not shards:
            raise CheckpointError(f"no *.safetensors shards in {model_dir}")
        self._handles = {}
        self._where: dict[str, str] = {}
        for shard in shards:
            handle = safe_open(str(shard), framework="pt")
            self._handles[shard.name] = handle
            for key in handle.keys():
                if key in self._where:
                    raise CheckpointError(f"{key} is in both {self._where[key]} and {shard.name}")
                self._where[key] = shard.name
        self.unread = set(self._where)

    def has(self, key: str) -> bool:
        return key in self._where

    def keys(self):
        return self._where.keys()

    def tensor(self, key: str):
        if key not in self._where:
            raise CheckpointError(f"checkpoint has no tensor {key}")
        self.unread.discard(key)
        return self._handles[self._where[key]].get_tensor(key)


class Nvfp4Decoder:
    """compressed-tensors `nvfp4-pack-quantized` to float32.

    Two E2M1 codes per byte with the low nibble first (the even column) and
    bit 3 the sign; one E4M3 block scale per 16 input columns; one float32
    global scale per tensor, which divides.
    """

    def __init__(self, device):
        import torch

        magnitudes = torch.tensor(E2M1_MAGNITUDES, dtype=torch.float32)
        e2m1 = torch.cat([magnitudes, -magnitudes])
        byte = torch.arange(256)
        # The two columns each packed byte expands to, low nibble first.
        self.pairs = torch.stack((e2m1[byte & 0x0F], e2m1[byte >> 4]), dim=-1).to(device)
        # Every E4M3 byte decoded by torch's own float8 conversion, which is
        # exact in float32 and independent of MLX.
        self.e4m3 = byte.to(torch.uint8).view(torch.float8_e4m3fn).to(torch.float32).to(device)

    def __call__(self, packed, scale_bytes, row_global, out=None):
        """`packed` [..., rows, cols/2] u8, `scale_bytes` [..., rows, cols/16]
        u8, `row_global` [..., rows] f32 -> [..., rows, cols] f32."""
        import torch

        lead, half = tuple(packed.shape[:-1]), packed.shape[-1]
        blocks = half * 2 // NVFP4_BLOCK
        if tuple(scale_bytes.shape) != (*lead, blocks):
            raise CheckpointError(
                f"block scales {tuple(scale_bytes.shape)} do not tile a {(*lead, half * 2)} plane"
            )
        values = self.pairs[packed.int()].view(*lead, blocks, NVFP4_BLOCK)
        block = (self.e4m3[scale_bytes.int()] / row_global.unsqueeze(-1)).unsqueeze(-1)
        if out is None:
            return (values * block).view(*lead, half * 2)
        torch.mul(values, block, out=out.view(*lead, blocks, NVFP4_BLOCK))
        return out


def scale_bytes_of(tensor):
    import torch

    if tensor.dtype == torch.float8_e4m3fn:
        return tensor.view(torch.uint8)
    if tensor.dtype == torch.uint8:
        return tensor
    raise CheckpointError(f"NVFP4 block scales must be F8_E4M3 or U8, found {tensor.dtype}")


def decode_linear(ckpt: Checkpoint, prefix: str, decoder: Nvfp4Decoder, device):
    """The dense float32 weight of one packed linear, `{prefix}.weight_packed`."""
    import torch

    packed = ckpt.tensor(f"{prefix}.weight_packed")
    scales = scale_bytes_of(ckpt.tensor(f"{prefix}.weight_scale"))
    g = ckpt.tensor(f"{prefix}.weight_global_scale").to(torch.float32).reshape(1)
    return decoder(packed.to(device), scales.to(device), g.expand(packed.shape[0]).to(device))


class PackedPlanes:
    """One stacked routed projection, still compressed: `packed` [experts,
    rows, cols/2] u8, `scale_bytes` [experts, rows, cols/16] u8, `row_global`
    [experts, rows] f32."""

    # Experts decoded per batch, bounding the int32 codes in flight.
    DECODE_BATCH = 32

    def __init__(self, packed, scale_bytes, row_global):
        self.packed = packed
        self.scale_bytes = scale_bytes
        self.row_global = row_global

    @property
    def shape(self) -> tuple[int, int, int]:
        experts, rows, half = self.packed.shape
        return (experts, rows, half * 2)

    def decode(self, decoder: Nvfp4Decoder, experts=None):
        """Decode to a full `[experts, rows, cols]` float32 tensor.

        With `experts` omitted, every expert is decoded, batched over
        contiguous ranges so at most `DECODE_BATCH` experts' codes are ever
        expanded to float32 at once. With `experts` given (the ids a
        forward actually selected), only those rows are decoded; every
        other expert's slot in the returned tensor is left uninitialized,
        which is sound only because the caller never reads it.
        """
        import torch

        out = torch.empty(self.shape, dtype=torch.float32, device=self.packed.device)
        if experts is None:
            for lo in range(0, self.shape[0], self.DECODE_BATCH):
                hi = lo + self.DECODE_BATCH
                decoder(
                    self.packed[lo:hi],
                    self.scale_bytes[lo:hi],
                    self.row_global[lo:hi],
                    out=out[lo:hi],
                )
            return out
        idx = experts.to(device=self.packed.device, dtype=torch.long)
        for lo in range(0, idx.shape[0], self.DECODE_BATCH):
            sel = idx[lo : lo + self.DECODE_BATCH]
            # `sel` is a tensor, so `out[sel]` is advanced indexing and would
            # hand `decoder(..., out=...)` a copy rather than a view; assign
            # the decoded batch back with `out[sel] = ...` instead, which
            # scatters correctly regardless of how `sel` is ordered.
            out[sel] = decoder(self.packed[sel], self.scale_bytes[sel], self.row_global[sel])
        return out


def stack_planes(
    ckpt: Checkpoint, prefix: str, projs: tuple[str, ...], num_experts: int, device
) -> PackedPlanes:
    """Stack `{prefix}.experts.{e}.{proj}` triplets, `projs` concatenated along
    the output rows in order (`gate_proj` then `up_proj` is `gate_up_proj`)."""
    import torch

    packed, scales, globals_ = [], [], []
    for e in range(num_experts):
        rows_p, rows_s, rows_g = [], [], []
        for proj in projs:
            base = f"{prefix}.experts.{e}.{proj}"
            if not ckpt.has(f"{base}.weight_packed"):
                raise CheckpointError(
                    f"{base}.weight_packed not found; routed experts are read from the "
                    "compressed-tensors NVFP4 layout only"
                )
            p = ckpt.tensor(f"{base}.weight_packed")
            rows_p.append(p)
            rows_s.append(scale_bytes_of(ckpt.tensor(f"{base}.weight_scale")))
            g = ckpt.tensor(f"{base}.weight_global_scale").to(torch.float32).reshape(1)
            rows_g.append(g.expand(p.shape[0]))
        packed.append(torch.cat(rows_p))
        scales.append(torch.cat(rows_s))
        globals_.append(torch.cat(rows_g))
    return PackedPlanes(
        torch.stack(packed).to(device),
        torch.stack(scales).to(device),
        torch.stack(globals_).to(device),
    )


class PackedExperts:
    """Keeps one experts module's planes compressed between forwards.

    For a module whose routed weights are 3-D parameters (Laguna's
    `LagunaExperts` holds `gate_up_proj` and `down_proj`), a forward pre-hook
    decodes the planes into those attributes as ordinary float32 tensors and a
    forward hook drops them again. The module's own forward runs unmodified on
    real tensors while only one layer's dense planes exist at once, and only
    the experts that layer's routing actually selected are ever decoded:
    `LagunaExperts.forward(hidden_states, top_k_index, top_k_weights)` only
    ever indexes `gate_up_proj[expert_idx]` / `down_proj[expert_idx]` for an
    `expert_idx` that appears in `top_k_index`, so decoding the rest is a
    wasted float32 write: an unselected expert's weights are never read
    (about 124 GB per forward on Laguna XS 2.1 decoding every expert of
    every layer, versus the handful actually routed to).
    """

    def __init__(self, decoder: Nvfp4Decoder, planes: dict[str, PackedPlanes]):
        self.decoder = decoder
        self.planes = planes
        self.materialized = 0

    def attach(self, module, name: str) -> None:
        for leaf, plane in self.planes.items():
            expected = tuple(module._parameters[leaf].shape)
            if plane.shape != expected:
                raise CheckpointError(
                    f"{name}.{leaf}: stacked planes {plane.shape}, model expects {expected}"
                )
            del module._parameters[leaf]
        module.register_forward_pre_hook(self._materialize)
        module.register_forward_hook(self._release)

    def _materialize(self, module, args) -> None:
        # `args` is `LagunaExperts.forward`'s positional arguments as the
        # module itself calls it, `(hidden_states, top_k_index,
        # top_k_weights)`; `top_k_index` names exactly the experts this
        # forward will read.
        top_k_index = args[1]
        selected = top_k_index.unique()
        for leaf, plane in self.planes.items():
            setattr(module, leaf, plane.decode(self.decoder, selected))
        self.materialized += 1

    def _release(self, module, args, output) -> None:
        for leaf in self.planes:
            delattr(module, leaf)


@contextmanager
def parameters_on_meta():
    """Register every parameter on the meta device while buffers stay real, so
    constructing the model allocates nothing and the rotary tables it computes
    in `__init__` survive."""
    from torch import nn

    original = nn.Module.register_parameter

    def register(module, name, param):
        original(module, name, param)
        if param is not None:
            module._parameters[name] = nn.Parameter(param.to("meta"), requires_grad=False)

    nn.Module.register_parameter = register
    try:
        yield
    finally:
        nn.Module.register_parameter = original
