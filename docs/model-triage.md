# Model coverage triage

Dispositions for model families that were reported as unsupported and then checked against the tree. Each entry gives the evidence it rests on. "Not found" means not found in the file or search named, not proof of absence.

Last checked against `main` at 6a115916.

## Already supported

| Family | Evidence |
|---|---|
| `falcon_ocr` | `src/models/detection.rs` (`"falcon_ocr" => FalconOcrVL`), `src/loading/vlm_falcon_ocr.rs`, `src/commands/generate_falcon_ocr.rs`, [supported-models.md](supported-models.md) |
| `jina_vlm` | `src/models/detection.rs` (`"jvlm" \| "jina_vlm" => JinaVLM`) |
| `laguna` | `src/models/detection.rs` (`"laguna" => Laguna`) |

## Not found in `detection.rs`

`zaya1_vl`, `hrm_text` and `multi_modality` have no `model_type` arm in `src/models/detection.rs`. They have not been triaged here.

## `falcon_perception`

Two public checkpoints, both Apache-2.0 with `model_type: falcon_perception` and fp32 weights:

| | `tiiuae/Falcon-Perception` | `tiiuae/Falcon-Perception-300M` |
|---|---|---|
| Size | 2.53 GB, 231 tensors | 1.27 GB |
| Decoder | dim 1024, 28 layers, head_dim 128 | dim 768, 22 layers, head_dim 64 |
| `do_segmentation` | true | false (detection only) |

### Decoder

`Attention`, `FeedForward` and `TransformerBlock` in the checkpoint's `modeling_falcon_perception.py` were diffed against `modeling_falcon_ocr.py`. Apart from comments and imports they are the same, so the decoder looks reusable from the `falcon_ocr` runtime with different sizes. The mlxcel runtime was not read to check that nothing in it hardcodes `dim` or `head_dim`.

### What a port adds on top of `falcon_ocr`

- `coord_encoder` and `size_encoder` (Fourier features: `embed` then `transform`) and `coord_decoder` and `size_decoder` (two-layer squared-ReLU MLPs). Their outputs replace the embedding at `coord_token_id` (240) and `size_token_id` (241).
- A box decoding loop. Coordinates are an argmax over bins. A prediction within 1% of an earlier box is rejected and drawn again, up to 100 attempts.
- Large checkpoint only: a segmentation path on `seg_token_id` (262). It uses `proj_segm`, `conv_segm` with weight `(256, 1024, 3, 3)` (the 4-D convolution axis order needs the usual check), and the AnyUp upsampler (`anyup.py`, 479 lines). The upsampler produces high-resolution features that are dotted with the projected token to form a mask, which is then RLE-encoded.
- A fixed prompt contract: `<|image|>Segment these expressions in the image:<|start_of_query|>{query}<|REF_SEG|>`.

### Suggested split

1. Port the 300M detection-only checkpoint first: reuse the `falcon_ocr` decoder and add the Fourier encoders and decoders and the box loop. Box output can follow the `mlxcel detect` precedent (`src/commands/detect.rs`).
2. Treat the mask path of the large checkpoint as a separate piece of work: AnyUp, `proj_segm`, `conv_segm` and RLE.

### Open decisions

- The output contract for boxes and masks beyond chat completions.
- Whether the 300M port is worth filing before the mask path is scoped.

### Not checked

- Whether the 300M `modeling_falcon_perception.py` is identical to the large checkpoint's.
- The 300M tensor names. Only the large checkpoint's safetensors header was read.
