//! Stage 1 of the #449 throughput milestone: drive the real Llama-3.2-1B through
//! a uniform-B (lockstep) batched decode graph from Rust and measure aggregate
//! tok/s scaling, token-exact.
//!
//! The single-seq prefill runs once (reusing the scalar `prefill` vmfb); its KV
//! is tiled across B rows inside the shim; the batched decode vmfb (emitter
//! `decode-batch-argmax <B>`) then advances all B rows in lockstep (shared
//! pos/cache_len). Two checks. First, token-exact: B identical rows each
//! reproduce the 48-token HF temp-0 reference, and every row is identical
//! (batching does not corrupt the result). Second, independence: with row 0
//! seeded from the real first token and rows >=1 seeded from a different
//! (perturbed) token over the SAME prompt KV, row 0 stays token-exact while
//! row 1 diverges, so the batch dim is genuinely independent, not
//! collapsed/averaged (which identical rows would hide). Throughput is reported
//! as B * steps / wall-time (aggregate) and steps / wall-time (per-sequence).
//!
//! Run (after compiling the prefill + decode-batch vmfbs for this B):
//!   IREE_DIST=.../iree-dist cargo run --release --bin llama_batch -- \
//!     --batch 8 --prefill prefill.vmfb --decode decode_b8.vmfb
//!   # CUDA: build with IREE_CUDA_HOME set, run with --device cuda and the
//!   #       .cuda.vmfb graphs.

use std::ffi::CString;
use std::fs::File;
use std::os::raw::{c_char, c_int};
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

#[repr(C)]
struct XlaCtx {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn xla_llama_create(
        device: *const c_char,
        prefill: *const c_char,
        decode: *const c_char,
        n_weights: c_int,
        weight_data: *const *const f32,
        weight_ranks: *const c_int,
        weight_dims: *const i64,
        vocab: c_int,
    ) -> *mut XlaCtx;
    fn xla_llama_prefill_batch(
        c: *mut XlaCtx,
        bsz: c_int,
        tokens: *const c_int,
        lp: c_int,
        positions: *const c_int,
        real_len: c_int,
        out_tokens: *mut c_int,
    ) -> c_int;
    fn xla_llama_decode_batch(
        c: *mut XlaCtx,
        bsz: c_int,
        tokens: *const c_int,
        pos: c_int,
        cache_len: c_int,
        out_tokens: *mut c_int,
    ) -> c_int;
    fn xla_llama_free(c: *mut XlaCtx);
}

const N_LAYERS: usize = 16;
const VOCAB: i32 = 128256;
const LP: usize = 256; // prefill bucket (== MAX_SEQ; emitter PREFILL_LP)
const MAX_NEW: usize = 48;

/// The 146 weight names in the emitter's exact arg order (embed, final_norm,
/// then per layer down, gate, in_ln, post_ln, up, wk, wo, wq, wv).
fn weight_names() -> Vec<String> {
    let mut names = vec![
        "model.embed_tokens.weight".to_string(),
        "model.norm.weight".to_string(),
    ];
    for i in 0..N_LAYERS {
        let p = format!("model.layers.{i}.");
        for suf in [
            "mlp.down_proj.weight",
            "mlp.gate_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "mlp.up_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_proj.weight",
            "self_attn.v_proj.weight",
        ] {
            names.push(format!("{p}{suf}"));
        }
    }
    names
}

/// bf16 little-endian bytes -> f32 (bf16 is the high 16 bits of f32).
fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

fn read_int_array(path: &Path, key: &str) -> Vec<i32> {
    let v: serde_json::Value =
        serde_json::from_reader(File::open(path).expect("open json")).expect("parse json");
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} not an array in {}", path.display()))
        .iter()
        .map(|x| x.as_i64().expect("int") as i32)
        .collect()
}

fn arg(flag: &str, default: &str) -> String {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

/// Run `steps` lockstep batched decode steps from the given per-row seed tokens.
/// Returns one token stream per row, each `1 + steps` long (seed + outputs), and
/// the wall-clock seconds spent in the decode calls.
fn decode_loop(
    ctx: *mut XlaCtx,
    bsz: usize,
    seed: &[i32],
    start_clen: i32,
    steps: usize,
) -> (Vec<Vec<i32>>, f64) {
    let mut streams: Vec<Vec<i32>> = seed.iter().map(|&t| vec![t]).collect();
    let mut cur = seed.to_vec();
    let mut clen = start_clen;
    let t0 = std::time::Instant::now();
    for _ in 0..steps {
        let mut out = vec![0i32; bsz];
        let rc = unsafe {
            xla_llama_decode_batch(
                ctx,
                bsz as c_int,
                cur.as_ptr(),
                clen,
                clen,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(rc, 0, "decode_batch failed (status {rc})");
        for r in 0..bsz {
            streams[r].push(out[r]);
        }
        cur = out;
        clen += 1;
    }
    (streams, t0.elapsed().as_secs_f64())
}

fn main() {
    let here = env!("CARGO_MANIFEST_DIR");
    let ref_dir = PathBuf::from(here).join("../openxla");
    let model = arg(
        "--model",
        ref_dir
            .join("models/Llama-3.2-1B-Instruct")
            .to_str()
            .unwrap(),
    );
    let prefill_vmfb = arg("--prefill", "prefill.vmfb");
    let decode_vmfb = arg("--decode", "decode_batch.vmfb");
    let device = arg("--device", "local-task");
    let bsz: usize = arg("--batch", "1")
        .parse()
        .expect("--batch must be an integer");
    assert!(bsz >= 1, "--batch must be >= 1");

    let prompt_ids = read_int_array(&ref_dir.join("artifacts/prompt_ids.json"), "prompt_ids");
    let hf_ids = read_int_array(&ref_dir.join("artifacts/results.json"), "hf_ids");
    let n = prompt_ids.len();
    println!("prompt tokens = {n} (bucket Lp = {LP}), device = {device}, batch = {bsz}");
    assert!(n <= LP, "prompt ({n}) exceeds bucket ({LP})");

    // --- load weights bf16 -> f32 in emitter arg order ---
    let t0 = std::time::Instant::now();
    let st_path = Path::new(&model).join("model.safetensors");
    let file = File::open(&st_path).expect("open safetensors");
    let mmap = unsafe { Mmap::map(&file).expect("mmap safetensors") };
    let st = SafeTensors::deserialize(&mmap).expect("parse safetensors");

    let names = weight_names();
    let mut bufs: Vec<Vec<f32>> = Vec::with_capacity(names.len());
    let mut ranks: Vec<c_int> = Vec::with_capacity(names.len());
    let mut dims: Vec<i64> = Vec::with_capacity(names.len() * 4);
    for name in &names {
        let t = st
            .tensor(name)
            .unwrap_or_else(|_| panic!("missing weight {name}"));
        assert_eq!(t.dtype(), Dtype::BF16, "{name} dtype {:?}", t.dtype());
        let shape = t.shape();
        assert!(shape.len() <= 4, "{name} rank {} > 4", shape.len());
        ranks.push(shape.len() as c_int);
        let mut d4 = [0i64; 4];
        for (k, &s) in shape.iter().enumerate() {
            d4[k] = s as i64;
        }
        dims.extend_from_slice(&d4);
        bufs.push(bf16_to_f32(t.data()));
    }
    let ptrs: Vec<*const f32> = bufs.iter().map(|b| b.as_ptr()).collect();
    println!(
        "loaded {} weight tensors ({:.1} GB f32) in {:.1}s",
        names.len(),
        bufs.iter().map(|b| b.len()).sum::<usize>() as f64 * 4.0 / 1e9,
        t0.elapsed().as_secs_f64()
    );

    // --- create the execution context (uploads weights to resident buffers) ---
    let c_dev = CString::new(device.clone()).unwrap();
    let c_pre = CString::new(prefill_vmfb).unwrap();
    let c_dec = CString::new(decode_vmfb).unwrap();
    let ctx = unsafe {
        xla_llama_create(
            c_dev.as_ptr(),
            c_pre.as_ptr(),
            c_dec.as_ptr(),
            names.len() as c_int,
            ptrs.as_ptr(),
            ranks.as_ptr(),
            dims.as_ptr(),
            VOCAB,
        )
    };
    assert!(!ctx.is_null(), "xla_llama_create returned null");
    drop(ptrs);
    drop(bufs); // weights now resident on device; free the host copy

    // --- prefill the padded prompt once; the shim tiles its KV across B rows ---
    let mut tokens = vec![0i32; LP];
    tokens[..n].copy_from_slice(&prompt_ids);
    let positions: Vec<i32> = (0..LP as i32).collect();
    let do_prefill = |ctx: *mut XlaCtx| -> (Vec<i32>, f64) {
        let mut firsts = vec![0i32; bsz];
        let t = std::time::Instant::now();
        let rc = unsafe {
            xla_llama_prefill_batch(
                ctx,
                bsz as c_int,
                tokens.as_ptr(),
                LP as c_int,
                positions.as_ptr(),
                n as c_int,
                firsts.as_mut_ptr(),
            )
        };
        assert_eq!(rc, 0, "prefill_batch failed (status {rc})");
        (firsts, t.elapsed().as_secs_f64() * 1e3)
    };

    let steps = MAX_NEW - 1; // 47 decode steps -> 48-token streams
    let start_clen = n as i32;

    // === run 1: identical rows -> token-exact + throughput =================
    let (firsts, prefill_ms) = do_prefill(ctx);
    let first = firsts[0];
    let (streams, secs) = decode_loop(ctx, bsz, &firsts, start_clen, steps);
    let agg_tok_s = (bsz * steps) as f64 / secs;
    let per_seq_tok_s = steps as f64 / secs;
    let ms_per_step = secs * 1e3 / steps as f64;

    let row0 = &streams[0];
    let m = row0.len().min(hf_ids.len());
    let matches = (0..m).filter(|&i| row0[i] == hf_ids[i]).count();
    let all_identical = streams.iter().all(|s| s == row0);
    let exact = matches == m && m == hf_ids.len() && all_identical;

    // === run 2: independence (row 0 real, rows >=1 perturbed) =============
    let _ = do_prefill(ctx); // reset KV to the prompt
    let pert = hf_ids
        .get(1)
        .copied()
        .filter(|&t| t != first)
        .unwrap_or((first + 1) % VOCAB);
    let mut seed2 = vec![pert; bsz];
    seed2[0] = first;
    let (streams2, _) = decode_loop(ctx, bsz, &seed2, start_clen, steps);
    let row0b = &streams2[0];
    let m2 = row0b.len().min(hf_ids.len());
    let matches2 = (0..m2).filter(|&i| row0b[i] == hf_ids[i]).count();
    let row0_exact = matches2 == m2 && m2 == hf_ids.len();
    let row1_diverges = bsz < 2 || streams2[1] != *row0b;

    unsafe { xla_llama_free(ctx) };

    // --- report ---
    println!("\n=== uniform-B batched decode (B={bsz}, {device}) ===");
    println!("generated ids (row 0): {row0:?}");
    println!(
        "prefill: {prefill_ms:.0} ms | decode: {ms_per_step:.1} ms/step | \
         per-seq {per_seq_tok_s:.2} tok/s | aggregate {agg_tok_s:.2} tok/s ({bsz}x)"
    );
    println!(
        "token match vs HF temp-0: {matches}/{m}  (all {bsz} rows identical: {all_identical})"
    );
    println!(
        "independence: row0 {matches2}/{m2} exact={row0_exact}, row1 diverges={row1_diverges}"
    );

    let ok = exact && row0_exact && row1_diverges;
    println!(
        "RESULT: {}",
        if ok { "TOKEN-EXACT PASS" } else { "MISMATCH" }
    );
    // Machine-readable line for the sweep script to scrape.
    println!(
        "SCALE\tdevice={device}\tB={bsz}\tms_per_step={ms_per_step:.3}\t\
         per_seq_tok_s={per_seq_tok_s:.3}\tagg_tok_s={agg_tok_s:.3}\tpass={ok}"
    );
    std::process::exit(if ok { 0 } else { 1 });
}
