//! End-to-end Rust proof for issue #449 Phase 3 M2: drive the real Llama-3.2-1B
//! through IREE from Rust, token-exact.
//!
//! Loads the bf16 weights (upcast f32) from the model dir, hands them to the C
//! shim as 146 resident device buffers, runs the #451-emitted `prefill` then
//! `decode_step` vmfbs (one IREE session, threaded KV cache), and checks the 48
//! generated tokens against the HF temp-0 reference in
//! `spike/openxla/artifacts/results.json`. This is the standalone gate before
//! the same shim is ported into `mlxcel-xla`.
//!
//! Run (after compiling prefill.vmfb / decode.vmfb with the dist iree-compile):
//!   IREE_DIST=.../iree-dist cargo run --release --bin llama
//!   IREE_DIST=.../iree-dist cargo run --release --bin llama -- --device cuda

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
    fn xla_llama_prefill(
        c: *mut XlaCtx,
        tokens: *const c_int,
        lp: c_int,
        positions: *const c_int,
        real_len: c_int,
        out_token: *mut c_int,
    ) -> c_int;
    fn xla_llama_decode(
        c: *mut XlaCtx,
        token: c_int,
        pos: c_int,
        cache_len: c_int,
        out_token: *mut c_int,
    ) -> c_int;
    fn xla_llama_free(c: *mut XlaCtx);
}

const N_LAYERS: usize = 16;
const VOCAB: i32 = 128256;
const LP: usize = 256; // prefill bucket (== MAX_SEQ; emitter PREFILL_LP)
const MAX_NEW: usize = 48;
const EOS: [i32; 3] = [128001, 128008, 128009];

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
    let decode_vmfb = arg("--decode", "decode.vmfb");
    let device = arg("--device", "local-task");

    let prompt_ids = read_int_array(&ref_dir.join("artifacts/prompt_ids.json"), "prompt_ids");
    let hf_ids = read_int_array(&ref_dir.join("artifacts/results.json"), "hf_ids");
    let n = prompt_ids.len();
    println!("prompt tokens = {n} (bucket Lp = {LP}), device = {device}");
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
    drop(bufs); // weights now resident on device; free the 5 GB host copy

    // --- prefill the whole padded prompt, then decode greedily ---
    let mut tokens = vec![0i32; LP];
    tokens[..n].copy_from_slice(&prompt_ids);
    let positions: Vec<i32> = (0..LP as i32).collect();

    let t0 = std::time::Instant::now();
    let mut first = 0i32;
    let rc = unsafe {
        xla_llama_prefill(
            ctx,
            tokens.as_ptr(),
            LP as c_int,
            positions.as_ptr(),
            n as c_int,
            &mut first,
        )
    };
    assert_eq!(rc, 0, "prefill failed (status {rc})");
    let prefill_ms = t0.elapsed().as_secs_f64() * 1e3;

    let mut out_ids = vec![first];
    let mut clen = n as i32;
    let mut next_tok = first;
    let t0 = std::time::Instant::now();
    let mut steps = 0usize;
    for _ in 0..(MAX_NEW - 1) {
        if EOS.contains(&next_tok) {
            break;
        }
        let mut nt = 0i32;
        let rc = unsafe { xla_llama_decode(ctx, next_tok, clen, clen, &mut nt) };
        assert_eq!(rc, 0, "decode failed (status {rc})");
        out_ids.push(nt);
        clen += 1;
        next_tok = nt;
        steps += 1;
    }
    let decode_ms = if steps > 0 {
        t0.elapsed().as_secs_f64() * 1e3 / steps as f64
    } else {
        f64::NAN
    };
    unsafe { xla_llama_free(ctx) };

    // --- compare to HF temp-0 reference ---
    let m = out_ids.len().min(hf_ids.len());
    let matches = (0..m).filter(|&i| out_ids[i] == hf_ids[i]).count();
    let first_div = (0..m).find(|&i| out_ids[i] != hf_ids[i]);

    // The shim auto-detects the output: a [V] logits graph is argmaxed on the
    // host; an on-device-argmax graph returns the token id directly (4 bytes).
    println!("\n=== Rust-driven IREE prefill + decode ===");
    println!("generated ids: {out_ids:?}");
    println!("prefill: {prefill_ms:.0} ms ({n} tok, bucket {LP}) | decode: {decode_ms:.0} ms/tok");
    match first_div {
        Some(i) => println!("token match vs HF temp-0: {matches}/{m}  first divergence at {i}"),
        None => println!("token match vs HF temp-0: {matches}/{m}  (EXACT)"),
    }
    let ok = matches == m && m == hf_ids.len();
    println!(
        "RESULT: {}",
        if ok { "TOKEN-EXACT PASS" } else { "MISMATCH" }
    );
    std::process::exit(if ok { 0 } else { 1 });
}
