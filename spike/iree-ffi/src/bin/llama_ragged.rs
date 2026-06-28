//! Stage 2a of the #449 throughput milestone: validate the ragged
//! (continuous-batching) decode graph by reference-equivalence.
//!
//! Continuous batching has no single reference, so this drives two paths over
//! the SAME weights:
//!   Phase 1 (references): for each of B prompts of DIFFERENT lengths
//!     (truncations of the reference prompt), run the single-seq prefill +
//!     decode to capture its independent greedy token stream.
//!   Phase 2 (ragged batch): prefill the B prompts into B slots at their own
//!     lengths, commit the rank-5 KV, then run the ragged decode in lockstep
//!     (every slot steps together, but each at its OWN position/length).
//! The gate: every slot's stream must match its independent single-seq
//! reference. That proves a sequence's output is invariant to which peers share
//! its batch, i.e. the per-row pos/cache_len/mask and the per-row KV write are
//! all correct. Throughput is reported as B * steps / wall.
//!
//! Two IREE sessions are used (one module can hold only one @decode_step): a
//! single-seq decode vmfb for Phase 1 and the ragged decode vmfb for Phase 2.
//! Weights are loaded once and uploaded to each session.
//!
//! Run (after compiling prefill + single-seq decode + ragged decode vmfbs):
//!   IREE_DIST=.../iree-dist cargo run --release --bin llama_ragged -- \
//!     --batch 4 --prefill prefill.vmfb --sdecode decode.vmfb --decode dr4.vmfb
//!   # CUDA: build with IREE_CUDA_HOME, run with --device cuda + the .cuda.vmfb graphs.

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
    // single-seq path (Phase 1 references)
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
    // ragged path (Phase 2)
    fn xla_llama_ragged_reset(c: *mut XlaCtx, bsz: c_int) -> c_int;
    fn xla_llama_prefill_slot(
        c: *mut XlaCtx,
        slot: c_int,
        tokens: *const c_int,
        lp: c_int,
        positions: *const c_int,
        real_len: c_int,
        out_first: *mut c_int,
    ) -> c_int;
    fn xla_llama_commit(c: *mut XlaCtx) -> c_int;
    fn xla_llama_decode_ragged(
        c: *mut XlaCtx,
        bsz: c_int,
        tokens: *const c_int,
        pos: *const c_int,
        cache_len: *const c_int,
        out_tokens: *mut c_int,
    ) -> c_int;
    fn xla_llama_free(c: *mut XlaCtx);
}

const N_LAYERS: usize = 16;
const VOCAB: i32 = 128256;
const LP: usize = 256; // prefill bucket (== MAX_SEQ; emitter PREFILL_LP)
const MAX_NEW: usize = 48;
const EOS: [i32; 3] = [128001, 128008, 128009];

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

/// Pad a prompt of length `n` into an `LP`-sized token buffer.
fn padded(prompt: &[i32], n: usize) -> Vec<i32> {
    let mut t = vec![0i32; LP];
    t[..n].copy_from_slice(&prompt[..n]);
    t
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
    let sdecode_vmfb = arg("--sdecode", "decode.vmfb"); // single-seq decode (references)
    let rdecode_vmfb = arg("--decode", "decode_ragged.vmfb"); // ragged decode (batch)
    let device = arg("--device", "local-task");
    let bsz: usize = arg("--batch", "4")
        .parse()
        .expect("--batch must be an integer");
    assert!(bsz >= 1, "--batch must be >= 1");

    let prompt_ids = read_int_array(&ref_dir.join("artifacts/prompt_ids.json"), "prompt_ids");
    let full = prompt_ids.len();
    // B prompts of distinct lengths: truncations of the reference prompt.
    let lens: Vec<usize> = (0..bsz)
        .map(|r| ((full as i32) - (r as i32) * 3).max(8) as usize)
        .collect();
    let positions: Vec<i32> = (0..LP as i32).collect();
    println!(
        "device = {device}, batch = {bsz}, prompt lengths = {lens:?} (full = {full}, bucket {LP})"
    );

    // --- load weights once (bf16 -> f32) ---
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
        "loaded {} weights ({:.1} GB f32) in {:.1}s",
        names.len(),
        bufs.iter().map(|b| b.len()).sum::<usize>() as f64 * 4.0 / 1e9,
        t0.elapsed().as_secs_f64()
    );

    let c_dev = CString::new(device.clone()).unwrap();
    let c_pre = CString::new(prefill_vmfb).unwrap();
    let create = |decode_vmfb: &str| -> *mut XlaCtx {
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
        ctx
    };

    // === Phase 1: single-seq references, one per prompt length ===
    let ctx1 = create(&sdecode_vmfb);
    let mut refs: Vec<Vec<i32>> = Vec::with_capacity(bsz);
    for &n in &lens {
        let tokens = padded(&prompt_ids, n);
        let mut first = 0i32;
        let rc = unsafe {
            xla_llama_prefill(
                ctx1,
                tokens.as_ptr(),
                LP as c_int,
                positions.as_ptr(),
                n as c_int,
                &mut first,
            )
        };
        assert_eq!(rc, 0, "ref prefill failed (status {rc})");
        let mut stream = vec![first];
        let mut clen = n as i32;
        let mut next = first;
        for _ in 0..(MAX_NEW - 1) {
            if EOS.contains(&next) {
                break;
            }
            let mut nt = 0i32;
            let rc = unsafe { xla_llama_decode(ctx1, next, clen, clen, &mut nt) };
            assert_eq!(rc, 0, "ref decode failed (status {rc})");
            stream.push(nt);
            clen += 1;
            next = nt;
        }
        refs.push(stream);
    }
    unsafe { xla_llama_free(ctx1) };
    println!("phase 1: captured {} single-seq references", refs.len());

    // === Phase 2: ragged batch, all slots stepped together (ragged positions) ===
    let ctx2 = create(&rdecode_vmfb);
    assert_eq!(unsafe { xla_llama_ragged_reset(ctx2, bsz as c_int) }, 0);
    let mut firsts = vec![0i32; bsz];
    for (r, &n) in lens.iter().enumerate() {
        let tokens = padded(&prompt_ids, n);
        let rc = unsafe {
            xla_llama_prefill_slot(
                ctx2,
                r as c_int,
                tokens.as_ptr(),
                LP as c_int,
                positions.as_ptr(),
                n as c_int,
                &mut firsts[r],
            )
        };
        assert_eq!(rc, 0, "prefill_slot {r} failed (status {rc})");
    }
    assert_eq!(unsafe { xla_llama_commit(ctx2) }, 0, "commit failed");

    let steps = MAX_NEW - 1;
    let mut streams: Vec<Vec<i32>> = firsts.iter().map(|&f| vec![f]).collect();
    let mut cur = firsts.clone();
    let mut clen: Vec<i32> = lens.iter().map(|&n| n as i32).collect();
    let t0 = std::time::Instant::now();
    for _ in 0..steps {
        let pos = clen.clone(); // pos[r] == cache_len[r] in decode
        let mut out = vec![0i32; bsz];
        let rc = unsafe {
            xla_llama_decode_ragged(
                ctx2,
                bsz as c_int,
                cur.as_ptr(),
                pos.as_ptr(),
                clen.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        assert_eq!(rc, 0, "decode_ragged failed (status {rc})");
        for r in 0..bsz {
            streams[r].push(out[r]);
        }
        cur = out;
        for v in clen.iter_mut() {
            *v += 1;
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    unsafe { xla_llama_free(ctx2) };

    // === compare each slot's stream to its independent single-seq reference ===
    let mut all_ok = true;
    for r in 0..bsz {
        let rf = &refs[r];
        let st = &streams[r];
        let m = rf.len(); // compare up to the reference length (ragged runs to MAX_NEW)
        let matches = (0..m).filter(|&i| st.get(i) == rf.get(i)).count();
        let first_ok = firsts[r] == rf[0];
        let ok = matches == m && first_ok;
        all_ok &= ok;
        let div = (0..m).find(|&i| st.get(i) != rf.get(i));
        println!(
            "slot {r} (len {:>3}): {matches}/{m} match{}{}",
            lens[r],
            if first_ok {
                ""
            } else {
                "  [first-token mismatch]"
            },
            match div {
                Some(i) => format!("  first divergence @ {i}"),
                None => "  (EXACT)".to_string(),
            }
        );
    }

    let agg_tok_s = (bsz * steps) as f64 / secs;
    println!(
        "\nragged decode: {:.1} ms/step | aggregate {agg_tok_s:.2} tok/s ({bsz}x, {steps} steps)",
        secs * 1e3 / steps as f64
    );
    println!(
        "RESULT: {}",
        if all_ok {
            "REFERENCE-EXACT PASS"
        } else {
            "MISMATCH"
        }
    );
    println!("RAGGED\tdevice={device}\tB={bsz}\tagg_tok_s={agg_tok_s:.3}\tpass={all_ok}");
    std::process::exit(if all_ok { 0 } else { 1 });
}
