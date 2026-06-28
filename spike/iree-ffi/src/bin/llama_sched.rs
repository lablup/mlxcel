//! Stage 2a-ii of the #449 throughput milestone: a minimal continuous-batching
//! scheduler over the ragged decode graph, validated by reference-equivalence
//! under dynamic batch membership.
//!
//! `B` slots serve `N > B` requests of DIFFERENT prompt lengths AND different
//! token caps, so requests finish at staggered times and queued requests are
//! admitted mid-stream into freed slots (slot recycling). Mid-stream admit is
//! refresh-mirror (pull the live KV of all active slots back to the host) +
//! prefill the new prompt into the freed slot + commit (re-upload), so admitting
//! one sequence does not disturb the others.
//!
//! The gate: every request's output stream must equal its INDEPENDENT single-seq
//! reference, regardless of when it was admitted or which peers shared its batch.
//! That proves admit/evict/recycle and the ragged decode are all correct.
//!
//! Two IREE sessions (one @decode_step each): a single-seq decode vmfb for the
//! Phase 1 references, the ragged decode vmfb for the Phase 2 scheduler.
//!
//! Run:
//!   IREE_DIST=.../iree-dist cargo run --release --bin llama_sched -- \
//!     --batch 4 --requests 8 --prefill prefill.vmfb --sdecode decode.vmfb --decode dr4.vmfb
//!   # CUDA: build with IREE_CUDA_HOME, run with --device cuda + the .cuda.vmfb graphs.

use std::collections::VecDeque;
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
    fn xla_llama_refresh_mirror(c: *mut XlaCtx) -> c_int;
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
const LP: usize = 256;
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

fn padded(prompt: &[i32], n: usize) -> Vec<i32> {
    let mut t = vec![0i32; LP];
    t[..n].copy_from_slice(&prompt[..n]);
    t
}

/// A request is finished when it reached its token cap or emitted an EOS token.
fn finished(stream: &[i32], cap: usize) -> bool {
    stream.len() >= cap || stream.last().is_some_and(|t| EOS.contains(t))
}

struct Req {
    len: usize,
    cap: usize,
    reference: Vec<i32>,
    stream: Vec<i32>,
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
    let sdecode_vmfb = arg("--sdecode", "decode.vmfb");
    let rdecode_vmfb = arg("--decode", "decode_ragged.vmfb");
    let device = arg("--device", "local-task");
    let bsz: usize = arg("--batch", "4").parse().expect("--batch int");
    let nreq: usize = arg("--requests", &(2 * bsz).to_string())
        .parse()
        .expect("--requests int");
    // Clamp per-request token caps (keeps the slow single-seq reference pass short
    // on CPU; the staggered-eviction behavior holds for any spread of caps).
    let maxcap: usize = arg("--maxcap", "48").parse().expect("--maxcap int");
    assert!(bsz >= 1 && nreq >= 1 && maxcap >= 2);

    let prompt_ids = read_int_array(&ref_dir.join("artifacts/prompt_ids.json"), "prompt_ids");
    let full = prompt_ids.len();
    let positions: Vec<i32> = (0..LP as i32).collect();

    // N requests with varied lengths AND varied caps, so they evict at staggered
    // times and queued requests are admitted into freed slots mid-stream.
    let mut reqs: Vec<Req> = (0..nreq)
        .map(|i| {
            let len = (20 + (i * 13) % (full.saturating_sub(20).max(1))).min(full);
            let cap = (12 + (i * 11) % 28).min(maxcap); // spread of caps, clamped
            Req {
                len,
                cap,
                reference: Vec::new(),
                stream: Vec::new(),
            }
        })
        .collect();
    println!("device = {device}, slots B = {bsz}, requests N = {nreq}, full prompt = {full}");
    for (i, r) in reqs.iter().enumerate() {
        print!(
            "  req {i}: len {} cap {}{}",
            r.len,
            r.cap,
            if i % 4 == 3 { "\n" } else { "" }
        );
    }
    println!();

    // --- load weights once ---
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
        assert_eq!(t.dtype(), Dtype::BF16);
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
        assert!(!ctx.is_null(), "create returned null");
        ctx
    };

    // === Phase 1: single-seq reference per request (capped) ===
    let ctx1 = create(&sdecode_vmfb);
    for r in reqs.iter_mut() {
        let tokens = padded(&prompt_ids, r.len);
        let mut first = 0i32;
        assert_eq!(
            unsafe {
                xla_llama_prefill(
                    ctx1,
                    tokens.as_ptr(),
                    LP as c_int,
                    positions.as_ptr(),
                    r.len as c_int,
                    &mut first,
                )
            },
            0
        );
        let mut stream = vec![first];
        let (mut cur, mut clen) = (first, r.len as i32);
        while !finished(&stream, r.cap) {
            let mut nt = 0i32;
            assert_eq!(
                unsafe { xla_llama_decode(ctx1, cur, clen, clen, &mut nt) },
                0
            );
            stream.push(nt);
            clen += 1;
            cur = nt;
        }
        r.reference = stream;
    }
    unsafe { xla_llama_free(ctx1) };
    println!(
        "phase 1: captured {} capped single-seq references",
        reqs.len()
    );

    // === Phase 2: continuous-batching scheduler ===
    let ctx2 = create(&rdecode_vmfb);
    assert_eq!(unsafe { xla_llama_ragged_reset(ctx2, bsz as c_int) }, 0);
    let mut queue: VecDeque<usize> = (0..nreq).collect();
    let mut slot_req: Vec<Option<usize>> = vec![None; bsz];
    let mut cur = vec![0i32; bsz];
    let mut clen = vec![0i32; bsz];
    let mut device_live = false;
    let mut done = 0usize;
    let mut admits = 0usize;
    let mut decode_steps = 0usize;
    let safety = nreq * 64 + 64;
    let t0 = std::time::Instant::now();

    let mut iter = 0usize;
    while done < nreq {
        iter += 1;
        assert!(iter < safety, "scheduler exceeded the safety iteration cap");

        // ADMIT queued requests into free slots
        let free: Vec<usize> = (0..bsz).filter(|&s| slot_req[s].is_none()).collect();
        let n_admit = free.len().min(queue.len());
        if n_admit > 0 {
            if device_live {
                assert_eq!(
                    unsafe { xla_llama_refresh_mirror(ctx2) },
                    0,
                    "refresh failed"
                );
            }
            for &s in free.iter().take(n_admit) {
                let ri = queue.pop_front().unwrap();
                let tokens = padded(&prompt_ids, reqs[ri].len);
                let mut first = 0i32;
                assert_eq!(
                    unsafe {
                        xla_llama_prefill_slot(
                            ctx2,
                            s as c_int,
                            tokens.as_ptr(),
                            LP as c_int,
                            positions.as_ptr(),
                            reqs[ri].len as c_int,
                            &mut first,
                        )
                    },
                    0,
                    "prefill_slot failed"
                );
                reqs[ri].stream.push(first);
                slot_req[s] = Some(ri);
                cur[s] = first;
                clen[s] = reqs[ri].len as i32;
                admits += 1;
            }
            assert_eq!(unsafe { xla_llama_commit(ctx2) }, 0, "commit failed");
            device_live = true;
            // a request that finished at its very first token frees its slot now
            for &s in free.iter().take(n_admit) {
                if let Some(ri) = slot_req[s]
                    && finished(&reqs[ri].stream, reqs[ri].cap)
                {
                    slot_req[s] = None;
                    done += 1;
                }
            }
        }

        let active: Vec<usize> = (0..bsz).filter(|&s| slot_req[s].is_some()).collect();
        if active.is_empty() {
            assert!(
                !queue.is_empty(),
                "no active slots and nothing queued but not done"
            );
            continue;
        }

        // DECODE all B slots (inactive rows are harmless: token 0, masked, ignored)
        let mut tok = vec![0i32; bsz];
        let mut pos = vec![0i32; bsz];
        let mut cl = vec![0i32; bsz];
        for s in 0..bsz {
            if slot_req[s].is_some() {
                tok[s] = cur[s];
                pos[s] = clen[s];
                cl[s] = clen[s];
            }
        }
        let mut out = vec![0i32; bsz];
        assert_eq!(
            unsafe {
                xla_llama_decode_ragged(
                    ctx2,
                    bsz as c_int,
                    tok.as_ptr(),
                    pos.as_ptr(),
                    cl.as_ptr(),
                    out.as_mut_ptr(),
                )
            },
            0,
            "decode_ragged failed"
        );
        decode_steps += 1;

        // ADVANCE + EVICT
        for s in 0..bsz {
            if let Some(ri) = slot_req[s] {
                let nt = out[s];
                reqs[ri].stream.push(nt);
                clen[s] += 1;
                cur[s] = nt;
                if finished(&reqs[ri].stream, reqs[ri].cap) {
                    slot_req[s] = None;
                    done += 1;
                }
            }
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    unsafe { xla_llama_free(ctx2) };

    // === compare each request's collected stream to its single-seq reference ===
    let mut all_ok = true;
    let mut total_tokens = 0usize;
    for (i, r) in reqs.iter().enumerate() {
        total_tokens += r.stream.len();
        let ok = r.stream == r.reference;
        all_ok &= ok;
        if !ok {
            let m = r.reference.len().min(r.stream.len());
            let div = (0..m).find(|&j| r.stream[j] != r.reference[j]);
            println!(
                "  req {i} (len {} cap {}): MISMATCH (got {} ref {}, first div {:?})",
                r.len,
                r.cap,
                r.stream.len(),
                r.reference.len(),
                div
            );
        }
    }
    let gen_tokens: usize = reqs.iter().map(|r| r.stream.len()).sum();
    println!(
        "\nscheduler: {iter} iters, {decode_steps} decode steps, {admits} admits, \
         {} requests served",
        reqs.len()
    );
    println!(
        "throughput: {:.2} tok/s ({total_tokens} tokens / {secs:.2}s)",
        gen_tokens as f64 / secs
    );
    println!(
        "RESULT: {}",
        if all_ok {
            "REFERENCE-EXACT PASS (all requests match their single-seq references)"
        } else {
            "MISMATCH"
        }
    );
    println!(
        "SCHED\tdevice={device}\tB={bsz}\tN={nreq}\ttok_s={:.3}\tpass={all_ok}",
        gen_tokens as f64 / secs
    );
    std::process::exit(if all_ok { 0 } else { 1 });
}
