//! Minimal StableHLO text builder.
//!
//! Tracks SSA value names and their tensor types, and exposes one method per
//! StableHLO op we need. Each method emits one textual line into the function
//! body and returns a typed `Val` handle, so the model code reads like ordinary
//! tensor algebra while the builder owns name generation and type bookkeeping.
//!
//! The op spellings mirror exactly what `jax.export` emitted in
//! `spike/openxla/artifacts/decode_step.stablehlo.mlir`, which `iree-compile`
//! already parses as text, so every form here is known-good for that toolchain.

use std::fmt::Write as _;

/// Tensor type: shape plus element type ("f32" | "i32" | "i1").
#[derive(Clone, Debug)]
pub struct Ty {
    pub shape: Vec<usize>,
    pub elt: &'static str,
}

impl Ty {
    pub fn new(shape: Vec<usize>, elt: &'static str) -> Self {
        Ty { shape, elt }
    }
    pub fn f32(shape: Vec<usize>) -> Self {
        Ty::new(shape, "f32")
    }
    pub fn scalar(elt: &'static str) -> Self {
        Ty::new(vec![], elt)
    }
    pub fn render(&self) -> String {
        if self.shape.is_empty() {
            format!("tensor<{}>", self.elt)
        } else {
            let dims: Vec<String> = self.shape.iter().map(|d| d.to_string()).collect();
            format!("tensor<{}x{}>", dims.join("x"), self.elt)
        }
    }
}

/// A typed SSA value handle (e.g. `%42 : tensor<2048xf32>`).
#[derive(Clone, Debug)]
pub struct Val {
    pub name: String,
    pub ty: Ty,
}

/// Contraction (matmul) input precision. `F32` is the default (unchanged
/// behavior). `F16` / `Bf16` demote the f32 inputs of every `dot_general` to the
/// narrow type while keeping the f32 accumulate and output, so only the matmuls
/// change and the sensitive elementwise ops (norm, softmax, RoPE) stay f32. This
/// is authored in the graph (portable to every IREE target), matching
/// `--iree-global-opt-demote-contraction-inputs-type=f16`. A blanket f32->f16 of
/// the whole program is deliberately NOT done (it regressed norm/softmax/accum).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Precision {
    #[default]
    F32,
    F16,
    Bf16,
}

impl Precision {
    /// The narrow element type to demote contraction inputs to, or `None` for
    /// f32 (no demotion).
    fn dot_elt(self) -> Option<&'static str> {
        match self {
            Precision::F32 => None,
            Precision::F16 => Some("f16"),
            Precision::Bf16 => Some("bf16"),
        }
    }
}

/// The explicit `MLXCEL_XLA_PRECISION` override (`f16` / `bf16`), or `None` when
/// unset / unrecognized (so the per-device default applies).
#[must_use]
pub fn precision_env_override() -> Option<Precision> {
    match std::env::var("MLXCEL_XLA_PRECISION").as_deref() {
        Ok("f16") => Some(Precision::F16),
        Ok("bf16") => Some(Precision::Bf16),
        // `f32` is an explicit override too, so it forces f32 even on a GPU device
        // whose default is f16. Only an unset / unrecognized value falls through
        // to the per-device default.
        Ok("f32") => Some(Precision::F32),
        _ => None,
    }
}

/// The contraction precision, honoring `MLXCEL_XLA_PRECISION` else f32. Used where
/// no HAL device is in scope (e.g. the emitter's byte-exact regression tests).
#[must_use]
pub fn precision_from_env() -> Precision {
    precision_env_override().unwrap_or(Precision::F32)
}

/// The default contraction precision for a HAL device when `MLXCEL_XLA_PRECISION`
/// is unset: `f16` on the GPU devices (`metal`, `cuda`), whose runtimes and the
/// pinned iree-compile handle f16 well and where it is ~2x, and `f32` on the CPU
/// (`local-task` / `local-sync`), where f16 gains little and can round-trip
/// through f32 anyway. CUDA is never *auto*-selected as a device, but if it is
/// the target this is its default precision.
#[must_use]
pub fn default_precision(device: &str) -> Precision {
    if device == "metal" || device == "cuda" {
        Precision::F16
    } else {
        Precision::F32
    }
}

/// The precision to emit for `device`: the `MLXCEL_XLA_PRECISION` override wins on
/// every device; otherwise the per-device default. Read once at graph emission; a
/// different value changes the MLIR, so the vmfb content-hash cache keys each
/// precision separately.
#[must_use]
pub fn resolve_precision(device: &str) -> Precision {
    precision_env_override().unwrap_or_else(|| default_precision(device))
}

pub struct Builder {
    body: String,
    next: usize,
    precision: Precision,
}

fn dims_list(d: &[usize]) -> String {
    let parts: Vec<String> = d.iter().map(|x| x.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

/// f32 -> MLIR scalar hex literal (big-endian bit pattern), exact and unambiguous.
pub fn f32_hex(x: f32) -> String {
    format!("0x{:08X}", x.to_bits())
}

/// f32 slice -> MLIR dense blob hex (little-endian raw bytes), matching JAX output.
pub fn f32_blob(data: &[f32]) -> String {
    let mut s = String::with_capacity(data.len() * 8);
    for &x in data {
        for b in x.to_le_bytes() {
            let _ = write!(s, "{:02X}", b);
        }
    }
    s
}

impl Builder {
    pub fn new() -> Self {
        Builder {
            body: String::new(),
            next: 0,
            precision: Precision::F32,
        }
    }

    /// Set the contraction precision (builder-style). Default is `F32`.
    #[must_use]
    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.precision = precision;
        self
    }

    pub fn body(&self) -> &str {
        &self.body
    }

    fn fresh(&mut self) -> String {
        let n = self.next;
        self.next += 1;
        format!("%{}", n)
    }

    fn line(&mut self, s: String) {
        self.body.push_str("    ");
        self.body.push_str(&s);
        self.body.push('\n');
    }

    /// Construct a handle to an existing func argument (not emitted).
    pub fn arg(idx: usize, ty: Ty) -> Val {
        Val {
            name: format!("%arg{}", idx),
            ty,
        }
    }

    // --- constants ---------------------------------------------------------

    pub fn const_f32(&mut self, x: f32) -> Val {
        let r = self.fresh();
        self.line(format!(
            "{} = stablehlo.constant dense<{}> : tensor<f32>",
            r,
            f32_hex(x)
        ));
        Val {
            name: r,
            ty: Ty::scalar("f32"),
        }
    }

    pub fn const_i32(&mut self, v: i32) -> Val {
        let r = self.fresh();
        self.line(format!(
            "{} = stablehlo.constant dense<{}> : tensor<i32>",
            r, v
        ));
        Val {
            name: r,
            ty: Ty::scalar("i32"),
        }
    }

    /// Dense f32 constant tensor from raw data (row-major), emitted as hex blob.
    pub fn const_tensor_f32(&mut self, data: &[f32], shape: Vec<usize>) -> Val {
        let ty = Ty::f32(shape);
        let r = self.fresh();
        self.line(format!(
            "{} = stablehlo.constant dense<\"0x{}\"> : {}",
            r,
            f32_blob(data),
            ty.render()
        ));
        Val { name: r, ty }
    }

    // --- structural --------------------------------------------------------

    pub fn iota(&mut self, n: usize) -> Val {
        let ty = Ty::new(vec![n], "i32");
        let r = self.fresh();
        self.line(format!("{} = stablehlo.iota dim = 0 : {}", r, ty.render()));
        Val { name: r, ty }
    }

    pub fn reshape(&mut self, x: &Val, shape: Vec<usize>) -> Val {
        let ty = Ty::new(shape, x.ty.elt);
        let r = self.fresh();
        self.line(format!(
            "{} = stablehlo.reshape {} : ({}) -> {}",
            r,
            x.name,
            x.ty.render(),
            ty.render()
        ));
        Val { name: r, ty }
    }

    pub fn broadcast(&mut self, x: &Val, dims: &[usize], out_shape: Vec<usize>) -> Val {
        let ty = Ty::new(out_shape, x.ty.elt);
        let r = self.fresh();
        self.line(format!(
            "{} = stablehlo.broadcast_in_dim {}, dims = {} : ({}) -> {}",
            r,
            x.name,
            dims_list(dims),
            x.ty.render(),
            ty.render()
        ));
        Val { name: r, ty }
    }

    /// Permute axes. `perm[k]` is the input axis that becomes output axis k.
    pub fn transpose(&mut self, x: &Val, perm: &[usize]) -> Val {
        let shape: Vec<usize> = perm.iter().map(|&p| x.ty.shape[p]).collect();
        let ty = Ty::new(shape, x.ty.elt);
        let r = self.fresh();
        self.line(format!(
            "{} = stablehlo.transpose {}, dims = {} : ({}) -> {}",
            r,
            x.name,
            dims_list(perm),
            x.ty.render(),
            ty.render()
        ));
        Val { name: r, ty }
    }

    /// Element-type conversion (`stablehlo.convert`), preserving shape. Used by
    /// the low-precision contraction path (f32 -> f16/bf16) and reserved for the
    /// int dequant path.
    pub fn convert(&mut self, x: &Val, elt: &'static str) -> Val {
        let ty = Ty::new(x.ty.shape.clone(), elt);
        let r = self.fresh();
        self.line(format!(
            "{} = stablehlo.convert {} : ({}) -> {}",
            r,
            x.name,
            x.ty.render(),
            ty.render()
        ));
        Val { name: r, ty }
    }

    /// Static slice. `ranges` is (start, limit) per dim, stride 1.
    pub fn slice(&mut self, x: &Val, ranges: &[(usize, usize)]) -> Val {
        let shape: Vec<usize> = ranges.iter().map(|(s, l)| l - s).collect();
        let ty = Ty::new(shape, x.ty.elt);
        let parts: Vec<String> = ranges.iter().map(|(s, l)| format!("{}:{}", s, l)).collect();
        let r = self.fresh();
        self.line(format!(
            "{} = stablehlo.slice {} [{}] : ({}) -> {}",
            r,
            x.name,
            parts.join(", "),
            x.ty.render(),
            ty.render()
        ));
        Val { name: r, ty }
    }

    pub fn concatenate(&mut self, a: &Val, b: &Val, dim: usize) -> Val {
        let mut shape = a.ty.shape.clone();
        shape[dim] = a.ty.shape[dim] + b.ty.shape[dim];
        let ty = Ty::new(shape, a.ty.elt);
        let r = self.fresh();
        self.line(format!(
            "{} = stablehlo.concatenate {}, {}, dim = {} : ({}, {}) -> {}",
            r,
            a.name,
            b.name,
            dim,
            a.ty.render(),
            b.ty.render(),
            ty.render()
        ));
        Val { name: r, ty }
    }

    pub fn dynamic_slice(&mut self, x: &Val, starts: &[&Val], sizes: Vec<usize>) -> Val {
        let ty = Ty::new(sizes.clone(), x.ty.elt);
        let names: Vec<String> = starts.iter().map(|v| v.name.clone()).collect();
        let idx_tys: Vec<String> = starts.iter().map(|v| v.ty.render()).collect();
        let sz: Vec<String> = sizes.iter().map(|s| s.to_string()).collect();
        let r = self.fresh();
        self.line(format!(
            "{} = stablehlo.dynamic_slice {}, {}, sizes = [{}] : ({}, {}) -> {}",
            r,
            x.name,
            names.join(", "),
            sz.join(", "),
            x.ty.render(),
            idx_tys.join(", "),
            ty.render()
        ));
        Val { name: r, ty }
    }

    pub fn dynamic_update_slice(&mut self, operand: &Val, update: &Val, starts: &[&Val]) -> Val {
        let ty = operand.ty.clone();
        let names: Vec<String> = starts.iter().map(|v| v.name.clone()).collect();
        let idx_tys: Vec<String> = starts.iter().map(|v| v.ty.render()).collect();
        let r = self.fresh();
        self.line(format!(
            "{} = stablehlo.dynamic_update_slice {}, {}, {} : ({}, {}, {}) -> {}",
            r,
            operand.name,
            update.name,
            names.join(", "),
            operand.ty.render(),
            update.ty.render(),
            idx_tys.join(", "),
            ty.render()
        ));
        Val { name: r, ty }
    }

    /// Row gather: `operand[N, M]` indexed by `indices[Lp, 1]` (i32) -> `[Lp, M]`.
    /// This is the multi-token embedding lookup `embed[tokens]`; the
    /// dimension_numbers and slice_sizes mirror the JAX-emitted `prefill`
    /// `stablehlo.gather` in spike/openxla/artifacts/prefill.stablehlo.mlir.
    pub fn gather(&mut self, operand: &Val, indices: &Val) -> Val {
        let lp = indices.ty.shape[0];
        let m = operand.ty.shape[1];
        let ty = Ty::new(vec![lp, m], operand.ty.elt);
        let r = self.fresh();
        self.line(format!(
            "{} = \"stablehlo.gather\"({}, {}) <{{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, {}>}}> : ({}, {}) -> {}",
            r,
            operand.name,
            indices.name,
            m,
            operand.ty.render(),
            indices.ty.render(),
            ty.render()
        ));
        Val { name: r, ty }
    }

    // --- elementwise -------------------------------------------------------

    fn binary(&mut self, op: &str, a: &Val, b: &Val) -> Val {
        let ty = a.ty.clone();
        let r = self.fresh();
        self.line(format!(
            "{} = stablehlo.{} {}, {} : {}",
            r,
            op,
            a.name,
            b.name,
            ty.render()
        ));
        Val { name: r, ty }
    }

    pub fn add(&mut self, a: &Val, b: &Val) -> Val {
        self.binary("add", a, b)
    }
    pub fn subtract(&mut self, a: &Val, b: &Val) -> Val {
        self.binary("subtract", a, b)
    }
    pub fn multiply(&mut self, a: &Val, b: &Val) -> Val {
        self.binary("multiply", a, b)
    }
    pub fn divide(&mut self, a: &Val, b: &Val) -> Val {
        self.binary("divide", a, b)
    }

    fn unary(&mut self, op: &str, a: &Val) -> Val {
        let ty = a.ty.clone();
        let r = self.fresh();
        self.line(format!(
            "{} = stablehlo.{} {} : {}",
            r,
            op,
            a.name,
            ty.render()
        ));
        Val { name: r, ty }
    }

    pub fn negate(&mut self, a: &Val) -> Val {
        self.unary("negate", a)
    }
    pub fn exponential(&mut self, a: &Val) -> Val {
        self.unary("exponential", a)
    }
    pub fn rsqrt(&mut self, a: &Val) -> Val {
        self.unary("rsqrt", a)
    }
    pub fn tanh(&mut self, a: &Val) -> Val {
        self.unary("tanh", a)
    }

    /// `compare DIR, a, b, SIGNED|FLOAT` -> i1 tensor of the same shape.
    pub fn compare(&mut self, dir: &str, a: &Val, b: &Val, kind: &str) -> Val {
        let ty = Ty::new(a.ty.shape.clone(), "i1");
        let r = self.fresh();
        self.line(format!(
            "{} = stablehlo.compare {}, {}, {}, {} : ({}, {}) -> {}",
            r,
            dir,
            a.name,
            b.name,
            kind,
            a.ty.render(),
            b.ty.render(),
            ty.render()
        ));
        Val { name: r, ty }
    }

    pub fn select(&mut self, pred: &Val, a: &Val, b: &Val) -> Val {
        let ty = a.ty.clone();
        let r = self.fresh();
        self.line(format!(
            "{} = stablehlo.select {}, {}, {} : {}, {}",
            r,
            pred.name,
            a.name,
            b.name,
            pred.ty.render(),
            ty.render()
        ));
        Val { name: r, ty }
    }

    // --- reductions --------------------------------------------------------

    fn reduce(&mut self, applies: &str, x: &Val, dim: usize, init: &Val) -> Val {
        let shape: Vec<usize> =
            x.ty.shape
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != dim)
                .map(|(_, d)| *d)
                .collect();
        let ty = Ty::new(shape, x.ty.elt);
        let r = self.fresh();
        self.line(format!(
            "{} = stablehlo.reduce({} init: {}) applies stablehlo.{} across dimensions = [{}] : ({}, {}) -> {}",
            r, x.name, init.name, applies, dim, x.ty.render(), init.ty.render(), ty.render()
        ));
        Val { name: r, ty }
    }

    pub fn reduce_add(&mut self, x: &Val, dim: usize, init: &Val) -> Val {
        self.reduce("add", x, dim, init)
    }
    pub fn reduce_max(&mut self, x: &Val, dim: usize, init: &Val) -> Val {
        self.reduce("maximum", x, dim, init)
    }

    /// On-device argmax over a `[V]` vector -> scalar `i32` index (the first
    /// index of the max, numpy/jax semantics). Mirrors the JAX/IREE-emitted
    /// argmax reducer in `spike/openxla/artifacts/fp32_decode_argmax.stablehlo.mlir`:
    /// a two-operand `stablehlo.reduce` over (values, iota indices) whose body
    /// keeps the larger value (or NaN), tie-breaking to the lower index. This is
    /// the Phase 2b on-device sampling: the graph returns a token id, so a decode
    /// step ships 4 bytes back instead of a `[V]` logits copy. The reducer block
    /// args are named (not `%argN` / `%N`) so they never collide with the
    /// function args or the builder's SSA counter.
    pub fn argmax(&mut self, logits: &Val) -> Val {
        let v = logits.ty.shape[0];
        let vf = Ty::f32(vec![v]).render();
        let vi = Ty::new(vec![v], "i32").render();
        let c0 = self.fresh();
        self.line(format!("{c0} = stablehlo.constant dense<0> : tensor<i32>"));
        let ninf = self.fresh();
        self.line(format!(
            "{ninf} = stablehlo.constant dense<0xFF800000> : tensor<f32>"
        ));
        let iota = self.fresh();
        self.line(format!("{iota} = stablehlo.iota dim = 0 : {vi}"));
        let res = self.fresh();
        self.line(format!(
            "{res}:2 = stablehlo.reduce({l} init: {ninf}), ({iota} init: {c0}) across dimensions = [0] : ({vf}, {vi}, tensor<f32>, tensor<i32>) -> (tensor<f32>, tensor<i32>)",
            l = logits.name
        ));
        self.argmax_reducer_block();
        Val {
            name: format!("{res}#1"),
            ty: Ty::scalar("i32"),
        }
    }

    /// Batched on-device argmax over `[B, V]` -> `[B] i32`: the per-row argmax
    /// index (first-max, numpy/jax tie-break). Same reducer as `argmax`, here
    /// reducing the last axis of a 2-D logits tensor; the index iota is `[B, V]`
    /// with each row `0..V-1` (`iota dim = 1`). Returns the index result (`#1`)
    /// as a `[B] i32`. This is the batched-decode sampling tail: each step ships
    /// `B` token ids (`4*B` bytes) back instead of a `[B, V]` logits copy.
    pub fn argmax_batched(&mut self, logits: &Val) -> Val {
        let bsz = logits.ty.shape[0];
        let v = logits.ty.shape[1];
        let bvf = Ty::f32(vec![bsz, v]).render();
        let bvi = Ty::new(vec![bsz, v], "i32").render();
        let bf = Ty::f32(vec![bsz]).render();
        let bi = Ty::new(vec![bsz], "i32").render();
        let c0 = self.fresh();
        self.line(format!("{c0} = stablehlo.constant dense<0> : tensor<i32>"));
        let ninf = self.fresh();
        self.line(format!(
            "{ninf} = stablehlo.constant dense<0xFF800000> : tensor<f32>"
        ));
        let iota = self.fresh();
        self.line(format!("{iota} = stablehlo.iota dim = 1 : {bvi}"));
        let res = self.fresh();
        self.line(format!(
            "{res}:2 = stablehlo.reduce({l} init: {ninf}), ({iota} init: {c0}) across dimensions = [1] : ({bvf}, {bvi}, tensor<f32>, tensor<i32>) -> ({bf}, {bi})",
            l = logits.name
        ));
        self.argmax_reducer_block();
        Val {
            name: format!("{res}#1"),
            ty: Ty::new(vec![bsz], "i32"),
        }
    }

    /// The shared `stablehlo.reduce` reducer region for argmax (scalar or
    /// batched): keep the larger value (NaN-propagating), tie-break to the lower
    /// index. The block operates on scalars regardless of the reduced rank, so
    /// the same body serves both. Block args are named (not `%argN`/`%N`) so they
    /// never collide with the function args or the builder's SSA counter.
    fn argmax_reducer_block(&mut self) {
        self.line(
            "reducer(%amv_l: tensor<f32>, %amv_r: tensor<f32>) (%ami_l: tensor<i32>, %ami_r: tensor<i32>) {"
                .to_string(),
        );
        let gt = self.fresh();
        self.line(format!(
            "  {gt} = stablehlo.compare GT, %amv_l, %amv_r, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>"
        ));
        let nan = self.fresh();
        self.line(format!(
            "  {nan} = stablehlo.compare NE, %amv_l, %amv_l, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>"
        ));
        let gt_nan = self.fresh();
        self.line(format!(
            "  {gt_nan} = stablehlo.or {gt}, {nan} : tensor<i1>"
        ));
        let eq = self.fresh();
        self.line(format!(
            "  {eq} = stablehlo.compare EQ, %amv_l, %amv_r, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>"
        ));
        let lt = self.fresh();
        self.line(format!(
            "  {lt} = stablehlo.compare LT, %ami_l, %ami_r, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>"
        ));
        let eq_lt = self.fresh();
        self.line(format!("  {eq_lt} = stablehlo.and {eq}, {lt} : tensor<i1>"));
        let idx_pred = self.fresh();
        self.line(format!(
            "  {idx_pred} = stablehlo.or {gt_nan}, {eq_lt} : tensor<i1>"
        ));
        let mv = self.fresh();
        self.line(format!(
            "  {mv} = stablehlo.select {gt_nan}, %amv_l, %amv_r : tensor<i1>, tensor<f32>"
        ));
        let mi = self.fresh();
        self.line(format!(
            "  {mi} = stablehlo.select {idx_pred}, %ami_l, %ami_r : tensor<i1>, tensor<i32>"
        ));
        self.line(format!(
            "  stablehlo.return {mv}, {mi} : tensor<f32>, tensor<i32>"
        ));
        self.line("}".to_string());
    }

    // --- dot_general -------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    pub fn dot_general(
        &mut self,
        lhs: &Val,
        rhs: &Val,
        lhs_batch: &[usize],
        rhs_batch: &[usize],
        lhs_contract: &[usize],
        rhs_contract: &[usize],
        out_shape: Vec<usize>,
    ) -> Val {
        // Low-precision mode: demote the f32 contraction inputs to the narrow
        // type, keeping the f32 accumulate + output (`ty` below stays f32). Only
        // matmuls change; norm/softmax/RoPE stay f32. No-op under `Precision::F32`.
        let lhs_demoted;
        let rhs_demoted;
        let (lhs, rhs) = match self.precision.dot_elt() {
            Some(elt) => {
                lhs_demoted = if lhs.ty.elt == "f32" {
                    self.convert(lhs, elt)
                } else {
                    lhs.clone()
                };
                rhs_demoted = if rhs.ty.elt == "f32" {
                    self.convert(rhs, elt)
                } else {
                    rhs.clone()
                };
                (&lhs_demoted, &rhs_demoted)
            }
            None => (lhs, rhs),
        };
        let ty = Ty::new(out_shape, "f32");
        let r = self.fresh();
        let batch = if lhs_batch.is_empty() && rhs_batch.is_empty() {
            String::new()
        } else {
            format!(
                "batching_dims = {} x {}, ",
                dims_list(lhs_batch),
                dims_list(rhs_batch)
            )
        };
        self.line(format!(
            "{} = stablehlo.dot_general {}, {}, {}contracting_dims = {} x {} : ({}, {}) -> {}",
            r,
            lhs.name,
            rhs.name,
            batch,
            dims_list(lhs_contract),
            dims_list(rhs_contract),
            lhs.ty.render(),
            rhs.ty.render(),
            ty.render()
        ));
        Val { name: r, ty }
    }

    /// Convenience: y = x @ W^T for x:[K], W:[N,K] (weights stored [out,in]).
    /// Contracts x dim 0 with W dim 1, yielding [N]. No transpose needed.
    pub fn linear(&mut self, x: &Val, w: &Val) -> Val {
        let n = w.ty.shape[0];
        self.dot_general(x, w, &[], &[], &[0], &[1], vec![n])
    }

    /// Sequence-batched linear: y[L,N] = x[L,K] @ W^T for W:[N,K] (stored
    /// [out,in]). Contracts the feature dim (x dim 1, W dim 1), keeps the [L]
    /// row axis. The prefill analog of `linear` for `[Lp, ...]` activations.
    pub fn linear_seq(&mut self, x: &Val, w: &Val) -> Val {
        let l = x.ty.shape[0];
        let n = w.ty.shape[0];
        self.dot_general(x, w, &[], &[], &[1], &[1], vec![l, n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_dot_emits_no_convert() {
        let mut b = Builder::new(); // default Precision::F32
        let x = Builder::arg(0, Ty::f32(vec![2048]));
        let w = Builder::arg(1, Ty::f32(vec![512, 2048]));
        let _ = b.linear(&x, &w);
        assert!(!b.body().contains("stablehlo.convert"));
        assert!(b.body().contains("stablehlo.dot_general"));
    }

    #[test]
    fn f16_dot_demotes_inputs_and_keeps_f32_accumulate() {
        let mut b = Builder::new().with_precision(Precision::F16);
        let x = Builder::arg(0, Ty::f32(vec![2048]));
        let w = Builder::arg(1, Ty::f32(vec![512, 2048]));
        let y = b.linear(&x, &w);
        let body = b.body();
        // Both f32 operands are demoted to f16 (two converts).
        assert_eq!(body.matches("stablehlo.convert").count(), 2);
        assert!(body.contains("-> tensor<2048xf16>"));
        assert!(body.contains("-> tensor<512x2048xf16>"));
        // The dot consumes f16 inputs but accumulates/outputs f32.
        assert!(body.contains("(tensor<2048xf16>, tensor<512x2048xf16>) -> tensor<512xf32>"));
        assert_eq!(y.ty.elt, "f32");
    }

    #[test]
    fn bf16_dot_demotes_to_bf16() {
        let mut b = Builder::new().with_precision(Precision::Bf16);
        let x = Builder::arg(0, Ty::f32(vec![8]));
        let w = Builder::arg(1, Ty::f32(vec![4, 8]));
        let _ = b.linear(&x, &w);
        assert!(b.body().contains("-> tensor<8xbf16>"));
        assert!(
            b.body()
                .contains("(tensor<8xbf16>, tensor<4x8xbf16>) -> tensor<4xf32>")
        );
    }

    #[test]
    fn default_precision_is_f16_on_gpu_and_f32_on_cpu() {
        assert_eq!(default_precision("metal"), Precision::F16);
        assert_eq!(default_precision("cuda"), Precision::F16);
        assert_eq!(default_precision("local-task"), Precision::F32);
        assert_eq!(default_precision("local-sync"), Precision::F32);
    }
}
