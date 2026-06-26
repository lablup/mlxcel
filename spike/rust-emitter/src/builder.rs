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

pub struct Builder {
    body: String,
    next: usize,
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
        }
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

    /// Unused by decode today; kept for the int4 dequant path (int -> f32).
    #[allow(dead_code)]
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
        self.line(format!("{} = stablehlo.{} {} : {}", r, op, a.name, ty.render()));
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
        let shape: Vec<usize> = x
            .ty
            .shape
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
}
