// FFI-gate proof: call the C shim (which uses the IREE runtime C API) to run a
// vmfb, confirming Rust can drive IREE execution on this aarch64 box via the
// prebuilt iree-dist, with no IREE source build. This is the substrate the
// mlxcel-xla backend (issue #449 Phase 3) needs.
use std::ffi::CString;
use std::os::raw::{c_char, c_float, c_int};

unsafe extern "C" {
    fn iree_gate_run_add(
        vmfb_path: *const c_char,
        a: *const c_float,
        b: *const c_float,
        n: c_int,
        out: *mut c_float,
    ) -> c_int;
}

fn main() {
    let vmfb = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "add.vmfb".to_string());
    let cpath = CString::new(vmfb.clone()).expect("vmfb path");
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [10.0f32, 20.0, 30.0, 40.0];
    let mut out = [0.0f32; 4];

    let rc =
        unsafe { iree_gate_run_add(cpath.as_ptr(), a.as_ptr(), b.as_ptr(), 4, out.as_mut_ptr()) };
    assert_eq!(rc, 0, "iree_gate_run_add failed (status {rc}) on {vmfb}");

    let expected = [11.0f32, 22.0, 33.0, 44.0];
    println!("IREE via Rust FFI: a + b = {out:?}");
    assert!(
        out.iter().zip(expected).all(|(g, e)| (g - e).abs() < 1e-5),
        "mismatch: got {out:?} expected {expected:?}"
    );
    println!("FFI GATE: PASS (Rust drove an IREE vmfb on aarch64 via the prebuilt runtime)");
}
