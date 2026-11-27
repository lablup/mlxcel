// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Build-time gate and artifact staging for the experimental cuda-oxide fused
//! residual-add + RMSNorm route (issue #2833).
//!
//! The switch is the build environment variable
//! `MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM`, not a Cargo feature and not a build
//! dependency. Both of those feed the build-script unit's metadata hash, which
//! names `mlxcel-core`'s `OUT_DIR`, so either would strand every warm MLX
//! build under the old hash and rebuild MLX from source (the reason
//! `ptx_cache_test_hooks.rs` reads a profile directory and `mlxcel-mlx-pin`
//! includes these files by path). An environment variable re-runs the build
//! script in place: CMake finds MLX already built and only the bridge is
//! recompiled.
//!
//! That is also why the SHA-256 below is written out instead of taken from the
//! `sha2` crate, and why the manifest is searched for two fields instead of
//! parsed with `serde_json`. The check is the build-time half only: it refuses
//! to embed a PTX whose bytes are not the ones its manifests pin. The runtime
//! validator in `src/cuda_oxide_fused_norm.rs` parses the manifests and
//! schemas in full before the route is ever taken.
//!
//! Included by path from `build.rs` and from `mlxcel-mlx-pin`, which runs the
//! unit tests without building MLX.

use std::fs;
use std::path::{Path, PathBuf};

/// The build environment variable that opts a build into the adapter.
pub const BUILD_ENV: &str = "MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM";

/// The PTX module, relative to the spike package's `artifacts/` directory.
pub const PTX_FILE: &str = "fused_add_rms_norm.sm_121.ptx";

/// The five entry points of the module, in manifest order.
pub const ENTRIES: &[&str] = &[
    "fused_add_rms_norm_bf16",
    "fused_add_rms_norm_f16",
    "fused_add_rms_norm_f32",
    "fused_add_rms_norm_f16_wf32",
    "fused_add_rms_norm_bf16_wf32",
];

/// File name of an entry's manifest under `artifacts/`.
pub fn manifest_file(entry: &str) -> String {
    format!("{entry}.sm_121.manifest.json")
}

/// File name of an entry's ABI schema under `abi/`.
pub fn schema_file(entry: &str) -> String {
    format!("{entry}.schema.json")
}

/// Whether the build asked for the adapter. Unset, empty and the usual "off"
/// spellings are off; the usual "on" spellings are on; anything else is an
/// error, so a typo cannot silently build without the adapter.
pub fn requested(value: Option<&str>) -> Result<bool, String> {
    let Some(value) = value else {
        return Ok(false);
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "0" | "false" | "off" | "no" => Ok(false),
        "1" | "true" | "on" | "yes" => Ok(true),
        other => Err(format!(
            "{BUILD_ENV}={other:?} is not a boolean; use 1/0, on/off, true/false or yes/no"
        )),
    }
}

/// The adapter exists only on a Linux CUDA build. Asking for it anywhere else
/// fails the build instead of producing a binary that quietly lacks it.
pub fn check_supported(target_os: &str, cuda: bool) -> Result<(), String> {
    if target_os != "linux" || !cuda {
        return Err(format!(
            "{BUILD_ENV} is set, but the cuda-oxide fused RMSNorm adapter only builds on Linux with the `cuda` feature (target_os={target_os}, cuda={cuda}). Unset it for this build."
        ));
    }
    Ok(())
}

/// SHA-256 of `data` as lowercase hex (FIPS 180-4).
pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());

    for block in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [
                t1.wrapping_add(t2),
                v[0],
                v[1],
                v[2],
                v[3].wrapping_add(t1),
                v[4],
                v[5],
                v[6],
            ];
        }
        for (state, value) in h.iter_mut().zip(v) {
            *state = state.wrapping_add(value);
        }
    }
    h.iter().map(|word| format!("{word:08x}")).collect()
}

/// The string value of `key`, searched for after each of `anchors` in turn.
///
/// Not a JSON parser: it finds `"anchor"` occurrences in order, then the first
/// `"key": "value"` after the last one. That is enough to read the two digests
/// below out of the generated manifests, and a manifest it misreads fails the
/// comparison rather than passing it.
pub fn string_field_after<'a>(text: &'a str, anchors: &[&str], key: &str) -> Option<&'a str> {
    let mut rest = text;
    for anchor in anchors {
        let needle = format!("\"{anchor}\"");
        rest = &rest[rest.find(&needle)? + needle.len()..];
    }
    let needle = format!("\"{key}\"");
    rest = &rest[rest.find(&needle)? + needle.len()..];
    let rest = rest.trim_start().strip_prefix(':')?.trim_start();
    let rest = rest.strip_prefix('"')?;
    Some(&rest[..rest.find('"')?])
}

/// Check that `manifest` pins exactly `ptx_sha256`, both as the container
/// payload and as the PTX artifact.
pub fn check_manifest_digest(
    manifest_name: &str,
    manifest: &str,
    ptx_sha256: &str,
) -> Result<(), String> {
    for (what, anchors, key) in [
        (
            "container.payload_sha256",
            &["container"][..],
            "payload_sha256",
        ),
        ("artifacts.ptx.sha256", &["artifacts", "ptx"][..], "sha256"),
    ] {
        let pinned = string_field_after(manifest, anchors, key)
            .ok_or_else(|| format!("{manifest_name}: no {what} field"))?;
        if pinned != ptx_sha256 {
            return Err(format!(
                "{manifest_name}: {what} is {pinned}, but {PTX_FILE} hashes to {ptx_sha256}"
            ));
        }
    }
    Ok(())
}

/// Verify the committed PTX against all five manifests and copy the PTX, the
/// manifests and the schemas into `stage_dir`, flat, for `include_bytes!`.
///
/// Returns the source files read, for `cargo:rerun-if-changed`. Nothing is
/// staged unless every manifest agrees with the PTX on disk.
pub fn verify_and_stage(spike_dir: &Path, stage_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let read = |path: &Path| {
        fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))
    };
    let ptx_path = spike_dir.join("artifacts").join(PTX_FILE);
    let ptx = read(&ptx_path)?;
    let digest = sha256_hex(&ptx);

    let mut sources = vec![ptx_path];
    let mut staged = vec![(PTX_FILE.to_string(), ptx)];
    for entry in ENTRIES {
        let manifest_name = manifest_file(entry);
        let manifest_path = spike_dir.join("artifacts").join(&manifest_name);
        let manifest = read(&manifest_path)?;
        let text =
            std::str::from_utf8(&manifest).map_err(|_| format!("{manifest_name}: not UTF-8"))?;
        check_manifest_digest(&manifest_name, text, &digest)?;

        let schema_name = schema_file(entry);
        let schema_path = spike_dir.join("abi").join(&schema_name);
        let schema = read(&schema_path)?;

        sources.push(manifest_path);
        sources.push(schema_path);
        staged.push((manifest_name, manifest));
        staged.push((schema_name, schema));
    }

    fs::create_dir_all(stage_dir)
        .map_err(|error| format!("cannot create {}: {error}", stage_dir.display()))?;
    for (name, bytes) in staged {
        let path = stage_dir.join(name);
        fs::write(&path, bytes)
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    }
    Ok(sources)
}

#[cfg(test)]
#[path = "cuda_oxide_fused_norm_tests.rs"]
mod tests;
