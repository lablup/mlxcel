#!/usr/bin/env bash
# Deterministic coverage for scripts/ci/check_used_by.py (#1141).
#
# The checker is green on the real tree, and a rule that is only ever run
# against a tree satisfying it is indistinguishable from no rule. So this test
# writes a small synthetic workspace (a root crate, an `mlxcel-core` crate with
# `utils.rs` and `layers.rs`, and a set of model files), then mutates a copy the
# way a real change would and asserts the checker rejects it, or, for the
# positive controls, still accepts it.
#
# The cases the issue asks for are all here: an absent annotation above the
# threshold, the stale roster (the pre-#1121 `create_causal_mask` text,
# verbatim), correct annotations in both forms, and the `softcap`
# disambiguation, where every decoy is one a substring match would count.
#
# The tree is passed through --root. No build, toolchain or network is needed.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
under_test="$script_dir/check_used_by.py"

work="$(mktemp -d "${TMPDIR:-/tmp}/used-by-test.XXXXXX")"
trap 'rm -rf "$work"' EXIT

failures=0

# Write the base workspace into $work/$1 and print its path.
#
# Fourteen model files call `create_causal_mask`, each in a different style
# (named import, brace group, `use ... as`, module import, glob, full path), so
# the resolver is exercised on every form it claims to read. Thirteen of them
# call `prefill_causal_mask`, whose annotation is a complete roster. All
# fourteen call `layers::attention_from_ptr`, whose annotation is a rule with a
# regenerating command and no count. Six call the three grandfathered,
# unannotated helpers. One calls `softcap`; the rest of the `softcap` text in
# the tree is decoys.
make_tree() {
  local dir="$work/$1"
  mkdir -p "$dir" || return 1
  python3 - "$dir" <<'PY' || return 1
import pathlib
import sys

root = pathlib.Path(sys.argv[1])


def write(rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


write("Cargo.toml", '[package]\nname = "mlxcel"\nversion = "0.0.0"\n')
write("src/lib.rs", "pub mod models;\n")
write(
    "src/lib/mlxcel-core/Cargo.toml",
    '[package]\nname = "mlxcel-core"\nversion = "0.0.0"\n',
)
write(
    "src/lib/mlxcel-core/src/lib.rs",
    """\
mod ffi {
    pub fn silu(x: f32) -> f32 {
        x
    }
    pub fn compiled_softcap(x: f32, cap: f32) -> f32 {
        x.min(cap)
    }
}

// The raw FFI functions are re-exported at the crate root, so
// `mlxcel_core::silu` is not `mlxcel_core::utils::silu`.
pub use ffi::*;

pub mod layers;
pub mod utils;
""",
)
write(
    "src/lib/mlxcel-core/src/utils.rs",
    """\
use crate::ffi;

pub fn slice_axis(x: f32) -> f32 {
    x
}

/// Create a causal attention mask.
///
/// Used by: decoders that materialize an explicit prefill mask instead of
/// leaving `mask: None` for fused SDPA. At this commit that is 14 non-test
/// files under `src/models`.
///
/// Not used by the mainstream dense decoders (Llama3, Mixtral, Phi and
/// similar): they pass `mask: None`.
///
/// Regenerate the exact list with
/// `grep -rln '\\bcreate_causal_mask(' src/models --include='*.rs' | grep -v 'tests\\.rs$'`.
pub fn create_causal_mask(size: i32, offset: i32) -> f32 {
    (size + offset) as f32
}

/// Used by: NemotronH, Lfm2, Jamba, FalconH1, Gemma3, Gemma 4, Qwen 3.5,
/// MiniMaxM3, RecurrentGemma, DeepSeekV2, GLM4V, HunyuanVL and Llama4.
pub fn prefill_causal_mask(size: i32) -> f32 {
    size as f32
}

/// SiLU through the raw FFI entry point.
#[inline]
pub fn silu(x: f32) -> f32 {
    ffi::silu(x)
}

/// Softcap function for Gemma 2/3 attention and logits.
///
/// Used by: RecurrentGemma (`src/models/recurrent_gemma.rs`) for final logit
/// softcapping. Gemma 2 applies the same math through `compiled_softcap`.
pub fn softcap(x: f32, cap: f32) -> f32 {
    (x / cap).tanh() * cap
}

pub fn stack_arrays(x: f32) -> f32 {
    x
}

/// Used by: DeepSeekV2 and Llama4.
pub fn repeat_kv(x: f32) -> f32 {
    x
}
""",
)
write(
    "src/lib/mlxcel-core/src/layers.rs",
    """\
/// Pointer-friendly attention wrapper.
///
/// Used by: model attention call sites that still store masks as raw pointers.
/// The caller set is too large to enumerate by name without going stale.
/// Regenerate the exact list with
/// `grep -rln '\\battention_from_ptr(' src --include='*.rs' | grep -v 'tests\\.rs$'`.
///
/// # Safety
///
/// `x` must be finite.
#[allow(
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]
pub unsafe fn attention_from_ptr(x: f32) -> f32 {
    x
}
""",
)

# (file stem, how it reaches `create_causal_mask`)
styles = [
    ("nemotron_h", "named"),
    ("lfm2", "brace"),
    ("jamba", "alias"),
    ("falcon_h1", "module"),
    ("plamo2", "glob"),
    ("gemma3", "path"),
    ("gemma4", "named"),
    ("qwen3_5", "self_group"),
    ("minimax_m3", "crate_alias"),
    ("recurrent_gemma", "named"),
    ("deepseek_v2", "path"),
    ("glm4v", "module"),
    ("hunyuan_vl", "brace"),
    ("llama4", "named"),
]
imports = {
    "named": "use mlxcel_core::utils::create_causal_mask;\n",
    "brace": "use mlxcel_core::{\n    layers,\n    utils::{create_causal_mask, prefill_causal_mask as pcm},\n};\n",
    "alias": "use mlxcel_core::utils::{create_causal_mask, prefill_causal_mask as prefill};\n",
    "module": "use mlxcel_core::utils;\n",
    "glob": "use mlxcel_core::utils::*;\n",
    "path": "",
    "self_group": "use mlxcel_core::utils::{self, create_causal_mask};\n",
    "crate_alias": "use mlxcel_core as mx;\n",
}
calls = {
    "named": "create_causal_mask(4, 0)",
    "brace": "create_causal_mask(4, 0)",
    "alias": "create_causal_mask(4, 0)",
    "module": "utils::create_causal_mask(4, 0)",
    "glob": "create_causal_mask(4, 0)",
    "path": "mlxcel_core::utils::create_causal_mask(4, 0)",
    "self_group": "create_causal_mask(4, 0)",
    "crate_alias": "mx::utils::create_causal_mask(4, 0)",
}
mods = []
for index, (stem, style) in enumerate(styles):
    body = imports[style] + "\npub fn forward() -> f32 {\n"
    body += f"    let mask = {calls[style]};\n"
    if stem != "plamo2":
        prefill = {"brace": "pcm(4)", "alias": "prefill(4)"}.get(
            style, "mlxcel_core::utils::prefill_causal_mask(4)"
        )
        body += f"    let mask = mask + {prefill};\n"
    if index < 6:
        body += "    let mask = mlxcel_core::utils::slice_axis(mask);\n"
        body += "    let mask = mlxcel_core::utils::silu(mask);\n"
        body += "    let mask = mlxcel_core::utils::stack_arrays(mask);\n"
    if stem in ("deepseek_v2", "llama4"):
        body += "    let mask = mlxcel_core::utils::repeat_kv(mask);\n"
    if stem == "recurrent_gemma":
        body = "use mlxcel_core::utils::softcap;\n" + body
        body += "    let mask = softcap(mask, 30.0);\n"
    body += "    unsafe { mlxcel_core::layers::attention_from_ptr(mask) }\n}\n"
    if stem == "recurrent_gemma":
        body += '\n#[cfg(test)]\n#[path = "recurrent_gemma_tests.rs"]\nmod tests;\n'
    if stem == "gemma3":
        # A test module is not a caller, whatever it imports.
        body += (
            "\n#[cfg(test)]\nmod mask_tests {\n"
            "    use mlxcel_core::utils::softcap;\n\n"
            "    #[test]\n    fn caps() {\n        assert!(softcap(1.0, 2.0) < 2.0);\n    }\n}\n"
        )
    write(f"src/models/{stem}.rs", body)
    mods.append(stem)

# A file declared under `#[cfg(test)]` is test code by that declaration.
write(
    "src/models/recurrent_gemma_tests.rs",
    "use mlxcel_core::utils::{create_causal_mask, softcap};\n\n"
    "#[test]\nfn softcap_bounds_logits() {\n"
    "    assert!(softcap(create_causal_mask(1, 0), 2.0) < 2.0);\n}\n",
)

# Families the pre-#1121 roster named. They pass `mask: None` and never call
# `create_causal_mask`; the comment is a mention, not a call.
for stem in ("llama3", "mixtral", "phi", "cohere", "starcoder2"):
    write(
        f"src/models/{stem}.rs",
        "// Fused SDPA applies causality itself, so there is no explicit mask\n"
        "// and nothing here goes through the shared mask builder.\n"
        "pub fn forward() -> Option<f32> {\n    let mask: Option<f32> = None;\n    mask\n}\n",
    )
    mods.append(stem)

# `softcap` decoys. Each one is a file a substring or bare-identifier match
# would count as a caller of `utils.rs::softcap`; none of them is.
write(
    "src/models/emitter_model.rs",
    """\
// A private function of the same name, under a glob import of the module
// that defines the real one. The local `fn` shadows the glob.
use mlxcel_core::utils::*;

fn softcap(x: f32, cap: f32) -> f32 {
    x.min(cap)
}

pub fn emit() -> f32 {
    softcap(slice_axis_free(1.0), 2.0)
}

fn slice_axis_free(x: f32) -> f32 {
    x
}
""",
)
write(
    "src/models/muse_glimmer.rs",
    """\
pub fn softcap_logits(logits: f32, cap: f32) -> f32 {
    logits.min(cap)
}

pub fn forward() -> f32 {
    softcap_logits(1.0, 2.0)
}
""",
)
write(
    "src/models/gemma2.rs",
    """\
pub struct Attention {
    softcap: f32,
}

impl Attention {
    fn softcap(&self, x: f32) -> f32 {
        // mlxcel_core::utils::softcap(x, cap) would be the unfused form.
        mlxcel_core::compiled_softcap(x, self.softcap)
    }

    pub fn forward(&self, softcap: f32) -> f32 {
        let label = "softcap(x, cap)";
        self.softcap(softcap + label.len() as f32)
    }
}
""",
)
write(
    "src/models/sampler.rs",
    """\
// `mlxcel_core::silu` is the raw FFI function re-exported at the crate root,
// not the `utils.rs` wrapper, and `softcap` here is a parameter.
pub fn activate(x: f32, softcap: f32) -> f32 {
    mlxcel_core::silu(x) + softcap
}
""",
)
mods += ["emitter_model", "muse_glimmer", "gemma2", "sampler"]
# Four more files with a private `softcap` of their own and a bare call to it,
# so that a matcher which takes any `softcap(` for a call sees six "callers".
for stem in ("audio_tower", "logit_ops", "norm_ops", "xla_emitter"):
    write(
        f"src/models/{stem}.rs",
        "fn softcap(x: f32, cap: f32) -> f32 {\n    x.min(cap)\n}\n\n"
        "pub fn forward() -> f32 {\n    softcap(1.0, 2.0)\n}\n",
    )
    mods.append(stem)

write("src/models/mod.rs", "".join(f"pub mod {name};\n" for name in sorted(mods)))
PY
  echo "$dir"
}

run_case() {
  local name="$1" dir="$2" expected_status="$3"
  shift 3
  local stdout="$work/$name.out"
  set +e
  python3 "$under_test" --root "$dir" "$@" >"$stdout" 2>&1
  local status=$?
  set -e
  if [ "$status" -ne "$expected_status" ]; then
    echo "FAIL $name: expected exit $expected_status, got $status" >&2
    sed -n '1,40p' "$stdout" >&2
    failures=$((failures + 1))
    return 1
  fi
  echo "ok   $name -> exit $status"
  return 0
}

assert_contains() {
  local name="$1" needle="$2"
  if ! grep -qF -- "$needle" "$work/$name.out"; then
    echo "FAIL $name: output does not mention '$needle'" >&2
    sed -n '1,40p' "$work/$name.out" >&2
    failures=$((failures + 1))
  fi
}

assert_not_contains() {
  local name="$1" needle="$2"
  if grep -qF -- "$needle" "$work/$name.out"; then
    echo "FAIL $name: output unexpectedly mentions '$needle'" >&2
    sed -n '1,40p' "$work/$name.out" >&2
    failures=$((failures + 1))
  fi
}

# Replace one literal string in a file, failing when it matches nothing, so a
# case cannot pass because its mutation silently stopped applying. Python
# rather than `sed -i`, whose flag differs between GNU and BSD sed.
replace_in() {
  python3 - "$@" <<'PY'
import pathlib
import sys

path, old, new = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
text = path.read_text()
if old not in text:
    raise SystemExit(f"fixture mutation: {old!r} not found in {path}")
path.write_text(text.replace(old, new))
PY
}

# Add a model file whose one function is `$3`. $1 tree, $2 file stem.
add_model() {
  local dir="$1" stem="$2" expression="$3"
  printf 'pub fn forward() -> f32 {\n    %s\n}\n' "$expression" >"$dir/src/models/$stem.rs"
}

utils="src/lib/mlxcel-core/src/utils.rs"
layers="src/lib/mlxcel-core/src/layers.rs"

# Control: the base tree passes, and the summary line reports what was checked,
# so every failure below is caused by its mutation and not by the fixture.
dir="$(make_tree control)"
if run_case control "$dir" 0; then
  assert_contains control "used-by: OK, 8 top-level pub fn in 2 shared files, 5 annotated"
  assert_not_contains control "not gated"
fi

# The softcap disambiguation. One file calls `utils.rs::softcap`. The private
# `fn softcap` under a glob import, `softcap_logits`, the `softcap` method,
# field, parameter, comment and string literal, the `#[cfg(test)]` module and
# the file declared under `#[cfg(test)]` are not callers. Nor is
# `mlxcel_core::silu` a caller of `utils.rs::silu`: six files call the wrapper
# and `sampler.rs` calls the FFI function.
if run_case control-report "$dir" 0 --report; then
  assert_contains control-report "  softcap:38 callers=1 annotation=roster"
  assert_contains control-report "      src/models/recurrent_gemma.rs:1 path"
  assert_contains control-report "  silu:30 callers=6 annotation=debt"
  assert_contains control-report "  create_causal_mask:18 callers=14 annotation=command"
  assert_contains control-report "  prefill_causal_mask:24 callers=13 annotation=roster"
  assert_contains control-report "      src/models/qwen3_5.rs:5 name"
  assert_contains control-report "  attention_from_ptr:15 callers=14 annotation=command"
  for decoy in emitter_model muse_glimmer gemma2 sampler recurrent_gemma_tests \
    audio_tower logit_ops norm_ops xla_emitter; do
    assert_not_contains control-report "src/models/$decoy.rs"
  done
fi

# The regenerating grep only matches the original spelling followed by "(".
# The directory named in the mask annotation covers the alias. Report the
# matching import line even if a different use statement precedes it.
dir="$(make_tree alias)"
cat >"$dir/src/models/alias.rs" <<'RS'
use mlxcel_core::layers;
use mlxcel_core::utils::create_causal_mask as ccm;
pub fn forward() -> f32 { ccm(1, 0) }
RS
if run_case alias "$dir" 0 --report; then
  assert_contains alias "src/models/alias.rs:2 path"
fi

dir="$(make_tree glob-alias)"
cat >"$dir/src/models/glob_alias.rs" <<'RS'
use mlxcel_core::utils::*;
use self::prefill_causal_mask as pcm;
pub fn forward() -> f32 { pcm(4) }
RS
if run_case glob-alias "$dir" 1; then
  assert_contains glob-alias 'annotation: src/models/glob_alias.rs.'
fi

dir="$(make_tree values)"
add_model "$dir" assignment 'use mlxcel_core::utils::*; let g = prefill_causal_mask; g(4)'
add_model "$dir" callback 'use mlxcel_core::utils::*; Some(4).map(prefill_causal_mask).unwrap()'
add_model "$dir" named 'use mlxcel_core::utils::prefill_causal_mask; Some(4).map(prefill_causal_mask).unwrap()'
if run_case values "$dir" 1; then
  assert_contains values 'annotation: src/models/assignment.rs, src/models/callback.rs, src/models/named.rs.'
fi

# A glob brings the function into scope, but a field, parameter or local of
# the same name does not use it. The local in one block must not hide a use
# in a later block.
dir="$(make_tree shadowed-values)"
cat >"$dir/src/models/shadow.rs" <<'RS'
use mlxcel_core::utils::*;
struct Options { prefill_causal_mask: i32 }
fn parameter(prefill_causal_mask: fn(i32) -> f32) -> f32 {
    prefill_causal_mask(4)
}
fn local() -> f32 {
    let prefill_causal_mask = |n: i32| n as f32;
    Some(4).map(prefill_causal_mask).unwrap()
}
RS
run_case shadowed-values "$dir" 0 || true
cat >>"$dir/src/models/shadow.rs" <<'RS'
fn use_after_block() -> f32 {
    { let prefill_causal_mask = 0; assert_eq!(prefill_causal_mask, 0); }
    Some(4).map(prefill_causal_mask).unwrap()
}
RS
if run_case value-after-block "$dir" 1; then
  assert_contains value-after-block 'annotation: src/models/shadow.rs.'
fi

# Re-exports offer downstream callers a path that omits the scoped module.
dir="$(make_tree re-export)"
cat >"$dir/src/models/facade.rs" <<'RS'
pub use mlxcel_core::utils::{prefill_causal_mask as mask, repeat_kv};
RS
if run_case re-export "$dir" 1; then
  assert_contains re-export "src/models/facade.rs:1: re-exports scoped function \`prefill_causal_mask\`"
  assert_contains re-export "src/models/facade.rs:1: re-exports scoped function \`repeat_kv\`"
fi
cat >"$dir/src/models/facade.rs" <<'RS'
use mlxcel_core::utils as shared;
pub use self::shared::prefill_causal_mask as mask;
RS
if run_case re-export-alias "$dir" 1; then
  assert_contains re-export-alias "src/models/facade.rs:2: re-exports scoped function \`prefill_causal_mask\`"
fi
cat >"$dir/src/models/facade.rs" <<'RS'
pub(crate) use mlxcel_core::utils::*;
RS
if run_case re-export-glob "$dir" 1; then
  assert_contains re-export-glob "src/models/facade.rs:1: re-exports scoped module \`mlxcel_core::utils\` through a glob"
fi

# The module, or a crate that contains it, is as good a facade as the function.
# Neither file spells a scoped function name, so the pre-filter must keep them.
cat >"$dir/src/models/facade.rs" <<'RS'
pub use mlxcel_core::utils as shared;
RS
if run_case re-export-module "$dir" 1; then
  assert_contains re-export-module "src/models/facade.rs:1: re-exports scoped module \`mlxcel_core::utils\` as a module"
fi
cat >"$dir/src/models/facade.rs" <<'RS'
pub use mlxcel_core::*;
RS
if run_case re-export-crate-glob "$dir" 1; then
  assert_contains re-export-crate-glob "src/models/facade.rs:1: re-exports \`mlxcel_core\`, which contains scoped module \`mlxcel_core::utils\`"
fi

# A generic helper called with a turbofish under a glob is still a caller.
dir="$(make_tree turbofish)"
add_model "$dir" turbofish 'use mlxcel_core::utils::*; prefill_causal_mask::<>(4)'
if run_case turbofish "$dir" 1; then
  assert_contains turbofish 'annotation: src/models/turbofish.rs.'
fi

# The same disambiguation, made to gate. With its annotation removed `softcap`
# still passes, because one caller is under the absence threshold. A checker
# that took every bare `softcap(` for a call would see six files (the caller
# and the five private functions of that name) and fail.
dir="$(make_tree softcap-unannotated)"
replace_in "$dir/$utils" "/// Used by: RecurrentGemma (\`src/models/recurrent_gemma.rs\`) for final logit
/// softcapping. Gemma 2 applies the same math through \`compiled_softcap\`.
" ''
run_case softcap-unannotated "$dir" 0 || true

# Absent annotation above the threshold. Six real callers of the unannotated
# `softcap` fail; the message names the function, the count and the callers.
for stem in mamba mamba2 rwkv7 dbrx bitnet; do
  add_model "$dir" "$stem" 'mlxcel_core::utils::softcap(1.0, 2.0)'
done
if run_case absent-annotation "$dir" 1; then
  assert_contains absent-annotation "$utils:36: \`softcap\` has 6 caller files outside its own module (threshold 5) and no \`/// Used by:\` annotation"
  assert_contains absent-annotation "src/models/bitnet.rs, src/models/dbrx.rs, src/models/mamba.rs"
  assert_not_contains absent-annotation "emitter_model"
fi

# A `Used by` mention that is not a `Used by:` line does not count as one.
dir="$(make_tree used-by-without-colon)"
replace_in "$dir/$utils" '/// Used by: NemotronH, Lfm2,' '/// Used by many models like NemotronH, Lfm2,'
if run_case used-by-without-colon "$dir" 1; then
  assert_contains used-by-without-colon "\`prefill_causal_mask\` has 13 caller files"
  assert_contains used-by-without-colon "its doc block mentions \`Used by\` but no line starts with \`Used by:\`"
fi

# The stale roster. This is the `create_causal_mask` doc block as it stood
# before #1121, verbatim. It names thirteen families; in this tree, as in the
# real one then, the files that call the function are mostly other ones, and
# Mixtral, Phi, Cohere and StarCoder2 do not call it at all. A check for
# absence passes this file: the annotation is there.
dir="$(make_tree stale-roster)"
python3 - "$dir/$utils" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
start = text.index("/// Create a causal attention mask.")
end = text.index("pub fn create_causal_mask(")
path.write_text(
    text[:start]
    + """\
/// Create a causal attention mask.
/// Used by: Llama, Qwen, Mixtral, Gemma, Cohere, Phi, OLMo, Exaone, GLM4,
/// MiniCPM, DeepSeek, Hunyuan, StarCoder2 and other causal attention callers
///
/// Creates a lower triangular mask of shape [size, size + offset] where:
/// - 1.0 indicates positions that can be attended to
/// - -inf indicates positions that should be masked
///
/// # Arguments
/// * `size` - Size of the query sequence
/// * `offset` - Offset for KV cache (number of previously cached tokens)
///
/// # Returns
/// Mask of shape [size, size + offset] with -inf in upper triangular region
"""
    + text[end:]
)
PY
if run_case stale-roster "$dir" 1; then
  assert_contains stale-roster "$utils:21: \`create_causal_mask\`: 12 of 14 caller files are not accounted for by the \`Used by:\` annotation"
  assert_contains stale-roster "src/models/falcon_h1.rs, src/models/gemma3.rs, src/models/gemma4.rs"
  assert_contains stale-roster "src/models/nemotron_h.rs"
  # `DeepSeek` and `Hunyuan` do name a caller each, so those two are not listed.
  assert_not_contains stale-roster "deepseek_v2.rs"
  assert_not_contains stale-roster "hunyuan_vl.rs"
  # The families it names that do not call it are not what is reported.
  assert_not_contains stale-roster "mixtral.rs"
  assert_not_contains stale-roster "starcoder2.rs"
fi

# A correct roster stays correct only while it is complete. A new caller that
# the roster does not name fails, and naming it passes again.
dir="$(make_tree roster-new-caller)"
add_model "$dir" rwkv7 'mlxcel_core::utils::prefill_causal_mask(4)'
if run_case roster-new-caller "$dir" 1; then
  assert_contains roster-new-caller "\`prefill_causal_mask\`: 1 of 14 caller files are not accounted for by the \`Used by:\` annotation: src/models/rwkv7.rs."
fi
replace_in "$dir/$utils" 'HunyuanVL and Llama4.' 'HunyuanVL, Llama4 and RWKV-7.'
run_case roster-new-caller-named "$dir" 0 || true

# A path names a caller as well as a family name does.
dir="$(make_tree roster-new-caller-path)"
add_model "$dir" rwkv7 'mlxcel_core::utils::prefill_causal_mask(4)'
replace_in "$dir/$utils" 'HunyuanVL and Llama4.' "HunyuanVL, Llama4 and \`src/models/rwkv7.rs\`."
run_case roster-new-caller-path "$dir" 0 || true

# A lower-case dictionary word is not a name: "the mamba models" names nothing.
dir="$(make_tree roster-prose-word)"
add_model "$dir" mamba 'mlxcel_core::utils::prefill_causal_mask(4)'
replace_in "$dir/$utils" 'HunyuanVL and Llama4.' 'HunyuanVL, Llama4 and the mamba models.'
if run_case roster-prose-word "$dir" 1; then
  assert_contains roster-prose-word "src/models/mamba.rs"
fi

# The rule form: a new caller is accounted for by the regenerating command,
# with no edit to the annotation, because that annotation records no count.
dir="$(make_tree command-new-caller)"
add_model "$dir" rwkv7 'unsafe { mlxcel_core::layers::attention_from_ptr(1.0) }'
run_case command-new-caller "$dir" 0 || true

# A recorded count is held to the command. A fifteenth file under `src/models`
# that calls `create_causal_mask` makes the stated 14 wrong.
dir="$(make_tree count-drift)"
add_model "$dir" rwkv7 'mlxcel_core::utils::create_causal_mask(4, 0)'
if run_case count-drift "$dir" 1; then
  assert_contains count-drift "\`create_causal_mask\`: the annotation states 14 files, but its regenerating command lists 15 at this commit"
  assert_not_contains count-drift "not accounted for"
fi
replace_in "$dir/$utils" 'that is 14 non-test' 'that is 15 non-test'
run_case count-drift-updated "$dir" 0 || true

# A comment changes the literal count without adding a production caller.
dir="$(make_tree count-comment)"
echo '// This path previously called create_causal_mask(4, 0).' >"$dir/src/models/old.rs"
if run_case count-comment "$dir" 1; then
  assert_contains count-comment "extra files may be comment or test mentions"
  assert_contains count-comment 'grep -v'
  assert_contains count-comment 'update the count'
  assert_not_contains count-comment "not accounted for"
fi

# "Not used by" is checked too. The annotation says Llama3 does not call the
# function; once `llama3.rs` does, the claim is reported along with the count.
dir="$(make_tree not-used-by-contradicted)"
add_model "$dir" llama3 'mlxcel_core::utils::create_causal_mask(4, 0)'
if run_case not-used-by-contradicted "$dir" 1; then
  assert_contains not-used-by-contradicted "the annotation says \`Not used by\` about src/models/llama3.rs, which call(s) it."
fi

# An annotation on a public item belongs in a `///` doc comment.
dir="$(make_tree plain-comment)"
replace_in "$dir/$utils" '/// Used by: DeepSeekV2 and Llama4.' '// Used by: DeepSeekV2 and Llama4.'
if run_case plain-comment "$dir" 1; then
  assert_contains plain-comment "\`repeat_kv\` keeps its \`Used by:\` annotation in a \`//\` comment"
fi

# A short roster with an unnamed caller is reported but not gated.
dir="$(make_tree short-roster-advisory)"
add_model "$dir" rwkv7 'mlxcel_core::utils::repeat_kv(1.0)'
if run_case short-roster-advisory "$dir" 0; then
  assert_contains short-roster-advisory "not gated, at or below the enumeration threshold of 12: 1 annotation(s) with 1 unnamed caller file(s)"
fi

# A regenerating command has to be one the check can re-run, and has to search
# for the function it sits on. Otherwise it would account for nothing, or for
# everything.
dir="$(make_tree command-unsupported)"
replace_in "$dir/$layers" "grep -rln '\\battention_from_ptr(' src --include='*.rs' | grep -v 'tests\\.rs\$'" "grep -rl attention_from_ptr src"
if run_case command-unsupported "$dir" 1; then
  assert_contains command-unsupported "the regenerating command \`grep -rl attention_from_ptr src\` is not in the supported form"
  assert_contains command-unsupported "\`attention_from_ptr\`: 14 of 14 caller files are not accounted for"
fi

dir="$(make_tree command-wrong-function)"
replace_in "$dir/$layers" "'\\battention_from_ptr('" "'\\battention('"
if run_case command-wrong-function "$dir" 1; then
  assert_contains command-wrong-function "does not search for \`attention_from_ptr\`"
fi

dir="$(make_tree command-missing-directory)"
replace_in "$dir/$layers" "attention_from_ptr(' src --include" "attention_from_ptr(' src/decoders --include"
if run_case command-missing-directory "$dir" 1; then
  assert_contains command-missing-directory "searches \`src/decoders\`, which does not exist"
fi

# UNANNOTATED_DEBT can only shrink: an entry that gained an annotation, or
# whose function is gone, fails until it is removed.
dir="$(make_tree debt-annotated)"
replace_in "$dir/$utils" 'pub fn slice_axis(' '/// Used by: NemotronH, Lfm2, Jamba, FalconH1, Plamo2 and Gemma3.
pub fn slice_axis('
if run_case debt-annotated "$dir" 1; then
  assert_contains debt-annotated "\`slice_axis\` is listed in UNANNOTATED_DEBT but now carries a \`Used by:\` annotation. Remove the entry."
fi

dir="$(make_tree debt-function-gone)"
replace_in "$dir/$utils" 'pub fn stack_arrays(x: f32) -> f32 {' 'pub fn stack_all(x: f32) -> f32 {'
if run_case debt-function-gone "$dir" 1; then
  assert_contains debt-function-gone "$utils::stack_arrays: listed in UNANNOTATED_DEBT but no such top-level \`pub fn\` exists. Remove the entry."
fi

# The scope is fail-closed: a scoped file that is renamed away fails rather
# than leaving the check to pass over what is left.
dir="$(make_tree scoped-file-gone)"
mv "$dir/$layers" "$dir/src/lib/mlxcel-core/src/attention_layers.rs"
if run_case scoped-file-gone "$dir" 1; then
  assert_contains scoped-file-gone "$layers: listed in SCOPED_FILES but does not exist"
fi

mkdir -p "$work/empty-tree/src"
if run_case empty-tree "$work/empty-tree" 1; then
  assert_contains empty-tree "$utils: listed in SCOPED_FILES but does not exist"
fi

if [ "$failures" -ne 0 ]; then
  echo "$failures case(s) failed" >&2
  exit 1
fi
echo "all used-by checker cases passed"
