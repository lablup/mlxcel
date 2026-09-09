# Shared model-store discovery guard for the benchmark shell harnesses.
#
# `MODELS_DIR` defaults to ./models, which holds checkpoints directly on one
# benchmark host and only container directories on the other, where the real
# roots are models/mlx and models/mlx-big. A sweep pointed at the wrong root
# enumerated nothing, wrote a header-only CSV and exited 0; the only signal was
# that a sweep budgeted in hours finished in two minutes. It happened twice in
# one day, in two different harnesses.
#
# Source this, then call `require_checkpoints "$MODELS_DIR" <label>` before the
# enumeration. A checkpoint is a directory holding config.json, the same
# predicate the sweeps use to classify SKIP:not_a_checkpoint.
#
# Call it before any output file is created. A guard that runs after the header
# is written cannot honestly say nothing was written.

require_checkpoints() {
  local root="$1" label="${2:-sweep}" entries=0 checkpoints=0 dir
  for dir in "$root"/*/; do
    [[ -d "$dir" ]] || continue
    entries=$((entries + 1))
    [[ -f "$dir/config.json" ]] && checkpoints=$((checkpoints + 1))
  done
  >&2 echo "Model store: $root ($entries entries, $checkpoints checkpoints)"
  if [[ "$checkpoints" -eq 0 ]]; then
    >&2 echo ""
    >&2 echo "Error: no checkpoints under '$root' for the $label."
    >&2 echo "  A sweep that measures nothing is a configuration error, not a"
    >&2 echo "  result. Nothing was written and no existing file was touched."
    >&2 echo "  If this host keeps checkpoints one level down, name that root:"
    >&2 echo "    MODELS_DIR=models/mlx $0 ..."
    exit 1
  fi
}

# Named-roster variant. `bench_longprompt.sh` takes a model list rather than
# enumerating, so its failure mode is different: the default roster named three
# directories that the 2026-09-09 store consolidation had renamed, and the
# sweep skipped them one at a time while still producing a CSV. Refuse when
# none of the named models resolves, and say which ones did not.
require_named_checkpoints() {
  local root="$1" label="$2"; shift 2
  local found=0 missing=() name
  for name in "$@"; do
    if [[ -f "$root/$name/config.json" ]]; then
      found=$((found + 1))
    else
      missing+=("$name")
    fi
  done
  >&2 echo "Model store: $root ($found of $# named checkpoints present)"
  if [[ ${#missing[@]} -gt 0 ]]; then
    >&2 echo "  not found: ${missing[*]}"
    >&2 echo "  A renamed directory reads as a skip, so check these against"
    >&2 echo "  docs/model-catalog.tsv rather than assuming they are absent."
  fi
  if [[ "$found" -eq 0 ]]; then
    >&2 echo ""
    >&2 echo "Error: none of the named checkpoints exists under '$root' for the $label."
    exit 1
  fi
}
