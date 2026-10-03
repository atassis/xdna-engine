#!/usr/bin/env bash
# Recreate every DATA input the served gemma4-12b resident-256k model needs, from the upstream
# checkpoint. Each stage is skipped when its output dir already carries a
# .recipe-manifest.json (producing command + input hashes + per-file sha256) that matches the
# dir's current content; a dir that exists with NO manifest (every live dir today) is treated as
# present-and-unmanaged and left untouched -- this script never rebuilds over a directory it did
# not itself write, so it can never touch the canonical live inputs by accident. Use --adopt to
# bless an existing unmanaged dir with a manifest (hashes it, writes nothing else) instead of
# --force, which rebuilds.
#
# Commands recovered from, per stage: dump_llm_weights.py's own --resume progress file
# (_dump_progress.json, the exact argv fingerprint of the live weights_int4g32sbf16_planar_qat_rg
# dump), xdna-engine/scenarios/generate-gemma4-12b.toml's recipe comment (source of truth per
# project doctrine), dump_gemma4_towers.py's docstring, weight_store.py / stack_prep.py source
# (module-level constants, now env-overridable), and resident_rf48C_p7148a7/meta.json
# (weight_dir/embedding_store).
set -euo pipefail
export LC_ALL=C   # locale collation puts "tokenizer.json" after "tokenizer_config.json"; sha_dir's
                   # bash sort must match Python's sorted() (codepoint order) or manifest_matches
                   # false-negatives on every rerun

HERE=$(dirname "$(readlink -f "$0")")
RF=$(dirname "$HERE")
ENGINE=$(cd "$RF/../.." && pwd)

OUT=${XDNA_DATA:-$ENGINE}/artifacts   # --out: root for checkpoint/weights/towers/store/hf_config/rf_stack
CHECKPOINT_REPO=google/gemma-4-12B-it-qat-q4_0-unquantized
CKPT_DIR_OVERRIDE=""             # --checkpoint-dir: read the checkpoint from elsewhere (e.g. an
                                  # already-present 24 GB download) instead of $OUT/gemma4-12b-qat/checkpoint
STAGES="checkpoint hf_config weights towers store rf_stack"
MODE=build   # build | adopt | force

usage() {
  cat <<EOF
usage: $0 [--out DIR] [--checkpoint-dir DIR] [--stages a,b,c] [--adopt] [--force]
  --out DIR            root for hf_config/weights/towers/store/rf_stack, and for the checkpoint
                        stage's own output unless --checkpoint-dir overrides it
                        (default $OUT; canonical paths -- an EXISTING unmanaged dir there is
                        always skipped, never overwritten)
  --checkpoint-dir DIR  read the checkpoint from DIR instead of \$OUT/gemma4-12b-qat/checkpoint --
                        for verifying downstream stages against an already-present checkpoint
                        without re-downloading 24 GB. Implies skipping the checkpoint stage.
  --stages LIST         comma-separated subset of: $STAGES (default: all, in this order)
  --adopt               for a dir that exists with no manifest: hash it and write the manifest,
                        do not rebuild
  --force                rebuild a stage even if its manifest already matches
EOF
}
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT=$2; shift 2 ;;
    --checkpoint-dir) CKPT_DIR_OVERRIDE=$2; shift 2 ;;
    --stages) STAGES=$(echo "$2" | tr , ' '); shift 2 ;;
    --adopt) MODE=adopt; shift ;;
    --force) MODE=force; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

# Data prep is pure numpy (no aiecc/IRON import), so it does not source env.sh (which requires
# IRON_DIR just to resolve the toolchain instance).
VENV_IRON="${VENV_IRON:-$ENGINE/.venv-iron}"
PY="$VENV_IRON/bin/python"
[ -x "$PY" ] || PY=python3

CKPT_DIR="${CKPT_DIR_OVERRIDE:-$OUT/gemma4-12b-qat/checkpoint}"
HFCFG_DIR="$OUT/gemma4-12b/hf_config"
WEIGHTS_DIR="$OUT/gemma4-12b/weights_int4g32sbf16_planar_qat_rg"
TOWERS_DIR="$OUT/gemma4-12b/towers_qat"
STORE_DIR="$OUT/gemma4-12b/store"
RF_STACK_DIR="$OUT/gemma4-12b/rf_stack"   # NEW default location, NOT scratch/rf/stack -- see note below

log() { printf '[gemma4_data] %s\n' "$*" >&2; }

sha_dir() {
  # sha256 of every regular file under $1, excluding .recipe-manifest.json* and *.tmp*, as a
  # sorted "relpath  sha256" stream -- stable across runs, independent of file order/mtime.
  find "$1" -type f ! -name '.recipe-manifest.json' ! -name '*.tmp*' -printf '%P\n' | sort |
    while read -r rel; do sha256sum "$1/$rel" | awk -v r="$rel" '{print r"  "$1}'; done
}

manifest_path() { echo "$1/.recipe-manifest.json"; }

manifest_matches() {
  # $1=dir. True iff a manifest exists there and its recorded file hashes equal the dir's
  # current content hashes (order-independent).
  local dir=$1 mf; mf=$(manifest_path "$dir")
  [ -f "$mf" ] || return 1
  local recorded current
  recorded=$($PY -c "import json,sys; m=json.load(open(sys.argv[1])); [print(f'{k}  {v}') for k,v in sorted(m['files'].items())]" "$mf")
  current=$(sha_dir "$dir")
  [ "$recorded" = "$current" ]
}

write_manifest() {
  # $1=dir $2=stage $3=command $4=json-inputs-fragment (may be "{}")
  local dir=$1 stage=$2 cmd=$3 inputs=$4 mf; mf=$(manifest_path "$dir")
  local files_json; files_json=$(sha_dir "$dir" | $PY -c "
import json, sys
files = {}
for line in sys.stdin:
    rel, h = line.rstrip('\n').rsplit('  ', 1)
    files[rel] = h
print(json.dumps(files, indent=1, sort_keys=True))")
  $PY -c "
import json, sys, time
mf, stage, cmd, inputs_s, files_s = sys.argv[1:6]
doc = {
    'stage': stage,
    'command': cmd,
    'inputs': json.loads(inputs_s),
    'created': time.strftime('%Y-%m-%d'),
    'files': json.loads(files_s),
}
json.dump(doc, open(mf, 'w'), indent=1, sort_keys=True)
" "$mf" "$stage" "$cmd" "$inputs" "$files_json"
}

# $1=dir $2=stage_name -> prints "skip"|"adopt"|"build"
decide() {
  local dir=$1
  if [ ! -d "$dir" ] || [ -z "$(find "$dir" -maxdepth 1 -type f 2>/dev/null)" ]; then
    echo build; return
  fi
  if [ -f "$(manifest_path "$dir")" ]; then
    if manifest_matches "$dir"; then
      [ "$MODE" = force ] && { echo build; return; }
      echo skip; return
    else
      echo build; return   # manifest present but stale -- this dir is script-managed, safe to rebuild
    fi
  fi
  # exists, non-empty, NO manifest: unmanaged legacy dir. Never rebuilt automatically.
  if [ "$MODE" = adopt ]; then echo adopt; else echo skip; fi
}

need_free_gb() {
  local need=$1 avail
  avail=$(df -BG --output=avail "$OUT" | tail -1 | tr -dc 0-9)
  if [ "$avail" -lt "$need" ]; then
    log "REFUSING: $OUT has ${avail}G free, need >= ${need}G for this stage"
    exit 1
  fi
}

build_tmp_then_rename() {
  # $1=final_dir $2=build_fn(tmp_dir) -- build_fn writes into $tmp, then this renames atomically.
  local final=$1 fn=$2 tmp
  tmp="${final}.rebuild.tmp.$$"
  rm -rf "$tmp"; mkdir -p "$tmp"
  "$fn" "$tmp"
  rm -rf "$final.old.$$"
  [ -d "$final" ] && mv "$final" "$final.old.$$"   # keep, don't delete -- caller inspects/removes
  mv "$tmp" "$final"
}

# ---------------------------------------------------------------------------
# stage: checkpoint  (google/gemma-4-12B-it-qat-q4_0-unquantized, ~24 GB safetensors)
# ---------------------------------------------------------------------------
stage_checkpoint() {
  local dir=$CKPT_DIR d
  if [ -n "$CKPT_DIR_OVERRIDE" ]; then
    log "checkpoint ($dir): skip (--checkpoint-dir override)"
    return
  fi
  d=$(decide "$dir")
  log "checkpoint ($dir): $d"
  case "$d" in
    skip) return ;;
    adopt) write_manifest "$dir" checkpoint "huggingface_hub.snapshot_download('$CHECKPOINT_REPO', local_dir=<dir>)" "{}"; return ;;
    build)
      need_free_gb 30
      build_tmp_then_rename "$dir" _build_checkpoint
      write_manifest "$dir" checkpoint "huggingface_hub.snapshot_download('$CHECKPOINT_REPO', local_dir=<dir>)" "{}"
      ;;
  esac
}
_build_checkpoint() {
  local tmp=$1
  $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$CHECKPOINT_REPO', local_dir='$tmp')
"
}

# ---------------------------------------------------------------------------
# stage: hf_config  (the checkpoint's own *.json files, no safetensors -- used by
# resident-forward's head_ref.py/stack_run.py as the served config+tokenizer)
# ---------------------------------------------------------------------------
stage_hf_config() {
  local dir=$HFCFG_DIR d
  d=$(decide "$dir")
  log "hf_config ($dir): $d"
  case "$d" in
    skip) return ;;
    adopt) write_manifest "$dir" hf_config "cp $CKPT_DIR/*.json $dir/" "{}"; return ;;
    build)
      [ -d "$CKPT_DIR" ] || { log "hf_config needs $CKPT_DIR (run the checkpoint stage first)"; exit 1; }
      build_tmp_then_rename "$dir" _build_hf_config
      write_manifest "$dir" hf_config "cp $CKPT_DIR/*.json $dir/" "{}"
      ;;
  esac
}
_build_hf_config() {
  local tmp=$1
  cp "$CKPT_DIR"/*.json "$tmp/"
}

# ---------------------------------------------------------------------------
# stage: weights  (int4 g32 QAT row_group_planar, shipped default since 2026-09-15 --
# generate-gemma4-12b.toml's recipe comment, confirmed byte-for-byte against the live dir's
# own _dump_progress.json 'fmt' fingerprint)
# ---------------------------------------------------------------------------
DUMP_WEIGHTS_CMD='python scripts/dump_llm_weights.py --spec gemma4-12b --checkpoint-dir <checkpoint> \
  --quant int4 --quant-group 32 --quant-layout row_group_planar --quant-scale-dtype bf16 \
  --quant-full-range --quant-clip-search \
  --quant-leaves gate_proj,up_proj,down_proj,o_proj,q_proj,k_proj,v_proj,embed_tokens \
  --resume --out <weights_dir>'

stage_weights() {
  local dir=$WEIGHTS_DIR d
  d=$(decide "$dir")
  log "weights ($dir): $d"
  case "$d" in
    skip) return ;;
    adopt) write_manifest "$dir" weights "$DUMP_WEIGHTS_CMD" "{}"; return ;;
    build)
      [ -d "$CKPT_DIR" ] || { log "weights needs $CKPT_DIR (run the checkpoint stage first)"; exit 1; }
      need_free_gb 15
      build_tmp_then_rename "$dir" _build_weights
      write_manifest "$dir" weights "$DUMP_WEIGHTS_CMD" "{}"
      ;;
  esac
}
_build_weights() {
  local tmp=$1
  ( cd "$ENGINE" && "$PY" scripts/dump_llm_weights.py --spec gemma4-12b --checkpoint-dir "$CKPT_DIR" \
      --quant int4 --quant-group 32 --quant-layout row_group_planar --quant-scale-dtype bf16 \
      --quant-full-range --quant-clip-search \
      --quant-leaves gate_proj,up_proj,down_proj,o_proj,q_proj,k_proj,v_proj,embed_tokens \
      --resume --out "$tmp" )
}

# ---------------------------------------------------------------------------
# stage: towers  (vision + audio tower tensors, sibling dump script, same checkpoint)
# ---------------------------------------------------------------------------
DUMP_TOWERS_CMD='python scripts/dump_gemma4_towers.py --checkpoint-dir <checkpoint> --out <towers_dir>'

stage_towers() {
  local dir=$TOWERS_DIR d
  d=$(decide "$dir")
  log "towers ($dir): $d"
  case "$d" in
    skip) return ;;
    adopt) write_manifest "$dir" towers "$DUMP_TOWERS_CMD" "{}"; return ;;
    build)
      [ -d "$CKPT_DIR" ] || { log "towers needs $CKPT_DIR (run the checkpoint stage first)"; exit 1; }
      need_free_gb 15
      build_tmp_then_rename "$dir" _build_towers
      write_manifest "$dir" towers "$DUMP_TOWERS_CMD" "{}"
      ;;
  esac
}
_build_towers() {
  local tmp=$1
  ( cd "$ENGINE" && "$PY" scripts/dump_gemma4_towers.py --checkpoint-dir "$CKPT_DIR" --out "$tmp" )
}

# ---------------------------------------------------------------------------
# stage: store  (resident embed/head content-addressed store -- weight_store.py, now
# RF_WDIR/RF_TOWERS_DIR/RF_STORE/RF_CHECKPOINT_DIR overridable; was hardcoded)
# ---------------------------------------------------------------------------
stage_store() {
  local dir=$STORE_DIR d
  d=$(decide "$dir")
  log "store ($dir): $d"
  case "$d" in
    skip) return ;;
    adopt) write_manifest "$dir" store "python weight_store.py (RF_WDIR/RF_TOWERS_DIR/RF_STORE/RF_CHECKPOINT_DIR)" "{}"; return ;;
    build)
      [ -d "$WEIGHTS_DIR" ] || { log "store needs $WEIGHTS_DIR (run the weights stage first)"; exit 1; }
      [ -d "$TOWERS_DIR" ] || { log "store needs $TOWERS_DIR (run the towers stage first)"; exit 1; }
      need_free_gb 10
      build_tmp_then_rename "$dir" _build_store
      write_manifest "$dir" store "python weight_store.py (RF_WDIR/RF_TOWERS_DIR/RF_STORE/RF_CHECKPOINT_DIR)" "{}"
      ;;
  esac
}
_build_store() {
  local tmp=$1
  ( cd "$RF" && RF_WDIR="$WEIGHTS_DIR" RF_TOWERS_DIR="$TOWERS_DIR" RF_STORE="$tmp" \
      RF_CHECKPOINT_DIR="$CKPT_DIR" "$PY" weight_store.py )
}

# ---------------------------------------------------------------------------
# stage: rf_stack  (per-layer device weight streams, stack_prep.py, RF_STORE/RF_STACK_OUT
# overridable). Writes to $OUT/gemma4-12b/rf_stack, not a scratch dir -- the streams a served
# artifact's meta.json points its weight_dir at should have the same lifecycle guarantee as the
# rest of the artifact tree.
# ---------------------------------------------------------------------------
stage_rf_stack() {
  local dir=$RF_STACK_DIR d
  d=$(decide "$dir")
  log "rf_stack ($dir): $d"
  case "$d" in
    skip)
      if [ ! -f "$dir/w_head.npy" ]; then
        _build_rf_head "$dir"
        write_manifest "$dir" rf_stack "python stack_prep.py 0 47; rhead.head_stream()" "{}"
      fi
      return ;;
    adopt) write_manifest "$dir" rf_stack "python stack_prep.py 0 47 (RF_STORE/RF_STACK_OUT)" "{}"; return ;;
    build)
      [ -d "$STORE_DIR" ] || { log "rf_stack needs $STORE_DIR (run the store stage first)"; exit 1; }
      need_free_gb 10
      build_tmp_then_rename "$dir" _build_rf_stack
      write_manifest "$dir" rf_stack "python stack_prep.py 0 47 (RF_STORE/RF_STACK_OUT)" "{}"
      ;;
  esac
}
_build_rf_stack() {
  local tmp=$1
  ( cd "$RF" && RF_WDIR="$WEIGHTS_DIR" RF_STORE="$STORE_DIR" RF_STACK_OUT="$tmp" "$PY" stack_prep.py 0 47 )
  _build_rf_head "$tmp"
}
_build_rf_head() {
  local dir=$1
  ( cd "$RF" && RF_STORE="$STORE_DIR" RF_STACK_OUT="$dir" "$PY" -c 'import rhead; rhead.head_stream()' )
}

# ---------------------------------------------------------------------------
main() {
  for s in $STAGES; do
    case "$s" in
      checkpoint) stage_checkpoint ;;
      hf_config)  stage_hf_config ;;
      weights)    stage_weights ;;
      towers)     stage_towers ;;
      store)      stage_store ;;
      rf_stack)   stage_rf_stack ;;
      *) log "unknown stage: $s"; exit 1 ;;
    esac
  done
  log "done."
}
main
