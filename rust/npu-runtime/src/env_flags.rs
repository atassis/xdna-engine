//! Inventory of `NPU_*`/related environment variables read by the SHIPPED crates -- a third
//! configuration plane alongside `engine.toml` ([`crate::config_doc`]) and the CLI flags
//! (`npu-cli/src/cli_def.rs`), and the only one of the three nothing could previously report as
//! active. `npu flags` (npu-cli) is this list rendered against the live process environment.
//!
//! Scope: crates in `rust/Cargo.toml`'s `default-members` (what `cargo build` at the workspace
//! root produces), excluding `npu-probes` (dev tooling, not shipped) and any read inside
//! `#[cfg(test)]` (never compiled into a shipped binary -- e.g. `QWEN3_TOKENIZER_DIR`,
//! `NPU_LLM_DEVICE_GATE`, `S2_ARTIFACT_DIR`). Build-time vars (`CARGO_*`, `OUT_DIR`) and pure
//! environment (`HOME`, `PATH`, `LD_LIBRARY_PATH`, `XRT_*`) are out of scope too.
//!
//! `npu_asr::tuning` already names two truth idioms (its `not_zero` closure and `is_one` function);
//! [`Semantics`] reuses that vocabulary rather than inventing a competing one, and adds the idioms
//! tuning.rs does not cover: `is_ok()` (ANY set value, including the string `"0"`, reads as true),
//! bare presence (`var_os(..).is_some()`), an explicit falsy-string list, and a plain value that is
//! consumed rather than reduced to a bool.
//!
//! This list is report-only by construction: it does not change what any site reads, parses, or
//! defaults to. A disagreement between two sites about what the SAME name means is recorded here,
//! not silently resolved -- resolving one is a shipped-behaviour change and belongs to its own
//! change, not to an inventory (`NPU_ENC_FFN_RESIDENT` was the one live instance; fixed 2026-09-08
//! by routing npu-whisper through npu-asr's accessor, see that flag's entry below).

/// A var's truth/parse idiom. The four not named after `TuningConfig`'s own two are the ones nine
/// flags already standardize on; everything else in the engine uses one of the other four.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Semantics {
    /// `npu_asr::tuning`'s `not_zero`: unset or any value but the literal `"0"` reads true; `"0"`
    /// reads false. The engine's most common default-ON idiom (opt out with `=0`).
    NotZero,
    /// `npu_asr::tuning`'s `is_one`: only the exact value `"1"` reads true; any other set value
    /// reads false. The engine's default-OFF idiom (opt in with exactly `=1`).
    IsOne,
    /// `.is_ok()`: ANY set value, INCLUDING the string `"0"`, reads true. The idiom most likely to
    /// surprise a reader coming from `not_zero` -- `FOO=0` turns an `is_ok()` flag ON.
    IsOk,
    /// `var_os(..).is_some()`: presence only, the value itself is never inspected.
    Presence,
    /// An explicit list of strings that read false (`"0"`, `"false"`, `"no"`); any other set value
    /// reads true; unset reads the default.
    FalsyList,
    /// The value itself is consumed (a path, a number, an enum word) rather than reduced to a
    /// bool.
    Value,
}

impl Semantics {
    /// Short column word for `npu flags`'s table (matches `TuningConfig`'s own closure names
    /// where one exists).
    pub fn code(self) -> &'static str {
        match self {
            Semantics::NotZero => "not_zero",
            Semantics::IsOne => "is_one",
            Semantics::IsOk => "is_ok",
            Semantics::Presence => "presence",
            Semantics::FalsyList => "falsy_list",
            Semantics::Value => "value",
        }
    }

    /// One line for `npu flags`: the parse rule in the reader's own terms.
    pub fn describe(self) -> &'static str {
        match self {
            Semantics::NotZero => "not_zero: unset or any value but \"0\" => true; \"0\" => false",
            Semantics::IsOne => "is_one: only \"1\" => true; any other set value => false",
            Semantics::IsOk => "is_ok: ANY set value (including \"0\") => true",
            Semantics::Presence => "presence: var_os(..).is_some(), value ignored",
            Semantics::FalsyList => "falsy-list: \"0\"/\"false\"/\"no\" => false; any other set value => true",
            Semantics::Value => "value: consumed directly, not reduced to a bool",
        }
    }
}

/// One environment-configured knob. `owner` names the crate(s) that read it; `site` is the
/// clearest read site when there are several (a merged entry says so and names the rest in `doc`).
pub struct Flag {
    pub name: &'static str,
    pub owner: &'static str,
    pub site: &'static str,
    pub semantics: Semantics,
    pub default: &'static str,
    pub doc: &'static str,
}

use Semantics::*;

/// The full census, in crate order. See the module doc for what is deliberately excluded.
pub const FLAGS: &[Flag] = &[
    // -- npu-engine: llm decode ------------------------------------------------------------------
    Flag { name: "NPU_LLM_REUSE_KV", owner: "npu-engine", site: "npu-engine/src/llm/npu_decode.rs:301",
        semantics: NotZero, default: "true",
        doc: "reuse the KV-cache buffers across requests instead of re-zeroing them each time. \
              Default ON: the buffers are zeroed explicitly at load and sm_mask excludes every \
              position at or beyond n_past, so the per-request pass cost 224 MiB of host memset \
              plus an arena write (~60 ms/request at S=2048) and changed no output. Set =0 to \
              restore it when bisecting a suspected KV bug." },
    Flag { name: "NPU_LLM_PREFILL_BATCHED", owner: "npu-engine", site: "npu-engine/src/llm/npu_prefill.rs:56",
        semantics: NotZero, default: "true",
        doc: "prime the KV cache over a prompt in batches of the prefill artifact's dims.M instead \
              of one dispatch per token. No effect unless the scenario names artifacts.prefill -- \
              without that artifact there is no batched ELF and the flag reads on a path that does \
              not exist. Set =0 for the A/B control behind every prefill measurement, and to \
              bisect a batched prompt that disagrees with P sequential steps. Both arms are \
              device-only: this is a step within the tier ladder, never a fall to host." },

    // -- npu-asr-host --------------------------------------------------------------------------
    Flag { name: "NPU_PAR_SUBSAMPLE", owner: "npu-asr-host", site: "npu-asr-host/src/lib.rs:507",
        semantics: NotZero, default: "true",
        doc: "host-side subsample matmul runs multithreaded via rayon; opt out with =0." },
    Flag { name: "NPU_HOST_PROF", owner: "npu-asr-host", site: "npu-asr-host/src/prof.rs:24",
        semantics: Presence, default: "false",
        doc: "enables the host profiler that times ASR host-side stages (prof::time/report)." },

    // -- npu-asr: npu_asr::tuning::TuningConfig::with_env_overrides -----------------------------
    // The nine flags already gated by TuningConfig::baked_default() + with_env_overrides(). Read
    // via `not_zero`/`is_one` closures parameterized by these literal keys, not by a direct
    // `env::var("LITERAL")` call -- which is why a plain grep for the literal undercounts this
    // crate.
    Flag { name: "NPU_MODAL_EPI", owner: "npu-asr", site: "npu-asr/src/tuning.rs:68",
        semantics: NotZero, default: "true",
        doc: "modal (fused) epilogue on the encoder GEMMs; gated to non-int8 at construction." },
    Flag { name: "NPU_SS_NPU", owner: "npu-asr", site: "npu-asr/src/tuning.rs:69",
        semantics: NotZero, default: "true",
        doc: "subsampling stage dispatches on the NPU instead of host." },
    Flag { name: "NPU_GLU_FUSED", owner: "npu-asr", site: "npu-asr/src/tuning.rs:70",
        semantics: NotZero, default: "true",
        doc: "collapses three HOST passes (bias add, transpose, GLU) into one rayon pass in \
              npu_asr_host::glu_fused. Despite the name nothing is fused into a GEMM epilogue and \
              no device call is involved; both arms are host-only and numerically exact. Also read independently, \
              same not_zero-equivalent check (`!= Ok(\"0\")`), at block.rs:328 -- the two agree." },
    Flag { name: "NPU_MM2_PIPELINE", owner: "npu-asr", site: "npu-asr/src/tuning.rs:71",
        semantics: NotZero, default: "true",
        doc: "pipelines the FFN's second matmul (mm2)." },
    Flag { name: "NPU_INT8_FASTEPI", owner: "npu-asr", site: "npu-asr/src/tuning.rs:72",
        semantics: NotZero, default: "true",
        doc: "fast int8 epilogue path; only meaningful under int8 precision." },
    Flag { name: "NPU_LN_NPU", owner: "npu-asr", site: "npu-asr/src/tuning.rs:74",
        semantics: IsOne, default: "false",
        doc: "LayerNorm dispatches on the NPU instead of host." },
    Flag { name: "NPU_QKV_OVERLAP", owner: "npu-asr", site: "npu-asr/src/tuning.rs:75",
        semantics: IsOne, default: "false",
        doc: "overlaps QKV projection dispatch with the previous stage. Also read independently, \
              same is_one-equivalent check (`== Ok(\"1\")`), at block.rs:330 (feature two_ctx, \
              default-on) -- the two agree." },
    Flag { name: "NPU_INT8_ONCHIP", owner: "npu-asr", site: "npu-asr/src/tuning.rs:76",
        semantics: IsOne, default: "false",
        doc: "on-chip int8 dequantization instead of host." },
    Flag { name: "NPU_ENC_FFN_RESIDENT", owner: "npu-asr", site: "npu-asr/src/tuning.rs:23",
        semantics: IsOne, default: "false",
        doc: "resident fc1->fc2 FFN intermediate stays on-device (draft, default OFF), read via \
              `tuning::ffn_resident_requested()` (E003 single accessor). Also read independently, \
              same is_one-equivalent check (`== Ok(\"1\")`), at block.rs:130 -- agrees. FIXED \
              2026-09-08: npu-whisper used to read the SAME name at its own site with is_ok() \
              semantics -- ANY set value, including \"0\", true -- so NPU_ENC_FFN_RESIDENT=0 \
              disabled this crate's residency and enabled whisper's from one export. \
              npu-whisper/src/encoder.rs now calls this accessor instead of reading env directly." },

    // -- npu-asr: everything else -----------------------------------------------------------------
    Flag { name: "NPU_CONV_TRANSPOSE", owner: "npu-asr", site: "npu-asr/src/conv_npu.rs:44",
        semantics: NotZero, default: "false",
        doc: "swaps the conv stem's M/N tiling to shrink the padded dispatch." },
    Flag { name: "NPU_KERNEL_MANIFEST_VERIFY", owner: "npu-asr, npu-engine, npu-parakeet",
        site: "npu-asr/src/conv_npu.rs:70", semantics: IsOk, default: "false",
        doc: "re-hashes xclbin/insts against kernel_manifest.json before load, so a stale artifact \
              fails loud instead of loading silently. Same is_ok() check at 3 independent sites \
              (also npu-engine/src/esm/native.rs:63, npu-parakeet/src/npu.rs:797) -- consistent." },
    Flag { name: "NPU_TILE", owner: "npu-asr", site: "npu-asr/src/ctx2.rs:74",
        semantics: Value, default: "derived from Precision::tile()",
        doc: "override the (m,k,n) kernel tile as `mxkxn`; diagnostic, panics on a bad format." },
    Flag { name: "NPU_PRECISION", owner: "npu-asr", site: "npu-asr/src/ctx2.rs:93",
        semantics: Value, default: "bf16 (Precision::FastBf16)",
        doc: "runtime precision selector for the encoder GEMMs: native|bf16|int8." },
    Flag { name: "NPU_ENC_GELU_FUSED", owner: "npu-asr", site: "npu-asr/src/ctx2.rs:316",
        semantics: IsOk, default: "false",
        doc: "picks the GELU-fused xclbin stem/required artifact set for the encoder. COUPLING: \
              npu-whisper reads the SAME name (encoder.rs) with the same is_ok() rule to decide \
              whether to skip the host GELU at runtime -- nothing ties the two reads together, so \
              setting it for one crate's artifacts without the other is a silent mismatch." },

    // -- npu-cli --------------------------------------------------------------------------------
    Flag { name: "NPU_CONFIG", owner: "npu-cli", site: "npu-cli/src/main.rs:32",
        semantics: Value, default: "$HOME/.config/npu/engine.toml",
        doc: "overrides the engine.toml config path (also settable via --config)." },
    Flag { name: "NPU_QUIET", owner: "npu-cli", site: "npu-cli/src/main.rs:89",
        semantics: Presence, default: "n/a (checks ABSENCE, not presence)",
        doc: "DOCUMENTED FALSE LEAD, correct as written: quiet_one_shot() checks \
              var_os(..).is_none() -- \"did the caller express a preference at all\" -- and only \
              then sets NPU_QUIET=1 for one-shot commands. It does not itself gate the banners; \
              npu-xrt's NPU_QUIET (this table, owner npu-xrt) is the actual consumer." },
    Flag { name: "XDNA_ENGINE_ROOT", owner: "npu-cli", site: "npu-cli/src/main.rs:112",
        semantics: Value, default: "derived (XDG_DATA_HOME, or cwd, checked for scenarios/)",
        doc: "explicit override for the repo root that scenario/artifact paths resolve against." },
    Flag { name: "XDG_DATA_HOME", owner: "npu-cli", site: "npu-cli/src/main.rs:116",
        semantics: Value, default: "$HOME/.local/share",
        doc: "XDG data-home candidate for the install root when XDNA_ENGINE_ROOT is unset." },
    Flag { name: "NPU_ASR_MAX_SPAN_S", owner: "npu-cli", site: "npu-cli/src/main.rs:417",
        semantics: Value, default: "18.0",
        doc: "max transcription window span in seconds; span-granularity only now that both ASR \
              backends window internally (not re-measured against them)." },

    // -- npu-dispatch -----------------------------------------------------------------------------
    Flag { name: "NPU_MARSH_PROF", owner: "npu-dispatch", site: "npu-dispatch/src/lib.rs:133",
        semantics: IsOk, default: "false",
        doc: "prints the per-op x per-stage dispatch marshaling table (dump/dump_dispatch_prof)." },

    // -- npu-engine -------------------------------------------------------------------------------
    Flag { name: "NPU_XCLBIN_ROOT", owner: "npu-engine", site: "npu-engine/src/asr/parakeet.rs:94",
        semantics: Value, default: "the scenario root",
        doc: "overrides where compiled xclbin/insts artifacts are read from. Same read at \
              npu-engine/src/asr/whisper.rs:295 for the Whisper backend." },
    Flag { name: "PARAKEET_WINDOW_DEBUG", owner: "npu-engine", site: "npu-engine/src/asr/parakeet.rs:262",
        semantics: Presence, default: "false",
        doc: "prints per-window TDT decode debug info (mel bounds, token count, text) to stderr." },
    Flag { name: "BERT_RESIDENT_FFN", owner: "npu-engine", site: "npu-engine/src/bert/encoder.rs:95",
        semantics: IsOne, default: "false",
        doc: "resident device-side BERT FFN rail (K=1024); anything but \"1\" falls back to host." },
    Flag { name: "NPU_DIARIZE_TIME", owner: "npu-engine", site: "npu-engine/src/diarize/mod.rs:49",
        semantics: Presence, default: "false",
        doc: "per-stage wall-clock timing for diarization (segmenter vs embedder)." },
    Flag { name: "NPU_DIARIZE_THREADS", owner: "npu-engine", site: "npu-engine/src/diarize/onnx.rs:58",
        semantics: Value, default: "physical_cores()",
        doc: "ONNX thread count for the diarization embedder (the segmenter stays pinned at 1)." },
    Flag { name: "NPU_DIARIZE_MEM_MB", owner: "npu-engine", site: "npu-engine/src/diarize/onnx.rs:131",
        semantics: Value, default: "768",
        doc: "memory budget (MB) for diarization embed batching (~55 MB/crop measured slope)." },
    Flag { name: "NPU_DIARIZE_DEBUG", owner: "npu-engine", site: "npu-engine/src/diarize/vbx.rs:191",
        semantics: Presence, default: "false",
        doc: "prints per-crop VBx diarization debug stats to stderr." },
    Flag { name: "NPU_WEIGHTS_CHECKPOINT", owner: "npu-engine, npu-parakeet",
        site: "npu-engine/src/esm/weights.rs:84", semantics: Value, default: "unset (use npy dirs)",
        doc: "staged bf16 checkpoint path; when set, loads tensors from it instead of the npy \
              dirs. Same read at npu-parakeet/src/weights.rs:106 -- consistent." },
    Flag { name: "NPU_DECODE_ATTN", owner: "npu-engine", site: "npu-engine/src/asr/whisper_decoder.rs:366",
        semantics: IsOk, default: "false",
        doc: "on-NPU self-attention in the HOST decoder's per-op path (HostDecoder::step, ~72 \
              dispatches/token) -- NOT the fused decoder, which never reads it. Pair with \
              NPU_DECODE=1, never NPU_DECODE_FUSED=1." },
    Flag { name: "NPU_DECODE_FUSED_PATCH", owner: "npu-engine", site: "npu-engine/src/asr/whisper_decoder.rs:1091",
        semantics: IsOk, default: "false",
        doc: "opts OUT of the resident scratchpad-arena decode path back to legacy per-token ELF \
              patch/reload (A/B); set = legacy path." },
    Flag { name: "FUSED_PHASE_TIMING", owner: "npu-engine", site: "npu-engine/src/asr/whisper_decoder.rs:1133",
        semantics: IsOk, default: "false",
        doc: "per-phase timing breakdown for the fused decode path. Read again at line 1817." },
    Flag { name: "NPU_DECODE_FUSED_REUSECTX", owner: "npu-engine", site: "npu-engine/src/asr/whisper_decoder.rs:1137",
        semantics: IsOk, default: "false",
        doc: "persistent hw_context across decode tokens (no per-token re-registration)." },
    Flag { name: "NPU_DECODE_FUSED_HOSTCROSS", owner: "npu-engine", site: "npu-engine/src/asr/whisper_decoder.rs:1146",
        semantics: IsOk, default: "false",
        doc: "opts OUT of the on-device cross-K/V GEMM fold back to the host f32 fold (A/B); set \
              = host fold. Read again at line 1820." },
    Flag { name: "NPU_DECODE_PROJOUT_CTX2", owner: "npu-engine", site: "npu-engine/src/asr/whisper_decoder.rs:1149",
        semantics: IsOk, default: "false",
        doc: "routes lm-head proj_out through the shared ctx2 kernel instead of the dedicated ELF \
              path; also disables NPU_DECODE_PROJOUT_ELF when set (see that entry)." },
    Flag { name: "NPU_DECODE_PROJOUT_ELF", owner: "npu-engine", site: "npu-engine/src/asr/whisper_decoder.rs:1168",
        semantics: FalsyList, default: "true",
        doc: "prebuilt lm-head projection ELF path, default ON; =0/false/no falls back to the \
              host lm-head, as does NPU_DECODE_PROJOUT_CTX2 being set." },
    Flag { name: "NPU_DECODE_PROJOUT_ELF_DIR", owner: "npu-engine", site: "npu-engine/src/asr/whisper_decoder.rs:1174",
        semantics: Value, default: "<fused_decode_dir>/../projout_elf",
        doc: "overrides the proj_out ELF artifact directory." },
    Flag { name: "INT8_CK_HEADROOM", owner: "npu-engine", site: "npu-engine/src/asr/whisper_decoder.rs:1354",
        semantics: Value, default: "1.0",
        doc: "int8 quantization headroom multiplier for cross-attn K (per-utterance per-channel scale)." },
    Flag { name: "INT8_CV_HEADROOM", owner: "npu-engine", site: "npu-engine/src/asr/whisper_decoder.rs:1370",
        semantics: Value, default: "1.0",
        doc: "int8 quantization headroom multiplier for cross-attn V." },
    Flag { name: "WHISPER_ENC_HOST", owner: "npu-engine", site: "npu-engine/src/asr/whisper.rs:317",
        semantics: IsOk, default: "false",
        doc: "runs the Whisper encoder on host ONNX instead of the NPU; opt-in, loud (never a \
              silent fallback)." },
    Flag { name: "NPU_DECODE_FUSED", owner: "npu-engine", site: "npu-engine/src/config.rs:147",
        semantics: IsOk, default: "false",
        doc: "whole 12-layer fused-ELF decode backend (1 dispatch/token); takes precedence over \
              NPU_DECODE and over the scenario's [decode] backend field \
              (see config::resolve_decode_backend)." },
    Flag { name: "NPU_DECODE", owner: "npu-engine", site: "npu-engine/src/config.rs:150",
        semantics: IsOk, default: "false",
        doc: "per-op NPU decode backend (~72 dispatches/token), used when NPU_DECODE_FUSED is unset; \
              overrides the scenario's [decode] backend field." },
    Flag { name: "NPU_DECODE_FUSED_BATCH", owner: "npu-engine", site: "npu-engine/src/asr/whisper.rs:343",
        semantics: IsOk, default: "false",
        doc: "batched/offline-bulk fused decoder subsystem, independent of the single-stream backend." },
    Flag { name: "NPU_DECODE_FUSED_BATCH_DIR", owner: "npu-engine", site: "npu-engine/src/asr/whisper.rs:356",
        semantics: Value, default: "none -- required when NPU_DECODE_FUSED_BATCH is set (load fails without it)",
        doc: "artifact dir for the batched fused decoder." },
    Flag { name: "NPU_DECODE_FUSED_DIR", owner: "npu-engine", site: "npu-engine/src/asr/whisper.rs:371",
        semantics: Value, default: "<root>/artifacts/fused_decode12",
        doc: "overrides the fused-decode-ELF artifact directory (A/B of alternate builds)." },
    Flag { name: "WHISPER_LANG_PIN", owner: "npu-engine", site: "npu-engine/src/asr/whisper.rs:836",
        semantics: IsOk, default: "false",
        doc: "pins the first window's detected language across all windows instead of per-window \
              detection (right only for known single-language recordings)." },
    Flag { name: "WHISPER_TIMING", owner: "npu-engine", site: "npu-engine/src/asr/whisper.rs:859",
        semantics: IsOk, default: "false",
        doc: "per-stage (prep/encode/decode) wall-clock timing for one transcribe window." },
    Flag { name: "NPU_DEBUG_TOKEN_IDS", owner: "npu-engine", site: "npu-engine/src/asr/whisper.rs:908",
        semantics: IsOk, default: "false",
        doc: "prints raw decoded token ids to stderr; bit-for-bit comparator, off by default." },

    // -- npu-parakeet -----------------------------------------------------------------------------
    Flag { name: "PARAKEET_DUMP_CONVIN", owner: "npu-parakeet", site: "npu-parakeet/src/encoder.rs:32",
        semantics: Value, default: "unset (off)",
        doc: "dumps the conv-front input tensor to {dir}/{tag}_b{blk}.npy, for parity bisection." },
    Flag { name: "PARAKEET_RESIDENT_FF", owner: "npu-parakeet", site: "npu-parakeet/src/encoder.rs:159",
        semantics: NotZero, default: "true",
        doc: "resident on-chip LN->fc1->SiLU FFN stage 1 on the modal resident path. NOT a \
              duplicate of PARAKEET_RESIDENT_FFN: this one gates the outer stage; FFN (below) \
              gates whether fc2's K-split ALSO stays on-device, nested inside this one." },
    Flag { name: "PARAKEET_RESIDENT_FFN", owner: "npu-parakeet", site: "npu-parakeet/src/encoder.rs:175",
        semantics: NotZero, default: "true",
        doc: "nested inside PARAKEET_RESIDENT_FF: keeps fc2's K-split accumulation on-device too \
              (deinterleave + sub-BO chunks + host-sum, bit-identical to the host 4x K-split)." },
    Flag { name: "PARAKEET_FFN_DEVACC", owner: "npu-parakeet", site: "npu-parakeet/src/encoder.rs:182",
        semantics: NotZero, default: "true (also requires !hybrid())",
        doc: "accumulates fc2 ON-DEVICE via the acc_add brick instead of host K-split-sum; falls \
              through to resident_ffn if the acc_add xclbin is absent. Read via resident_on(), a \
              dynamic-name closure (encoder.rs:276) -- invisible to a grep for the literal string." },
    Flag { name: "PARAKEET_HYBRID", owner: "npu-parakeet", site: "npu-parakeet/src/encoder.rs:270",
        semantics: NotZero, default: "false",
        doc: "hybrid encoder mode (measured 8.425% WER at ~1.89 s/clip vs the default's 8.791%/ \
              0.98s); forces every PARAKEET_RESIDENT_*/FUSED_BLOCK off via resident_on(), and uses \
              all 16 of the driver's hw_contexts." },
    Flag { name: "PARAKEET_RESIDENT_MHA", owner: "npu-parakeet", site: "npu-parakeet/src/encoder.rs:386",
        semantics: NotZero, default: "true (also requires !hybrid())",
        doc: "resident on-chip multi-head attention. Read via resident_on() (dynamic name); also \
              checked at line 540." },
    Flag { name: "PARAKEET_RESIDENT_CONV", owner: "npu-parakeet", site: "npu-parakeet/src/encoder.rs:777",
        semantics: NotZero, default: "true (also requires !hybrid())",
        doc: "resident on-chip conv module. Read via resident_on() (dynamic name)." },
    Flag { name: "PARAKEET_RESIDENT_SILU", owner: "npu-parakeet", site: "npu-parakeet/src/encoder.rs:779",
        semantics: NotZero, default: "true (also requires !hybrid())",
        doc: "resident on-chip SiLU activation. Read via resident_on() (dynamic name)." },
    Flag { name: "PARAKEET_FUSED_BLOCK", owner: "npu-parakeet", site: "npu-parakeet/src/encoder.rs:948",
        semantics: NotZero, default: "true (also requires !hybrid())",
        doc: "fully fused encoder block dispatch. Read via resident_on() (dynamic name)." },
    Flag { name: "PARAKEET_SUBSAMPLE_OUT_NPU", owner: "npu-parakeet", site: "npu-parakeet/src/encoder.rs:358",
        semantics: NotZero, default: "false",
        doc: "subsampling output stage dispatches on NPU." },
    Flag { name: "PARAKEET_MHA_HOSTQKV", owner: "npu-parakeet", site: "npu-parakeet/src/encoder.rs:394",
        semantics: IsOk, default: "false (set = diagnostic ON)",
        doc: "DIAGNOSTIC, inverted polarity: when SET, keeps the resident attention block but \
              feeds it HOST f32-LN + mm_lazy q/k/v instead of the resident QKV, to isolate the \
              LN->QKV seam. No effect unless PARAKEET_RESIDENT_MHA is active." },
    Flag { name: "PARAKEET_MHA_SPLITA", owner: "npu-parakeet", site: "npu-parakeet/src/encoder.rs:404",
        semantics: NotZero, default: "true",
        doc: "bf16x2 device-A split for resident MHA (WER-neutral 8.5); opt out =0 for the old \
              single-bf16-A path (WER 8.9)." },
    Flag { name: "PARAKEET_MHA_QKV_AB", owner: "npu-parakeet", site: "npu-parakeet/src/encoder.rs:458",
        semantics: IsOk, default: "false",
        doc: "A/B diagnostic: resident LN->QKV vs host layernorm+matmul, rel-L2 per projection." },
    Flag { name: "PARAKEET_MHA_AB", owner: "npu-parakeet", site: "npu-parakeet/src/encoder.rs:548",
        semantics: IsOk, default: "false",
        doc: "A/B diagnostic: resident MHA context vs f32 host golden, for head 0 and a mid head." },
    Flag { name: "PARAKEET_CONVEYOR_MHA", owner: "npu-parakeet", site: "npu-parakeet/src/encoder.rs:680",
        semantics: IsOk, default: "false",
        doc: "conveyor (8-head merged dispatch) MHA path. The \"TODO stub\" this doc used to claim was \
              wired by 9ef97ea on 2026-07-17 and the source comment corrected by 9abcaf2; this \
              registry entry copied the dead text three weeks later. Non-functional for two OTHER \
              reasons: artifacts/conveyor/single/ does not exist (conveyor_block() now declines \
              gracefully instead of panicking on that -- relpos-loader-bypasses-the-manifest-and-\
              can-load-a-mislabelled-bucket, loader B), and the 16 hw_context budget is already spent." },
    Flag { name: "PARAKEET_CONVEYOR_BD", owner: "npu-parakeet", site: "npu-parakeet/src/npu.rs:126",
        semantics: Value, default: "Plain",
        doc: "BD-belt carry precision for the conveyor path: \"split\" for hi+lo bf16, else plain \
              (Deliverable-1 verdict: plain sufficient, half the BD belt bytes)." },
    Flag { name: "PARAKEET_RELPOS_ROWTILE", owner: "npu-parakeet", site: "npu-parakeet/src/npu.rs:186",
        semantics: Presence, default: "false",
        doc: "selects the row-tiled relpos bucket table (must match the artifact set PARAKEET_RELPOS_DIR names)." },
    Flag { name: "PARAKEET_RELPOS_NOPEEL", owner: "npu-parakeet", site: "npu-parakeet/src/npu.rs:188",
        semantics: Presence, default: "false",
        doc: "selects the no-peel relpos bucket table; only checked when ROWTILE is unset." },
    Flag { name: "PARAKEET_LN_MODE", owner: "npu-parakeet", site: "npu-parakeet/src/npu.rs:718",
        semantics: NotZero, default: "false",
        doc: "mode-carrying LN panel; opt-in, moves the encoder onto a different xclbin (default \
              flip is a separate, deliberately deferred decision)." },
    Flag { name: "PARAKEET_MODAL_EPI_SUFFIX", owner: "npu-parakeet", site: "npu-parakeet/src/npu.rs:731",
        semantics: Value, default: "\"\" (empty)",
        doc: "suffix appended to the fc1 panel / insts-only stem names for artifact selection. \
              Read again at line 3587 (skipped under fold_fc1())." },
    Flag { name: "PARAKEET_FOLD_FC1", owner: "npu-parakeet", site: "npu-parakeet/src/npu.rs:796",
        semantics: NotZero, default: "false",
        doc: "folds fc1 into the resident modal xclbin (prices folding fc1 into the modal path)." },
    Flag { name: "PARAKEET_FOLD_GLU", owner: "npu-parakeet", site: "npu-parakeet/src/npu.rs:811",
        semantics: NotZero, default: "false",
        doc: "folds GLU into the pw1 resident dispatch." },
    Flag { name: "NPU_NATIVE", owner: "npu-parakeet", site: "npu-parakeet/src/npu.rs:881",
        semantics: IsOk, default: "false",
        doc: "native bf16 32x32x32 resident tile instead of the default fast-bfp16 64x32x128." },
    Flag { name: "NPU_RESIDENT_XCLBIN", owner: "npu-parakeet", site: "npu-parakeet/src/npu.rs:902",
        semantics: Value, default: "unset",
        doc: "arbitrary override path for the resident xclbin (manual/debug knob, no guaranteed \
              final_{stem}.xclbin naming). Its mere PRESENCE (var_os(..).is_some(), line 2094) \
              also flips c_elem_bytes() to probe the loaded xclbin instead of assuming f32." },
    Flag { name: "PARAKEET_RELPOS_DIR", owner: "npu-parakeet", site: "npu-parakeet/src/npu.rs:1007",
        semantics: Value, default: "<root>/artifacts/relpos",
        doc: "alternative relpos artifact directory (A/B arm); must match whichever bucket table \
              relpos_buckets() returns." },
    Flag { name: "PARAKEET_RELPOS_NO_BUCKET", owner: "npu-parakeet", site: "npu-parakeet/src/npu.rs:1075",
        semantics: Presence, default: "false",
        doc: "forces the ceiling relpos bucket for every clip (pre-T-bucketing baseline, A/B)." },
    Flag { name: "PARAKEET_RELPOS_2BUCKET", owner: "npu-parakeet", site: "npu-parakeet/src/npu.rs:1080",
        semantics: Presence, default: "false",
        doc: "uses only the first + ceiling relpos bucket, skipping mid buckets (A/B); no effect \
              when PARAKEET_RELPOS_NO_BUCKET is also set." },
    Flag { name: "PARAKEET_RELPOS_TACTIVE_PROBE", owner: "npu-parakeet", site: "npu-parakeet/src/npu.rs:1217",
        semantics: Value, default: "unset",
        doc: "TIMING PROBE: dispatches a false t_active to partition the relpos dispatch cost. \
              Output is GARBAGE by construction -- timing only, never a correctness run." },
    Flag { name: "PARAKEET_LN_FUSED", owner: "npu-parakeet", site: "npu-parakeet/src/npu.rs:1573",
        semantics: NotZero, default: "true",
        doc: "fused ctxLN->affine_cast dispatch when built; =0 forces the two-dispatch chain back." },
    Flag { name: "PARAKEET_FC2_ONEDISPATCH", owner: "npu-parakeet", site: "npu-parakeet/src/npu.rs:2897",
        semantics: NotZero, default: "true",
        doc: "one-dispatch fc2 collapse path; declines outright (rather than mis-dispatching) when \
              the required apanel1024 insts artifact is missing." },
    Flag { name: "PARAKEET_FC2_K4096", owner: "npu-parakeet", site: "npu-parakeet/src/npu.rs:2946",
        semantics: NotZero, default: "false",
        doc: "one-dispatch K=DFF (4096) fc2 collapse (opt-in)." },
    Flag { name: "PARAKEET_FC1_PACK_IN_DRAIN", owner: "npu-parakeet", site: "npu-parakeet/src/npu.rs:2951",
        semantics: NotZero, default: "true",
        doc: "packs fc1's output during the drain (default on since 2026-07-28); =0 reverts to the fc1+deint pair." },
    Flag { name: "PARAKEET_PHASE_TIMING", owner: "npu-parakeet", site: "npu-parakeet/src/prof.rs:77",
        semantics: Presence, default: "false",
        doc: "phase-timing profiler (Npu/Host/Marshal buckets) for the encode path." },

    // -- npu-runtime ------------------------------------------------------------------------------
    Flag { name: "FFMPEG", owner: "npu-runtime", site: "npu-runtime/src/media.rs:18",
        semantics: Value, default: "\"ffmpeg\"",
        doc: "ffmpeg binary override, for a box where it is not on PATH." },
    Flag { name: "RUNTIME_DIRECTORY", owner: "npu-runtime", site: "npu-runtime/src/status_file.rs:30",
        semantics: Value, default: "unset",
        doc: "systemd RuntimeDirectory= for publishing the service status file; the unit sets \
              this. Falls back to XDG_RUNTIME_DIR when unset." },
    Flag { name: "XDG_RUNTIME_DIR", owner: "npu-runtime", site: "npu-runtime/src/status_file.rs:31",
        semantics: Value, default: "unset",
        doc: "user-session runtime dir fallback for the status file when RUNTIME_DIRECTORY is unset." },

    // -- npu-weights ------------------------------------------------------------------------------
    Flag { name: "XDNA_CHECKPOINT_DIR", owner: "npu-weights", site: "npu-weights/src/spec.rs:54",
        semantics: Value, default: "<root>/artifacts/checkpoints",
        doc: "overrides where baked checkpoint .safetensors files are written/read." },

    // -- npu-whisper ------------------------------------------------------------------------------
    Flag { name: "NPU_ENC_GELU_FUSED", owner: "npu-whisper", site: "npu-whisper/src/encoder.rs:116",
        semantics: IsOk, default: "false",
        doc: "folds GELU into fc1's on-chip epilogue at runtime (drops ~260 ms/utt host GELU). \
              COUPLING: npu-asr reads the SAME name (ctx2.rs) with the same is_ok() rule to pick \
              the matching xclbin artifact set -- see that entry; nothing ties the two together." },
    Flag { name: "NPU_ENC_MHA_NPU", owner: "npu-whisper", site: "npu-whisper/src/encoder.rs:138",
        semantics: NotZero, default: "true",
        doc: "encoder MHA on NPU by default (the comment names this the tuning.rs not_zero \
              convention explicitly); =0 opts out to host f32 -- a measured +2.6% latency \
              regression kept anyway for the single-hardware rule." },
    Flag { name: "NPU_ENC_CONV_NPU", owner: "npu-whisper", site: "npu-whisper/src/encoder.rs:185",
        semantics: IsOk, default: "false",
        doc: "routes the conv stem through the M-stationary GEMM conv (prebuilt 768 band)." },
    Flag { name: "NPU_ENC_MHA_MAXLAYER", owner: "npu-whisper", site: "npu-whisper/src/encoder.rs:321",
        semantics: Value, default: "usize::MAX (all layers)",
        doc: "caps NPU MHA to the first N encoder blocks; the bf16 attention error compounds over \
              layers, so a partial offload can stay WER-acceptable." },
    Flag { name: "ENC_PEROP_TIMING", owner: "npu-whisper", site: "npu-whisper/src/encoder.rs:417",
        semantics: IsOk, default: "false",
        doc: "per-op timing breakdown for the Whisper encoder forward pass." },

    // -- npu-xrt ----------------------------------------------------------------------------------
    Flag { name: "NPU_QUIET", owner: "npu-xrt", site: "npu-xrt/src/lib.rs:23",
        semantics: NotZero, default: "false",
        doc: "the ACTUAL consumer: suppresses load-time informational banners (precision picked, \
              resident xclbin loaded). Read once and cached -- must be set before the first model \
              load. npu-cli's quiet_one_shot() (this table, owner npu-cli) sets it to \"1\" ahead \
              of a one-shot command only when the caller expressed no preference; NPU_QUIET=0 \
              always forces the banners back on regardless of that." },
    Flag { name: "PARAKEET_SIM_BF16_GEMM_OUT", owner: "npu-xrt", site: "npu-xrt/src/lib.rs:50",
        semantics: NotZero, default: "false",
        doc: "narrows every resident-modal dispatch's f32 C buffer to bf16 in place, simulating a \
              bf16-out epilogue. Round-trips through the host: SLOW, measurement instrument only, \
              never a shipping path." },
    Flag { name: "PARAKEET_SIM_BF16_ONLY_N", owner: "npu-xrt", site: "npu-xrt/src/lib.rs:67",
        semantics: Value, default: "every width",
        doc: "restricts sim_bf16 narrowing to dispatches whose output is N wide, to bisect which \
              modal GEMM the fold's accuracy cost lives in." },
    Flag { name: "NPU_DISPATCH_LOG", owner: "npu-xrt", site: "npu-xrt/src/lib.rs:121",
        semantics: NotZero, default: "false",
        doc: "logs per-(xclbin, insts) dispatch blocking time and hw_context-transition counts." },
    Flag { name: "NPU_XCLBIN_CACHE_BY_CONTENT", owner: "npu-xrt", site: "npu-xrt/src/lib.rs:604",
        semantics: NotZero, default: "false",
        doc: "keys the hw_context cache on the xclbin's CONTENT hash instead of its path; a \
              diagnostic for finding a duplicate-path load, not the cure (the cure is one path)." },
];

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole reason this registry exists: `npu flags` and any future validator read exactly
    /// this list, so a knob cannot be invisible without also being unregistered.
    #[test]
    fn every_flag_has_a_name_site_and_doc() {
        for f in FLAGS {
            assert!(f.name.starts_with(|c: char| c.is_ascii_uppercase()), "{:?}: name not SCREAMING_CASE", f.name);
            assert!(f.site.contains(':'), "{}: site {:?} is not file:line", f.name, f.site);
            assert!(!f.doc.is_empty(), "{}: empty doc", f.name);
            assert!(!f.owner.is_empty(), "{}: empty owner", f.name);
            assert!(!f.default.is_empty(), "{}: empty default", f.name);
        }
    }

    /// A sanity floor, not a magic count (the same reasoning `completion_coverage.rs` uses for its
    /// own threshold): this inventory was built against a specific measured surface, so a FLAGS
    /// list that shrinks a lot is itself a signal something regressed silently.
    #[test]
    fn registry_is_not_vacuous() {
        assert!(FLAGS.len() > 60, "only {} flags registered -- census likely regressed", FLAGS.len());
    }

    #[test]
    fn semantics_describe_is_non_empty_for_every_variant() {
        for s in [NotZero, IsOne, IsOk, Presence, FalsyList, Value] {
            assert!(!s.describe().is_empty());
        }
    }
}
