"""Fixture-only shell gates for the public IRON source contract.

The fake git executable models source provenance without touching a real checkout.
"""
import os
from pathlib import Path
import subprocess


AMD_PATHS = Path(__file__).resolve().parents[2] / "amd_paths.sh"
WANT = "a" * 40
OTHER = "b" * 40


def fixture_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "third_party" / "iron").mkdir(parents=True)
    (repo / "toolchain.lock").write_text(f"IRON_SOURCE_COMMIT={WANT}\n")
    return repo


def fake_git(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    git = bindir / "git"
    git.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "args=(\"$@\")\n"
        "for i in \"${!args[@]}\"; do\n"
        "  case \"${args[$i]}\" in\n"
        "    rev-parse)\n"
        "      next=${args[$((i + 1))]:-}\n"
        "      [ \"${FAKE_NO_GIT:-}\" != 1 ] || exit 1\n"
        "      if [ \"$next\" = \"--is-inside-work-tree\" ]; then echo true; else echo \"${FAKE_HEAD:?}\"; fi\n"
        "      exit 0 ;;\n"
        "    status) printf '%s' \"${FAKE_DIRTY:-}\"; exit 0 ;;\n"
        "    diff) printf '%s' \"${FAKE_DIFF:-}\"; exit 0 ;;\n"
        "    ls-files) printf '%b' \"${FAKE_UNTRACKED:-}\"; exit 0 ;;\n"
        "    hash-object) echo \"${FAKE_UNTRACKED_HASH:-}\"; exit 0 ;;\n"
        "  esac\n"
        "done\n"
        "exit 1\n"
    )
    git.chmod(0o755)
    return bindir


def run_contract(repo, bindir, command, **extra_env):
    env = {
        "PATH": f"{bindir}:/usr/bin:/bin",
        "REPO": str(repo),
        "IRON_LOCK": str(repo / "toolchain.lock"),
        "FAKE_HEAD": WANT,
        **extra_env,
    }
    return subprocess.run(
        ["bash", "-c", f'. "{AMD_PATHS}"; {command}'],
        capture_output=True,
        text=True,
        env=env,
    )


def test_no_local_env_or_author_path_is_needed_for_the_exact_default_submodule_source(tmp_path):
    repo = fixture_repo(tmp_path)
    bindir = fake_git(tmp_path)
    result = run_contract(repo, bindir, 'iron_require_source; printf "%s" "$IRON_SOURCE_IDENTITY"')
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"pinned:{WANT}"


def test_missing_or_uninitialized_default_dependency_fails_before_build(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "toolchain.lock").write_text(f"IRON_SOURCE_COMMIT={WANT}\n")
    bindir = fake_git(tmp_path)
    result = run_contract(repo, bindir, "iron_require_source")
    assert result.returncode != 0
    assert "missing or uninitialized" in result.stderr


def test_wrong_ref_is_rejected_without_an_explicit_development_override(tmp_path):
    repo = fixture_repo(tmp_path)
    bindir = fake_git(tmp_path)
    result = run_contract(repo, bindir, "iron_require_source", FAKE_HEAD=OTHER)
    assert result.returncode != 0
    assert WANT[:12] in result.stderr
    assert OTHER[:12] in result.stderr


def test_explicit_unpinned_override_records_the_actual_source_identity(tmp_path):
    repo = fixture_repo(tmp_path)
    bindir = fake_git(tmp_path)
    result = run_contract(
        repo,
        bindir,
        'iron_require_source; printf "%s" "$IRON_SOURCE_IDENTITY"',
        FAKE_HEAD=OTHER,
        IRON_ALLOW_UNPINNED="1",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"override:{OTHER}"


def test_dirty_source_is_rejected_or_recorded_as_dirty_when_explicitly_allowed(tmp_path):
    repo = fixture_repo(tmp_path)
    bindir = fake_git(tmp_path)
    dirty = " M iron/operators/gemm/design.py\n"
    rejected = run_contract(repo, bindir, "iron_require_source", FAKE_DIRTY=dirty)
    assert rejected.returncode != 0
    assert "dirty" in rejected.stderr.lower()

    accepted = run_contract(
        repo,
        bindir,
        'iron_require_source; printf "%s" "$IRON_SOURCE_IDENTITY"',
        FAKE_DIRTY=dirty,
        FAKE_DIFF="diff --git a/x b/x\n",
        IRON_ALLOW_DIRTY="1",
    )
    assert accepted.returncode == 0, accepted.stderr
    assert accepted.stdout.startswith(f"dirty:{WANT}:")


def test_dirty_identity_changes_when_an_untracked_file_changes(tmp_path):
    repo = fixture_repo(tmp_path)
    bindir = fake_git(tmp_path)
    common = {
        "FAKE_DIRTY": "?? generated.py\\n",
        "FAKE_UNTRACKED": "generated.py\\0",
        "FAKE_DIFF": "",
        "IRON_ALLOW_DIRTY": "1",
    }
    first = run_contract(
        repo, bindir, 'iron_require_source; printf "%s" "$IRON_SOURCE_IDENTITY"',
        FAKE_UNTRACKED_HASH="1" * 64, **common,
    )
    second = run_contract(
        repo, bindir, 'iron_require_source; printf "%s" "$IRON_SOURCE_IDENTITY"',
        FAKE_UNTRACKED_HASH="2" * 64, **common,
    )
    assert first.returncode == second.returncode == 0
    assert first.stdout != second.stdout


def test_replay_snapshot_requires_verified_file_content(tmp_path):
    repo = fixture_repo(tmp_path)
    bindir = fake_git(tmp_path)
    source = repo / "third_party" / "iron" / "source.py"
    source.write_text("original\n")
    digest = subprocess.check_output(["sha256sum", source], text=True).split()[0]
    snapshot = tmp_path / "iron-source.snapshot"
    snapshot.write_text(
        "iron-source-snapshot-v1\n"
        f"identity\tpinned:{WANT}\n"
        f"dir\t{repo / 'third_party' / 'iron'}\n"
        f"file\t{digest}\t{source}\n"
    )
    accepted = run_contract(
        repo, bindir, 'iron_require_source; printf "%s" "$IRON_SOURCE_IDENTITY"',
        FAKE_NO_GIT="1", IRON_SOURCE_SNAPSHOT=str(snapshot),
    )
    assert accepted.returncode == 0, accepted.stderr
    source.write_text("mutated\n")
    rejected = run_contract(
        repo, bindir, "iron_require_source",
        FAKE_NO_GIT="1", IRON_SOURCE_SNAPSHOT=str(snapshot),
    )
    assert rejected.returncode != 0
    snapshot.write_text(
        "iron-source-snapshot-v1\n"
        f"identity\tpinned:{WANT}\n"
        f"dir\t{repo / 'third_party' / 'iron'}\n"
    )
    partial = run_contract(
        repo, bindir, "iron_require_source",
        FAKE_NO_GIT="1", IRON_SOURCE_SNAPSHOT=str(snapshot),
    )
    assert partial.returncode != 0


def test_missing_fused_attention_sibling_fails_before_recipe_execution(tmp_path):
    repo = fixture_repo(tmp_path)
    bindir = fake_git(tmp_path)
    missing = run_contract(repo, bindir, "iron_require_fused_attn")
    assert missing.returncode != 0
    assert "fused_attn.cc" in missing.stderr

    source = repo / "third_party" / "iron" / "aie_kernels" / "aie2p"
    source.mkdir(parents=True)
    (source / "fused_attn.cc").write_text("fixture\n")
    found = run_contract(repo, bindir, "iron_require_fused_attn")
    assert found.returncode == 0, found.stderr


def test_public_bootstrap_declares_recursive_source_and_environment_order():
    source = (AMD_PATHS.parent / "bootstrap_public_env.sh").read_text()
    assert "git submodule update --init --recursive" in source
    assert source.index("iron_require_source") < source.index("scripts/setup_kernel_env.sh")
    assert source.index("scripts/setup_kernel_env.sh") < source.index("scripts/fetch_mlir_distro.sh")
    assert source.index("scripts/fetch_mlir_distro.sh") < source.index("scripts/toolchain_up.sh")
    assert source.index("scripts/toolchain_up.sh") < source.index("scripts/setup_export_venv.sh")


def test_bootstrap_uses_declared_environment_requirements():
    repo = AMD_PATHS.parents[1]
    kernel_env = (repo / "scripts" / "setup_kernel_env.sh").read_text()
    export_env = (repo / "scripts" / "setup_export_venv.sh").read_text()
    assert '. "$REPO/toolchain.lock"' in kernel_env
    assert "-r scripts/requirements-export.txt" in export_env


def test_install_has_no_author_venv_discovery_default():
    install = (AMD_PATHS.parents[1] / "install.sh").read_text()
    assert "npuvox-asr-bench" not in install
    assert "ONNX_ASR_VENV_CANDIDATES" not in install
    assert "IRON_SOURCE_COMMIT" in install
    assert "iron-artifact-floor" not in install
    assert "for key in decode prefill resident" in install
    assert "check_iron_artifact_provenance.py" in install


def test_newstack_compat_retires_the_silent_worker_monkeypatch():
    compat = (AMD_PATHS.parents[1] / "designs" / "decode_fused" / "newstack_compat.py").read_text()
    assert "_drop_allocation_scheme" not in compat


def test_host_quant_references_default_to_the_declared_iron_checkout():
    repo = AMD_PATHS.parents[1]
    for relative in ("scripts/llm_hf_bf16_ref.py", "scripts/qwen35_ref.py"):
        source = (repo / relative).read_text()
        assert "wt-iron-integ" not in source
        assert '"third_party", "iron"' in source
