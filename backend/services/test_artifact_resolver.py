from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from backend.config import Settings
from backend.services import artifact_resolver
from backend.services.artifact_resolver import ArtifactFile, clear_artifact_resolution_cache, resolve_workflow_artifacts


@pytest.fixture(autouse=True)
def reset_resolution_cache() -> None:
    clear_artifact_resolution_cache()
    yield
    clear_artifact_resolution_cache()


def _set_test_workflow(monkeypatch: pytest.MonkeyPatch, root: Path, files: dict[str, str]) -> dict[str, Path]:
    local_paths = {name: root / "local" / Path(remote).name for name, remote in files.items()}
    workflow = {
        name: ArtifactFile(remote, (local_paths[name],))
        for name, remote in files.items()
    }
    monkeypatch.setitem(artifact_resolver.ARTIFACTS, "test", workflow)
    return local_paths


def _settings(root: Path, token: str | None = None) -> Settings:
    return Settings(
        _env_file=None,
        FINSIGHT_HF_REPO_ID="test-owner/test-artifacts",
        HF_TOKEN=token,
        FINSIGHT_ARTIFACT_DIR=str(root / "runtime"),
        FINSIGHT_HF_REVISION="test-revision",
    )


def test_existing_local_artifacts_are_preferred(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    local = _set_test_workflow(monkeypatch, tmp_path, {"model": "risk/model.json", "metadata": "risk/meta.json"})
    for path in local.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("local", encoding="utf-8")

    snapshot_download = Mock(side_effect=AssertionError("Hugging Face should not be called"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))

    result = resolve_workflow_artifacts("test", _settings(tmp_path))

    assert result == {name: path.resolve() for name, path in local.items()}
    snapshot_download.assert_not_called()


@pytest.mark.parametrize("token, expected_token", [(None, False), ("test-token", "test-token")])
def test_missing_local_artifacts_download_only_allowlisted_files_and_token_is_optional(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    token: str | None,
    expected_token: str | bool,
) -> None:
    files = {"model": "risk/model.json", "metadata": "risk/meta.json"}
    _set_test_workflow(monkeypatch, tmp_path, files)
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        local_dir = Path(str(kwargs["local_dir"]))
        for remote_path in kwargs["allow_patterns"]:  # type: ignore[union-attr]
            destination = local_dir / str(remote_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("downloaded", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake_snapshot_download))
    result = resolve_workflow_artifacts("test", _settings(tmp_path, token))

    assert set(result) == set(files)
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "test-owner/test-artifacts"
    assert calls[0]["allow_patterns"] == list(files.values())
    assert calls[0]["token"] == expected_token
    assert calls[0]["revision"] == "test-revision"
    assert all(path.is_file() for path in result.values())


def test_partial_local_set_downloads_only_missing_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    files = {"model": "risk/model.json", "metadata": "risk/meta.json"}
    local = _set_test_workflow(monkeypatch, tmp_path, files)
    local["model"].parent.mkdir(parents=True, exist_ok=True)
    local["model"].write_text("local", encoding="utf-8")
    downloaded: list[str] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        local_dir = Path(str(kwargs["local_dir"]))
        for remote_path in kwargs["allow_patterns"]:  # type: ignore[union-attr]
            downloaded.append(str(remote_path))
            destination = local_dir / str(remote_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("downloaded", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake_snapshot_download))
    result = resolve_workflow_artifacts("test", _settings(tmp_path))

    assert result["model"] == local["model"].resolve()
    assert result["metadata"].read_text(encoding="utf-8") == "downloaded"
    assert downloaded == ["risk/meta.json"]


def test_resolution_is_cached_for_repeated_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_test_workflow(monkeypatch, tmp_path, {"index": "rag/index.faiss"})
    calls = 0

    def fake_snapshot_download(**kwargs: object) -> str:
        nonlocal calls
        calls += 1
        time.sleep(0.02)
        destination = Path(str(kwargs["local_dir"])) / "rag" / "index.faiss"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("downloaded", encoding="utf-8")
        return str(kwargs["local_dir"])

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake_snapshot_download))
    settings = _settings(tmp_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: resolve_workflow_artifacts("test", settings), range(2)))

    assert calls == 1
    assert results[0] == results[1]


def test_only_registered_workflows_can_be_resolved() -> None:
    with pytest.raises(ValueError, match="Unknown artifact workflow"):
        resolve_workflow_artifacts("../../unlisted", Settings(_env_file=None))


def test_workflow_allowlists_contain_only_required_runtime_artifacts() -> None:
    expected = {
        "risk": {"risk/xgboost_risk_model.json", "risk/model_metadata.json"},
        "stock": {
            "stock/return_lstm_stock_model.keras",
            "stock/return_feature_scaler.pkl",
            "stock/return_target_scaler.pkl",
            "stock/return_metadata.json",
            "stock/AAPL_engineered_features.csv",
        },
        "sentiment": {
            "sentiment/model_metadata.json",
            "sentiment/finbert_model/config.json",
            "sentiment/finbert_model/model.safetensors",
            "sentiment/finbert_model/special_tokens_map.json",
            "sentiment/finbert_model/tokenizer.json",
            "sentiment/finbert_model/tokenizer_config.json",
            "sentiment/finbert_model/vocab.txt",
        },
        "rag": {
            "rag/AAPL_annual_report_2024.faiss",
            "rag/AAPL_annual_report_2024_embeddings.jsonl",
            "rag/embedding_metadata.json",
            "rag/index_metadata.json",
        },
    }
    actual = {
        workflow: {artifact.repository_path for artifact in artifacts.values()}
        for workflow, artifacts in artifact_resolver.ARTIFACTS.items()
    }
    assert actual == expected


def test_missing_downloaded_file_has_clear_workflow_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_test_workflow(monkeypatch, tmp_path, {"index": "rag/index.faiss"})
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=Mock(return_value="")))

    with pytest.raises(FileNotFoundError, match=r"Required test artifact.*rag/index\.faiss"):
        resolve_workflow_artifacts("test", _settings(tmp_path))


def test_download_errors_do_not_expose_token_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_test_workflow(monkeypatch, tmp_path, {"model": "risk/model.json"})

    def fake_snapshot_download(**kwargs: object) -> str:
        raise RuntimeError("request failed with token=do-not-print-this")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake_snapshot_download))
    with pytest.raises(FileNotFoundError) as error:
        resolve_workflow_artifacts("test", _settings(tmp_path, "configured-test-token"))

    assert "do-not-print-this" not in str(error.value)
    assert "configured-test-token" not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__
