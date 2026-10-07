from __future__ import annotations

import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from backend.config import Settings, get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ArtifactFile:
    repository_path: str
    local_candidates: tuple[Path, ...]


def _artifact(repository_path: str, *local_paths: str) -> ArtifactFile:
    return ArtifactFile(repository_path, tuple(PROJECT_ROOT / path for path in local_paths))


# Only files loaded by the API services are allowlisted here.
ARTIFACTS: Mapping[str, Mapping[str, ArtifactFile]] = {
    "risk": {
        "model": _artifact("risk/xgboost_risk_model.json", "ml/risk_prediction/xgboost_risk_model.json"),
        "metadata": _artifact("risk/model_metadata.json", "ml/risk_prediction/model_metadata.json"),
    },
    "stock": {
        "model": _artifact("stock/return_lstm_stock_model.keras", "ml/stock_prediction/return_lstm_stock_model.keras"),
        "feature_scaler": _artifact("stock/return_feature_scaler.pkl", "ml/stock_prediction/return_feature_scaler.pkl", "data/processed/return_feature_scaler.pkl"),
        "target_scaler": _artifact("stock/return_target_scaler.pkl", "ml/stock_prediction/return_target_scaler.pkl", "data/processed/return_target_scaler.pkl"),
        "metadata": _artifact("stock/return_metadata.json", "ml/stock_prediction/return_metadata.json", "data/processed/return_metadata.json"),
        "engineered_data": _artifact("stock/AAPL_engineered_features.csv", "data/processed/AAPL_engineered_features.csv"),
    },
    "sentiment": {
        "metadata": _artifact("sentiment/model_metadata.json", "ml/sentiment/model_metadata.json"),
        "model_config": _artifact("sentiment/finbert_model/config.json", "ml/sentiment/finbert_model/config.json"),
        "model_weights": _artifact("sentiment/finbert_model/model.safetensors", "ml/sentiment/finbert_model/model.safetensors"),
        "special_tokens": _artifact("sentiment/finbert_model/special_tokens_map.json", "ml/sentiment/finbert_model/special_tokens_map.json"),
        "tokenizer": _artifact("sentiment/finbert_model/tokenizer.json", "ml/sentiment/finbert_model/tokenizer.json"),
        "tokenizer_config": _artifact("sentiment/finbert_model/tokenizer_config.json", "ml/sentiment/finbert_model/tokenizer_config.json"),
        "vocabulary": _artifact("sentiment/finbert_model/vocab.txt", "ml/sentiment/finbert_model/vocab.txt"),
    },
    "rag": {
        "index": _artifact("rag/AAPL_annual_report_2024.faiss", "rag/retrieval/AAPL_annual_report_2024.faiss"),
        "index_metadata": _artifact("rag/index_metadata.json", "rag/retrieval/index_metadata.json"),
        "embedding_metadata": _artifact("rag/embedding_metadata.json", "rag/embeddings/embedding_metadata.json"),
        "embeddings": _artifact("rag/AAPL_annual_report_2024_embeddings.jsonl", "rag/embeddings/AAPL_annual_report_2024_embeddings.jsonl"),
    },
}

_resolution_lock = threading.RLock()
_resolved_workflows: dict[tuple[str, str, str | None, str], dict[str, Path]] = {}


def resolve_workflow_artifacts(workflow: str, settings: Settings | None = None) -> dict[str, Path]:
    """Resolve a workflow's local artifacts, downloading only its missing allowlisted files."""
    if workflow not in ARTIFACTS:
        raise ValueError(f"Unknown artifact workflow: {workflow!r}.")

    settings = settings or get_settings()
    artifact_dir_value = settings.FINSIGHT_ARTIFACT_DIR or str(Path(tempfile.gettempdir()) / "finsight-artifacts")
    artifact_dir = Path(artifact_dir_value).expanduser().resolve()
    cache_key = (workflow, settings.FINSIGHT_HF_REPO_ID, settings.FINSIGHT_HF_REVISION, str(artifact_dir))

    with _resolution_lock:
        cached = _resolved_workflows.get(cache_key)
        if cached is not None and all(path.is_file() for path in cached.values()):
            return dict(cached)

        workflow_artifacts = ARTIFACTS[workflow]
        resolved: dict[str, Path] = {}
        missing: dict[str, Path] = {}
        sentiment_model_names = {
            "model_config",
            "model_weights",
            "special_tokens",
            "tokenizer",
            "tokenizer_config",
            "vocabulary",
        }
        local_sentiment_model_complete = workflow != "sentiment" or all(
            any(path.is_file() for path in workflow_artifacts[name].local_candidates)
            for name in sentiment_model_names
        )
        for name, artifact in workflow_artifacts.items():
            local_path = next((path for path in artifact.local_candidates if path.is_file()), None)
            if workflow == "sentiment" and name in sentiment_model_names and not local_sentiment_model_complete:
                local_path = None
            runtime_path = artifact_dir / artifact.repository_path
            if local_path is not None:
                resolved[name] = local_path.resolve()
            elif runtime_path.is_file():
                resolved[name] = runtime_path.resolve()
            else:
                missing[name] = runtime_path

        if missing:
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:  # pragma: no cover - declared dependency
                raise FileNotFoundError(
                    f"Cannot resolve {workflow} artifacts: huggingface_hub is not installed."
                ) from exc

            repository_paths = [workflow_artifacts[name].repository_path for name in missing]
            options = {
                "repo_id": settings.FINSIGHT_HF_REPO_ID,
                "allow_patterns": repository_paths,
                "local_dir": str(artifact_dir),
                # False explicitly prevents implicit credentials when HF_TOKEN is unset.
                "token": settings.HF_TOKEN or False,
            }
            if settings.FINSIGHT_HF_REVISION:
                options["revision"] = settings.FINSIGHT_HF_REVISION
            try:
                snapshot_download(**options)
            except Exception:
                missing_names = ", ".join(repository_paths)
                raise FileNotFoundError(
                    f"Unable to download required {workflow} artifacts ({missing_names}) "
                    f"from Hugging Face repository {settings.FINSIGHT_HF_REPO_ID!r}. "
                    "Check repository access, network availability, and artifact files."
                ) from None

            for name, path in missing.items():
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Required {workflow} artifact is missing after Hugging Face resolution: "
                        f"{workflow_artifacts[name].repository_path}."
                    )
                resolved[name] = path.resolve()

        if set(resolved) != set(workflow_artifacts):
            missing_names = sorted(set(workflow_artifacts) - set(resolved))
            raise FileNotFoundError(
                f"Unable to resolve required {workflow} artifacts: {', '.join(missing_names)}."
            )

        _resolved_workflows[cache_key] = dict(resolved)
        return resolved


def clear_artifact_resolution_cache() -> None:
    """Clear process-local resolution state for tests or controlled reloads."""
    with _resolution_lock:
        _resolved_workflows.clear()
