from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any


DATA_ROOT_ENV_VAR = "QRESEARCH_DATA_ROOT"
DEFAULT_REGISTRY_NAME = "dataset_registry.json"


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").exists() and (parent / "data").exists():
            return parent
    raise FileNotFoundError("Could not locate repository root containing both 'src' and 'data'.")


def dataset_root(root: Path | str | None = None) -> Path:
    if root is not None:
        resolved = Path(root).expanduser().resolve()
    else:
        env_root = os.environ.get(DATA_ROOT_ENV_VAR)
        resolved = Path(env_root).expanduser().resolve() if env_root else (_repo_root() / "data").resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {resolved}")
    return resolved


def raw_dir(root: Path | str | None = None) -> Path:
    path = dataset_root(root) / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def processed_dir(root: Path | str | None = None) -> Path:
    path = dataset_root(root) / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def registry_path(root: Path | str | None = None) -> Path:
    return processed_dir(root) / DEFAULT_REGISTRY_NAME


@dataclass(frozen=True)
class DatasetArtifactMetadata:
    dataset_name: str
    path: str
    source_system: str
    schema_version: int
    format: str
    build_timestamp: str | None = None
    upstream_raw_inputs: list[str] = field(default_factory=list)
    source_quality: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def absolute_path(self, *, root: Path | str | None = None) -> Path:
        base = processed_dir(root)
        return (base / self.path).resolve()

    def sidecar_path(self, *, root: Path | str | None = None) -> Path:
        artifact = self.absolute_path(root=root)
        return metadata_sidecar_path(artifact)

    def write(self, *, root: Path | str | None = None, artifact_path: Path | str | None = None) -> Path:
        artifact = Path(artifact_path) if artifact_path is not None else self.absolute_path(root=root)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        sidecar = metadata_sidecar_path(artifact)
        sidecar.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return sidecar


def metadata_sidecar_path(path: Path | str) -> Path:
    artifact = Path(path)
    return artifact.with_suffix(".meta.json")


def load_dataset_registry(root: Path | str | None = None) -> dict[str, Any]:
    path = registry_path(root)
    if not path.exists():
        return {"datasets": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    datasets = data.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError(f"Invalid dataset registry at {path}: expected top-level 'datasets' object.")
    return data


def save_dataset_registry(registry: dict[str, Any], *, root: Path | str | None = None) -> Path:
    path = registry_path(root)
    path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    return path


def resolve_dataset_path(name: str, *, root: Path | str | None = None) -> Path:
    registry = load_dataset_registry(root)
    entry = registry.get("datasets", {}).get(name)
    if entry is None:
        raise KeyError(f"Unknown dataset '{name}'.")
    rel_path = entry.get("path")
    if not rel_path:
        raise ValueError(f"Dataset '{name}' is missing a canonical 'path'.")
    return (processed_dir(root) / rel_path).resolve()


def _metadata_from_registry_entry(name: str, entry: dict[str, Any]) -> DatasetArtifactMetadata:
    return DatasetArtifactMetadata(
        dataset_name=name,
        path=str(entry["path"]),
        source_system=str(entry["source_system"]),
        schema_version=int(entry["schema_version"]),
        format=str(entry["format"]),
        build_timestamp=entry.get("build_timestamp"),
        upstream_raw_inputs=list(entry.get("upstream_raw_inputs", [])),
        source_quality=entry.get("source_quality"),
        parameters=dict(entry.get("parameters", {})),
    )


def load_dataset_metadata(
    path_or_name: Path | str,
    *,
    root: Path | str | None = None,
) -> DatasetArtifactMetadata | None:
    if isinstance(path_or_name, Path) or str(path_or_name).endswith((".parquet", ".csv", ".jsonl", ".xlsx")):
        artifact = Path(path_or_name)
        sidecar = metadata_sidecar_path(artifact)
        if sidecar.exists():
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            return DatasetArtifactMetadata(**payload)
        return None

    name = str(path_or_name)
    registry = load_dataset_registry(root)
    entry = registry.get("datasets", {}).get(name)
    if entry is None:
        return None

    artifact = resolve_dataset_path(name, root=root)
    sidecar = metadata_sidecar_path(artifact)
    if sidecar.exists():
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        return DatasetArtifactMetadata(**payload)
    return _metadata_from_registry_entry(name, entry)
