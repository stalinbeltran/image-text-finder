"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SplitModel(BaseModel):
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1


class BuildPatchRequest(BaseModel):
    source: str = Field(..., description="Source dataset id (relative to DATASETS_ROOT) or absolute path.")
    name: str = Field(..., description="Output patch-dataset name (subdir under PATCH_DATASETS_DIR).")
    patch_size: int = 40
    stride: int = 20
    target_kinds: list[str] = ["paragraph"]
    drop_overlap: bool = False
    split: SplitModel = SplitModel()
    seed: int = 1


class ModelConfigModel(BaseModel):
    input_size: int = 40
    in_channels: int = 1
    backbone: list[dict]
    head: dict = {}
    border_features: bool = False


class TrainRequest(BaseModel):
    data: str = Field(..., description="Patch-dataset name (under PATCH_DATASETS_DIR) or absolute path.")
    name: str = Field(..., description="Run name (subdir under RUNS_DIR).")
    model: ModelConfigModel
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    optimizer: str = "adam"
    weight_decay: float = 0.0
    lambda_pos: float = 1.0
    pos_weight: float | None = None
    seed: int = 1
    device: str = "cpu"
    num_workers: int = 0


class SaveModelRequest(BaseModel):
    name: str
    model: ModelConfigModel


class PredictPathRequest(BaseModel):
    """Run inference on an image that already lives on the server's disk."""

    path: str = Field(..., description="Absolute path to a local image file.")
    run: str
    checkpoint: str = "best"
    threshold: float = 0.5
    stride: int | None = None
    device: str = "cpu"


class RenameRunRequest(BaseModel):
    new_name: str = Field(..., description="New run name (subdir under RUNS_DIR).")


class RetrainRequest(BaseModel):
    """Start a new run reusing an existing run's config, with optional overrides."""

    name: str = Field(..., description="New run name for the retrained model.")
    epochs: int | None = None
    batch_size: int | None = None
    lr: float | None = None
    optimizer: str | None = None
    lambda_pos: float | None = None
    device: str | None = None
