from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    dataset_root: str = ""
    processed_dir: str = "artifacts/processed"
    image_size: int = 224
    max_source_length: int = 32
    max_target_length: int = 128
    num_workers: int = 2
    train_csv: Optional[str] = None
    val_csv: Optional[str] = None
    test_csv: Optional[str] = None


@dataclass
class ModelConfig:
    vision_model_name: str = "google/vit-base-patch16-224-in21k"
    text_model_name: str = "google/flan-t5-base"
    observation_dim: int = 256
    alignment_dim: int = 256
    adapter_dim: int = 256
    num_observations: int = 14
    max_observation_phrases: int = 6
    dropout: float = 0.1
    freeze_vision_encoder: bool = False
    freeze_text_decoder: bool = False


@dataclass
class TrainConfig:
    output_dir: str = "artifacts/runs/default"
    seed: int = 42
    epochs: int = 5
    batch_size: int = 4
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_accum_steps: int = 1
    num_beams: int = 4
    lambda_nmi: float = 0.1
    lambda_obs: float = 0.5
    lambda_proto: float = 0.1
    lambda_sparse: float = 0.02
    fp16: bool = True
    log_every_n_steps: int = 20
    save_every_n_epochs: int = 1
    device: str = "cuda"


@dataclass
class EvalConfig:
    report_csv: str = ""
    batch_size: int = 4
    explainability_topk: int = 20
    run_faithfulness: bool = True


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        import yaml

        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
            eval=EvalConfig(**raw.get("eval", {})),
        )
