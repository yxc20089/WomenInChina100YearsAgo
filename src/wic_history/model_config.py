"""Load the single authoritative model/runtime configuration."""

from __future__ import annotations

import hashlib
import os
import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


DEFAULT_CONFIG_PATH = Path("config/pipeline-models.toml")
CONFIG_PATH_ENVIRONMENT_VARIABLE = "WIC_PIPELINE_MODEL_CONFIG"
SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
REVISION_PATTERN = r"^[0-9a-f]{40}$"
SPOTTING_JSON_PROMPT = (
    "检测并识别图中所有的文字行，请按从上到下、从左到右的阅读顺序进行识别。 "
    "输出格式为 JSON 数组，每个元素必须包含："
    '"box": [xmin, ymin, xmax, ymax]（坐标需归一化到 [0, 1000] 范围内）；'
    '"text": "识别出的文字内容"。 '
    "注意：请直接输出 JSON 数组，不要包含任何多余的描述性文字。"
)
LAYOUT_PARSE_PROMPT = (
    "提取文档图片中所有内容用markdown格式表示，表格用html格式表达，"
    "文档中公式用latex格式表示，请按照阅读顺序组织进行全文解析，并输出版式分析信息。"
)


class FrozenConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HunyuanOCRModel(FrozenConfiguration):
    """The one learned OCR/layout model allowed by the ingestion contract."""

    engine: Literal["transformers"]
    model_name: str = Field(min_length=1)
    model_revision: str = Field(pattern=REVISION_PATTERN)
    toolkit_name: str = Field(min_length=1)
    toolkit_revision: str = Field(pattern=REVISION_PATTERN)
    pipeline: Literal["spotting_json+layout_parse"]
    spotting_task: Literal["spotting_json"]
    spotting_prompt: str = Field(min_length=1)
    layout_task: Literal["layout_parse"]
    layout_prompt: str = Field(min_length=1)
    language: Literal["zh-Hant"]
    temperature: Literal[0.0]
    top_p: Literal[1.0]
    runtime: Literal["transformers-cuda"]
    dtype: Literal["bfloat16"]
    device: str = Field(pattern=r"^cuda(?::[0-9]+)?$")
    max_new_tokens: int = Field(gt=0, le=32768)
    repetition_penalty: float = Field(ge=1, le=2)
    confidence_status: Literal["not_emitted_by_model"]
    confidence_calibration: Literal["not_available"]

    @model_validator(mode="after")
    def validate_official_prompts(self) -> "HunyuanOCRModel":
        # These strings are frozen at the pinned toolkit revision. Accepting a
        # free-form near-equivalent would change the model contract while
        # retaining the same identity.
        if self.spotting_prompt != SPOTTING_JSON_PROMPT:
            raise ValueError("spotting_prompt must equal the pinned official prompt")
        if self.layout_prompt != LAYOUT_PARSE_PROMPT:
            raise ValueError("layout_prompt must equal the pinned official prompt")
        return self

class OllamaSemanticModel(FrozenConfiguration):
    """Local pinned semantic deployment with verifiable runtime and weights."""

    provider: Literal["ollama"]
    base_url: str = Field(pattern=r"^https?://")
    served_model: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_revision: str = Field(pattern=REVISION_PATTERN)
    ollama_manifest_digest: str = Field(pattern=SHA256_PATTERN)
    model_blob_sha256: str = Field(pattern=SHA256_PATTERN)
    quantization: str = Field(min_length=1)
    runtime_name: Literal["ollama"]
    runtime_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    runtime_executable: Path
    runtime_executable_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    thinking: Literal[False]
    temperature: Literal[0.0]
    seed: int
    context_length: int = Field(ge=1024)
    max_output_tokens: int = Field(gt=0, le=32768)
    timeout_seconds: float = Field(gt=0, le=600)
    keep_alive: str = Field(min_length=1)
    structured_output: Literal["native_json_schema"]
    acceleration: Literal["none", "draft-mtp"]
    mtp_environment_variable: Literal["LLAMA_ARG_SPEC_TYPE"]
    mtp_environment_value: Literal["draft-mtp"]
    schema_canary_repetitions: int = Field(ge=2, le=20)
    expected_canary_raw_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_schema_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_format_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def provenance_identity(self) -> dict[str, Any]:
        """Exact recorded identity for run provenance and receipts."""
        return {
            "provider": self.provider,
            "endpoint": self.base_url,
            "served_model": self.served_model,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "ollama_manifest_digest": self.ollama_manifest_digest,
            "model_blob_sha256": self.model_blob_sha256,
            "quantization": self.quantization,
            "runtime_name": self.runtime_name,
            "runtime_version": self.runtime_version,
            "acceleration": self.acceleration,
        }


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterSemanticModel(FrozenConfiguration):
    """Hosted OpenAI-compatible semantic deployment behind OpenRouter.

    OpenRouter exposes neither an immutable model revision nor weight hashes,
    so those provenance fields are pinned to explicit unavailability markers.
    The API key is read from the environment only and is never part of this
    configuration, artifacts, or logs.
    """

    provider: Literal["openrouter"]
    base_url: Literal["https://openrouter.ai/api/v1"]
    served_model: str = Field(pattern=r"^[a-z0-9-]+/[A-Za-z0-9.:_-]+$")
    model_name: str = Field(min_length=1)
    api_key_environment_variable: Literal["OPENROUTER_API_KEY"]
    model_revision_status: Literal["not_available"]
    weight_hash_status: Literal["not_available"]
    quantization_status: Literal["not_disclosed_by_provider"]
    thinking: Literal[False]
    temperature: Literal[0.0]
    seed: int
    context_length: int = Field(ge=1024)
    max_output_tokens: int = Field(gt=0, le=32768)
    timeout_seconds: float = Field(gt=0, le=300)
    structured_output: Literal["openai_response_format_json_schema"]

    def provenance_identity(self) -> dict[str, Any]:
        """Honest recorded identity: unavailable fields stay explicitly so."""
        return {
            "provider": self.provider,
            "endpoint": self.base_url,
            "served_model": self.served_model,
            "model_name": self.model_name,
            "model_revision": self.model_revision_status,
            "weight_hashes": self.weight_hash_status,
            "quantization": self.quantization_status,
            "runtime_name": self.provider,
            "runtime_version": "not_available",
            "acceleration": "not_applicable",
        }


SemanticModel = Annotated[
    OllamaSemanticModel | OpenRouterSemanticModel,
    Field(discriminator="provider"),
]


class EmbeddingModel(FrozenConfiguration):
    engine: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_revision: str = Field(pattern=REVISION_PATTERN)
    dimension: int = Field(gt=0)
    normalize: bool


class RerankerModel(FrozenConfiguration):
    engine: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_revision: str = Field(pattern=REVISION_PATTERN)


class RetrievalModels(FrozenConfiguration):
    passage_embedding: EmbeddingModel


class IdentityModels(FrozenConfiguration):
    enabled: Literal[False]
    embedding: EmbeddingModel
    reranker: RerankerModel


class PipelineModelConfiguration(FrozenConfiguration):
    schema_version: Literal[1]
    ocr: HunyuanOCRModel
    semantic: SemanticModel
    retrieval: RetrievalModels
    identity: IdentityModels
    source_path: Path = Field(exclude=True)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def layout(self) -> HunyuanOCRModel:
        """Compatibility view: layout and OCR are the same pinned model."""
        return self.ocr


def resolve_configuration_path(path: str | Path | None = None) -> Path:
    """Resolve the complete configuration path without model-level overrides."""
    if path is not None:
        return Path(path).expanduser().resolve()
    configured = os.environ.get(CONFIG_PATH_ENVIRONMENT_VARIABLE)
    if configured:
        return Path(configured).expanduser().resolve()
    working_tree_path = (Path.cwd() / DEFAULT_CONFIG_PATH).resolve()
    if working_tree_path.is_file():
        return working_tree_path
    repository_path = (Path(__file__).resolve().parents[2] / DEFAULT_CONFIG_PATH).resolve()
    return repository_path


def load_pipeline_model_configuration(
    path: str | Path | None = None,
) -> PipelineModelConfiguration:
    """Load, hash, and strictly validate the complete model configuration."""
    resolved = resolve_configuration_path(path)
    raw = resolved.read_bytes()
    payload = tomllib.loads(raw.decode("utf-8"))
    return PipelineModelConfiguration.model_validate(
        {
            **payload,
            "source_path": resolved,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    )
