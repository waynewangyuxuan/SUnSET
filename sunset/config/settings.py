"""
SUnSET Configuration Module

Supports environment variables and YAML config files.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()


@dataclass
class LLMConfig:
    """LLM configuration for SET extraction."""
    # Provider: "openai" or "vllm"
    provider: str = "openai"

    # OpenAI settings
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"
    openai_base_url: Optional[str] = None

    # VLLM settings
    vllm_url: str = "http://ds-serv11.ucsd.edu:18000/v1"
    vllm_model: str = "Qwen/Qwen3-32B"

    # Generation parameters
    temperature: float = 0.0
    max_tokens: int = 4096

    def __post_init__(self):
        # Load from environment if not set
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class EmbeddingConfig:
    """Embedding configuration for event clustering."""
    # Provider: "api" for remote API, "local" for sentence-transformers
    provider: str = "local"

    # Local model (sentence-transformers)
    # Paper uses: Alibaba-NLP/gte-modernbert-base
    local_model: str = "Alibaba-NLP/gte-modernbert-base"

    # API settings (if provider="api")
    url: str = "http://ds-serv11.ucsd.edu:18003/v1"
    model: str = "qwen3-embed-0.6b"
    batch_size: int = 100


@dataclass
class WikidataConfig:
    """Wikidata configuration for coreference resolution."""
    proxy_url: str = "https://proxy.frederickpi.com/proxy/random/normal"
    cache_file: str = "entity_cache.json"
    max_concurrent: int = 5
    timeout: int = 30


@dataclass
class PipelineConfig:
    """Pipeline hyperparameters from paper."""
    # Relevance scoring (Equation 1)
    beta: float = 1.0

    # Graph construction (Equation 4)
    em_n: int = 1  # BoolEM_n threshold
    top_k: int = 20  # Top-k neighbors

    # Stakeholder constraints
    max_stakeholders_per_event: int = 5

    # Timeline generation
    max_timeline_entries: int = 50


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"  # "json" or "text"
    log_file: Optional[str] = None


@dataclass
class Config:
    """Main configuration container."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    wikidata: WikidataConfig = field(default_factory=WikidataConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Paths
    data_path: str = "timeline17.pkl"
    results_dir: str = "results"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "Config":
        """Create config from dictionary."""
        config = cls()

        if "llm" in data:
            config.llm = LLMConfig(**data["llm"])
        if "embedding" in data:
            config.embedding = EmbeddingConfig(**data["embedding"])
        if "wikidata" in data:
            config.wikidata = WikidataConfig(**data["wikidata"])
        if "pipeline" in data:
            config.pipeline = PipelineConfig(**data["pipeline"])
        if "logging" in data:
            config.logging = LoggingConfig(**data["logging"])
        if "data_path" in data:
            config.data_path = data["data_path"]
        if "results_dir" in data:
            config.results_dir = data["results_dir"]

        return config

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "llm": {
                "provider": self.llm.provider,
                "openai_model": self.llm.openai_model,
                "openai_base_url": self.llm.openai_base_url,
                "vllm_url": self.llm.vllm_url,
                "vllm_model": self.llm.vllm_model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
            },
            "embedding": {
                "url": self.embedding.url,
                "model": self.embedding.model,
                "batch_size": self.embedding.batch_size,
            },
            "wikidata": {
                "proxy_url": self.wikidata.proxy_url,
                "cache_file": self.wikidata.cache_file,
                "max_concurrent": self.wikidata.max_concurrent,
                "timeout": self.wikidata.timeout,
            },
            "pipeline": {
                "beta": self.pipeline.beta,
                "em_n": self.pipeline.em_n,
                "top_k": self.pipeline.top_k,
                "max_stakeholders_per_event": self.pipeline.max_stakeholders_per_event,
                "max_timeline_entries": self.pipeline.max_timeline_entries,
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "log_file": self.logging.log_file,
            },
            "data_path": self.data_path,
            "results_dir": self.results_dir,
        }


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration.

    Priority:
    1. YAML file if path provided
    2. config.yaml in current directory if exists
    3. Default config
    """
    if config_path and Path(config_path).exists():
        return Config.from_yaml(config_path)

    default_path = Path("config.yaml")
    if default_path.exists():
        return Config.from_yaml(str(default_path))

    return Config()
