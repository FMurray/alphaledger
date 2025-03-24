from pydantic_settings import BaseSettings, SettingsConfigDict, CliImplicitFlag
from pathlib import Path
from pydantic import Field, field_validator, AliasChoices
from typing import Optional, Literal
import os

from rich import print
from rich.console import Console
from pydantic import BaseModel, ValidationError


class AlphaLedgerSettings(BaseSettings):
    """Global configuration settings for AlphaLedger."""

    model_config = SettingsConfigDict(
        env_prefix="ALPHALEDGER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        cli_parse_args=True,  # Enable command-line parsing
        cli_kebab_case=True,  # Use kebab-case for CLI arguments
    )

    # Base directory for universes
    universe_dir: Path = Field(
        default=None,  # Default set in validator after checking .env
        description="Base directory containing universe definitions",
    )

    # Output directory for knowledge base
    output_dir: Path = Field(
        default=None,  # Default set in validator after checking .env
        description="Directory to store output files",
    )

    # Databricks settings (not prefixed with ALPHALEDGER_)
    databricks_host: Optional[str] = None
    databricks_token: Optional[str] = None

    # Azure OpenAI settings
    azure_openai_api_key: Optional[str] = None
    azure_openai_api_base: Optional[str] = None
    azure_openai_api_version: Optional[str] = None
    azure_openai_embeding_deployment_id: Optional[str] = None
    azure_openai_embeding_endpoint: Optional[str] = None
    # SEC API settings
    sec_user_agent: str = Field(
        default="AlphaLedger (contact@alphaledger.com)",
        description="Your name, email, and organization as required by SEC",
    )
    sec_request_interval: float = Field(
        default=0.1,
        description="Time between SEC API requests in seconds (SEC rate limit is 10 requests/second)",
    )
    sec_base_url: str = Field(
        default="https://www.sec.gov/Archives/",
        description="Base URL for SEC EDGAR archives",
    )

    # Knowledge base specific settings
    universe_name: str = Field(
        default="sectors/cloud_computing",
        description="The universe of securities to analyze",
        validation_alias=AliasChoices("universe", "u"),
    )
    kb_depth: int = Field(
        default=1,
        description="Depth of information to collect for knowledge base",
        validation_alias=AliasChoices("depth", "d"),
    )
    start_year: Optional[int] = Field(
        default=2022, description="Start year for analysis period"
    )
    end_year: Optional[int] = Field(
        default=2023, description="End year for analysis period"
    )
    verbose: CliImplicitFlag[bool] = Field(
        default=True,
        description="Enable verbose output",
        validation_alias=AliasChoices("verbose", "v"),
    )
    kb_uri: str = Field(
        default="",  # Temporary placeholder, will be set in validator
        description="URI for the knowledge base",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model to use for the knowledge base",
    )
    kb_index_type: str = Field(
        default="IVF_PQ",
        description="Type of index to use for the knowledge base",
    )
    kb_index_metric: Literal["l2", "cosine", "dot"] = Field(
        default="l2",
        description="Metric to use for the knowledge base index",
    )
    kb_index_num_partitions: int = Field(
        default=256,
        description="Number of partitions for the IVF_PQ index",
    )
    kb_index_num_sub_vectors: int = Field(
        default=96,  # This will be overridden by dimension/16 if not specified
        description="Number of sub-vectors for the IVF_PQ index",
    )
    kb_index_num_bits: int = Field(
        default=8,
        description="Number of bits for encoding each sub-vector (4 or 8)",
    )
    kb_accelerator: Optional[Literal["cuda", "mps"]] = Field(
        default=None,
        description="GPU acceleration for index building ('cuda' for NVIDIA GPUs, 'mps' for Apple Silicon)",
    )

    @field_validator("universe_dir", "output_dir", mode="before")
    def convert_to_path(cls, v):
        """Convert string paths to Path objects"""
        if v is not None:
            return Path(v)
        return v

    @field_validator("universe_dir", mode="after")
    def set_default_universe_dir(cls, v):
        """Set default universe directory if not provided"""
        if v is not None:
            return v

        # Check if set in environment first
        env_var = os.environ.get("ALPHALEDGER_UNIVERSE_DIR")
        if env_var:
            return Path(env_var)

        # Default to project structure if not specified
        return Path(__file__).parent.parent.parent / "universes"

    @field_validator("output_dir", mode="after")
    def set_default_output_dir(cls, v):
        """Set default output directory if not provided"""
        if v is not None:
            return v

        # Check if set in environment first
        env_var = os.environ.get("ALPHALEDGER_OUTPUT_DIR")
        if env_var:
            return Path(env_var)

        # Default to project structure if not specified
        return Path(__file__).parent.parent.parent / "output"

    @field_validator("kb_index_num_bits")
    def validate_num_bits(cls, v):
        """Validate num_bits is either 4 or 8"""
        if v not in [4, 8]:
            raise ValueError("num_bits must be either 4 or 8")
        return v

    @field_validator("kb_accelerator")
    def validate_accelerator(cls, v):
        """Validate accelerator is None, 'cuda', or 'mps'"""
        if v is not None and v not in ["cuda", "mps"]:
            raise ValueError("accelerator must be 'cuda', 'mps', or None")
        return v

    @field_validator("kb_uri", mode="after")
    def set_kb_uri(cls, v, info):
        """Set kb_uri based on output_dir if not explicitly provided"""
        if v:  # If kb_uri is explicitly set, use it
            return v

        # Get the output_dir from the data
        output_dir = info.data.get("output_dir")
        if output_dir:
            # Create a local URI using the output_dir
            return f"{output_dir}/kb_data"
        return "output/kb_data"  # Fallback default


console = Console()


def print_pydantic_errors(e: ValidationError):
    console.print("[bold red]Validation Errors[/bold red]")
    for error in e.errors():
        console.print(f"[bold magenta]Location:[/bold magenta] {error['loc']}")
        console.print(f"[bold magenta]Type:[/bold magenta] {error['type']}")
        console.print(f"[bold magenta]Message:[/bold magenta] {error['msg']}")
        console.print(
            f"[bold magenta]Input Value:[/bold magenta] {error.get('input', 'N/A')}"
        )
        console.print(
            f"[bold magenta]Context:[/bold magenta] {error.get('ctx', 'N/A')}"
        )
        console.print()  # Empty line for readability


try:
    # Create a global settings instance
    settings = AlphaLedgerSettings()
except ValidationError as e:
    print_pydantic_errors(e)
