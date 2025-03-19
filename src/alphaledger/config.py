from pydantic_settings import BaseSettings, SettingsConfigDict, CliImplicitFlag
from pathlib import Path
from pydantic import Field, field_validator, AliasChoices
from typing import Optional
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
