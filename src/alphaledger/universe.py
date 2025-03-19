import json
import yaml
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import os
from alphaledger.config import settings
import datetime


@dataclass
class Security:
    """Represents a publicly traded security."""

    ticker: str
    name: str
    exchange: str
    sector: Optional[str] = None  # GICS sector
    industry: Optional[str] = None  # GICS industry
    currency: str = "USD"
    country: str = "US"

    # Additional classification fields
    custom_sector: Optional[str] = None  # For custom/informal classification
    custom_industry: Optional[str] = None  # For custom/informal classification
    subsector: Optional[str] = None  # More granular than GICS sector
    theme: Optional[List[str]] = None  # Thematic classifications (cloud, AI, etc.)

    def __str__(self) -> str:
        return f"{self.ticker} - {self.name} ({self.exchange})"


class Universe:
    """A collection of securities that defines the investment universe."""

    def __init__(
        self,
        name: str = "Default Universe",
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ):
        self.name = name
        self.securities: Dict[str, Security] = {}
        self.start_year = start_year
        self.end_year = end_year

    def add_security(self, security: Security) -> None:
        """Add a security to the universe."""
        self.securities[security.ticker] = security

    def remove_security(self, ticker: str) -> None:
        """Remove a security from the universe by ticker."""
        if ticker in self.securities:
            del self.securities[ticker]

    def get_security(self, ticker: str) -> Optional[Security]:
        """Get a security by ticker."""
        return self.securities.get(ticker)

    def get_all_securities(self) -> List[Security]:
        """Get all securities in the universe."""
        return list(self.securities.values())

    def get_tickers(self) -> List[str]:
        """Get all tickers in the universe."""
        return list(self.securities.keys())

    def __len__(self) -> int:
        return len(self.securities)

    def get_filing_years(self) -> List[int]:
        """Get the range of years to consider for SEC filings."""
        current_year = datetime.datetime.now().year

        start = self.start_year if self.start_year is not None else current_year - 5
        end = self.end_year if self.end_year is not None else current_year

        return list(range(start, end + 1))

    def __str__(self) -> str:
        time_range = ""
        if self.start_year or self.end_year:
            start = self.start_year if self.start_year is not None else "earliest"
            end = self.end_year if self.end_year is not None else "latest"
            time_range = f" [{start}-{end}]"
        return f"{self.name}{time_range} ({len(self)} securities)"


def load_from_json(
    filepath: str, start_year: Optional[int] = None, end_year: Optional[int] = None
) -> Universe:
    """
    Load a universe of securities from a JSON file.

    Args:
        filepath: Path to the JSON file
        start_year: Optional start year for filtering SEC filings
        end_year: Optional end year for filtering SEC filings
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(path, "r") as f:
        data = json.load(f)

    universe = Universe(
        name=data.get("name", "Default Universe"),
        start_year=start_year,
        end_year=end_year,
    )

    for sec_data in data.get("securities", []):
        security = Security(
            ticker=sec_data["ticker"],
            name=sec_data["name"],
            exchange=sec_data["exchange"],
            sector=sec_data.get("sector"),
            industry=sec_data.get("industry"),
            currency=sec_data.get("currency", "USD"),
            country=sec_data.get("country", "US"),
        )
        universe.add_security(security)

    return universe


def load_from_yaml(
    filepath: str, start_year: Optional[int] = None, end_year: Optional[int] = None
) -> Universe:
    """
    Load a universe of securities from a YAML file.

    Args:
        filepath: Path to the YAML file
        start_year: Optional start year for filtering SEC filings
        end_year: Optional end year for filtering SEC filings
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    universe = Universe(
        name=data.get("name", "Default Universe"),
        start_year=start_year,
        end_year=end_year,
    )

    for sec_data in data.get("securities", []):
        security = Security(
            ticker=sec_data["ticker"],
            name=sec_data["name"],
            exchange=sec_data["exchange"],
            sector=sec_data.get("sector"),
            industry=sec_data.get("industry"),
            currency=sec_data.get("currency", "USD"),
            country=sec_data.get("country", "US"),
        )
        universe.add_security(security)

    return universe


def save_to_json(universe: Universe, filepath: str) -> None:
    """Save a universe to a JSON file."""
    data = {
        "name": universe.name,
        "securities": [
            {
                "ticker": s.ticker,
                "name": s.name,
                "exchange": s.exchange,
                "sector": s.sector,
                "industry": s.industry,
                "currency": s.currency,
                "country": s.country,
            }
            for s in universe.get_all_securities()
        ],
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def save_to_yaml(universe: Universe, filepath: str) -> None:
    """Save a universe to a YAML file."""
    data = {
        "name": universe.name,
        "securities": [
            {
                "ticker": s.ticker,
                "name": s.name,
                "exchange": s.exchange,
                "sector": s.sector,
                "industry": s.industry,
                "currency": s.currency,
                "country": s.country,
            }
            for s in universe.get_all_securities()
        ],
    }

    with open(filepath, "w") as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)


def get_universe_path(universe_name: str) -> Path:
    """
    Get the path to a universe file by name.

    Args:
        universe_name: Name of the universe file (with or without extension)

    Returns:
        Path object to the universe file

    Raises:
        FileNotFoundError: If the universe file cannot be found
    """
    # Use the global settings for universe directory
    universe_dir = settings.universe_dir

    # If universe_name is an absolute path, use it directly
    if os.path.isabs(universe_name):
        path = Path(universe_name)
        if path.exists():
            return path
        raise FileNotFoundError(f"Universe file not found at: {path}")

    # Handle both filenames with and without extensions
    name_with_ext = (
        universe_name
        if (universe_name.endswith(".yaml") or universe_name.endswith(".json"))
        else None
    )
    name_without_ext = universe_name.split(".")[0] if name_with_ext else universe_name

    # First try direct lookup without walking (more efficient)
    direct_paths = [
        universe_dir / name_with_ext if name_with_ext else None,
        universe_dir / f"{name_without_ext}.yaml",
        universe_dir / f"{name_without_ext}.json",
    ]

    for path in direct_paths:
        if path and path.exists():
            return path

    # Fall back to walking directories if not found directly
    for root, _, files in os.walk(universe_dir):
        root_path = Path(root)

        # Try exact match with extension
        if name_with_ext and (root_path / name_with_ext).exists():
            return root_path / name_with_ext

        # Try both yaml and json without extension
        if (root_path / f"{name_without_ext}.yaml").exists():
            return root_path / f"{name_without_ext}.yaml"

        if (root_path / f"{name_without_ext}.json").exists():
            return root_path / f"{name_without_ext}.json"

    # Provide more detailed error message
    raise FileNotFoundError(
        f"Universe '{universe_name}' not found. "
        f"Searched in {universe_dir} and its subdirectories. "
        f"Set ALPHALEDGER_UNIVERSE_DIR environment variable to specify a custom location."
    )


def load_universe(
    universe_name: str, start_year: Optional[int] = None, end_year: Optional[int] = None
):
    """
    Load a universe by name.

    Args:
        universe_name: Name of the universe file
        start_year: Optional start year for filtering SEC filings
        end_year: Optional end year for filtering SEC filings
    """
    path = get_universe_path(universe_name)

    if path.suffix.lower() == ".json":
        return load_from_json(str(path), start_year, end_year)
    elif path.suffix.lower() == ".yaml":
        return load_from_yaml(str(path), start_year, end_year)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
