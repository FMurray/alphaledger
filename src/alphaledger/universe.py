import json
import yaml
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path
import os
from alphaledger.config import settings
import datetime
import polars as pl

from alphaledger.sec import EDGARFetcher, load_ticker_to_cik_mapping
from alphaledger import get_logger

logger = get_logger(__name__)


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
        self.filings_df: Optional[pl.DataFrame] = None
        self._fetcher: Optional[EDGARFetcher] = None

    def _get_fetcher(self) -> EDGARFetcher:
        """Initializes or returns the EDGARFetcher instance."""
        if self._fetcher is None:
            self._fetcher = EDGARFetcher()
        return self._fetcher

    def _get_filings_path(self, file_format="parquet") -> Path:
        """Determines the standard path for saving/loading filings for this universe."""
        filings_dir = settings.output_dir / "filings" / self.name
        filings_dir.mkdir(parents=True, exist_ok=True)
        return filings_dir / f"sec_filings.{file_format}"

    def fetch_or_load_filings(
        self,
        force_fetch: bool = False,
        max_age_days: int = 7,
        file_format="parquet",
    ) -> None:
        """
        Fetches SEC 10-K filings for the universe or loads them from disk if recent.

        Args:
            force_fetch: If True, always fetch from SEC, ignoring cached file.
            max_age_days: Max age in days for the cached file to be considered recent.
            file_format: Format to use for saving/loading ('parquet' or 'csv').
        """
        filings_path = self._get_filings_path(file_format)
        load_from_disk = False

        if not force_fetch and filings_path.exists():
            file_mod_time = datetime.datetime.fromtimestamp(
                filings_path.stat().st_mtime
            )
            age = datetime.datetime.now() - file_mod_time
            if age.days < max_age_days:
                logger.info(
                    f"Found recent filings file ({age.days} days old) at: {filings_path}"
                )
                load_from_disk = True
            else:
                logger.info(
                    f"Filings file at {filings_path} is older than {max_age_days} days. Will re-fetch."
                )
        else:
            logger.info(
                f"No recent filings file found at {filings_path} or force_fetch=True. Fetching from SEC."
            )

        fetcher = self._get_fetcher()

        if load_from_disk:
            try:
                logger.info(f"Loading filings from {filings_path}...")
                loaded_df = fetcher.load_filings_from_disk(str(filings_path))
                if not loaded_df.is_empty():
                    self.filings_df = loaded_df
                    logger.info(
                        f"Successfully loaded {len(self.filings_df)} filings from disk."
                    )
                else:
                    logger.warning(f"Loaded empty DataFrame from {filings_path}.")
                    self.filings_df = pl.DataFrame()
            except Exception as e:
                logger.error(
                    f"Error loading filings from {filings_path}: {e}. Will attempt fetch.",
                    exc_info=True,
                )
                load_from_disk = False

        if not load_from_disk:
            logger.info("Fetching filings from SEC EDGAR...")
            tickers = self.get_tickers()
            if not tickers:
                logger.warning(
                    f"Universe '{self.name}' has no tickers. Cannot fetch filings."
                )
                self.filings_df = pl.DataFrame()
                return

            try:
                cache_file_path = settings.output_dir / "cache" / "ticker_to_cik.json"
                ticker_to_cik = load_ticker_to_cik_mapping(
                    tickers, str(cache_file_path)
                )
                logger.info(f"Using {len(ticker_to_cik)} CIKs for fetching.")

                fetched_df = fetcher.fetch_filings_for_universe(self, ticker_to_cik)

                if not fetched_df.is_empty():
                    logger.info(
                        f"Successfully fetched {len(fetched_df)} filings from SEC."
                    )
                    saved_path = fetcher.save_filings_to_disk(
                        fetched_df,
                        output_path=filings_path.parent,
                        file_format=file_format,
                        universe_name=self.name,
                    )
                    if saved_path:
                        logger.info(f"Saved fetched filings to: {saved_path}")
                    self.filings_df = fetched_df
                else:
                    logger.warning("Fetching from SEC returned no filings.")
                    self.filings_df = pl.DataFrame()

            except Exception as e:
                logger.error(
                    f"Error fetching filings for universe {self.name}: {e}",
                    exc_info=True,
                )
                self.filings_df = None

    def get_filings(self) -> Optional[pl.DataFrame]:
        """
        Returns the DataFrame of SEC filings associated with this universe.

        Ensures filings are loaded/fetched if they haven't been already.
        """
        if self.filings_df is None:
            logger.info("Filings not yet loaded. Calling fetch_or_load_filings().")
            self.fetch_or_load_filings()
        return self.filings_df

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

        if start > end:
            logger.warning(
                f"Universe start year ({start}) is after end year ({end}). Adjusting range."
            )
            return [end] if end is not None else [current_year]

        return list(range(start, end + 1))

    def __str__(self) -> str:
        time_range = ""
        if self.start_year or self.end_year:
            start = self.start_year if self.start_year is not None else "earliest"
            end = self.end_year if self.end_year is not None else "latest"
            time_range = f" [{start}-{end}]"
        filings_status = (
            "(Filings loaded)"
            if self.filings_df is not None
            else "(Filings not loaded)"
        )
        return f"{self.name}{time_range} ({len(self)} securities) {filings_status}"


def load_from_json(
    filepath: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    load_filings: bool = True,
) -> Universe:
    """
    Load a universe of securities from a JSON file.

    Args:
        filepath: Path to the JSON file
        start_year: Optional start year for filtering SEC filings
        end_year: Optional end year for filtering SEC filings
        load_filings: If True, automatically fetch or load associated SEC filings.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(path, "r") as f:
        data = json.load(f)

    default_name = path.stem
    universe_name = data.get("name", default_name)

    universe = Universe(
        name=universe_name,
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

    if load_filings:
        logger.info(
            f"Automatically triggering filings load for universe '{universe.name}'..."
        )
        universe.fetch_or_load_filings()

    return universe


def load_from_yaml(
    filepath: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    load_filings: bool = True,
) -> Universe:
    """
    Load a universe of securities from a YAML file.

    Args:
        filepath: Path to the YAML file
        start_year: Optional start year for filtering SEC filings
        end_year: Optional end year for filtering SEC filings
        load_filings: If True, automatically fetch or load associated SEC filings.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    default_name = path.stem
    universe_name = data.get("name", default_name)

    universe = Universe(
        name=universe_name,
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

    if load_filings:
        logger.info(
            f"Automatically triggering filings load for universe '{universe.name}'..."
        )
        universe.fetch_or_load_filings()

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
    universe_dir = settings.universe_dir

    if os.path.isabs(universe_name):
        path = Path(universe_name)
        if path.exists():
            return path
        raise FileNotFoundError(f"Universe file not found at: {path}")

    name_with_ext = (
        universe_name
        if (universe_name.endswith(".yaml") or universe_name.endswith(".json"))
        else None
    )
    name_without_ext = universe_name.split(".")[0] if name_with_ext else universe_name

    direct_paths = [
        universe_dir / name_with_ext if name_with_ext else None,
        universe_dir / f"{name_without_ext}.yaml",
        universe_dir / f"{name_without_ext}.json",
    ]

    for path in direct_paths:
        if path and path.exists():
            return path

    for root, _, files in os.walk(universe_dir):
        root_path = Path(root)

        if name_with_ext and (root_path / name_with_ext).exists():
            return root_path / name_with_ext

        if (root_path / f"{name_without_ext}.yaml").exists():
            return root_path / f"{name_without_ext}.yaml"

        if (root_path / f"{name_without_ext}.json").exists():
            return root_path / f"{name_without_ext}.json"

    raise FileNotFoundError(
        f"Universe '{universe_name}' not found. "
        f"Searched in {universe_dir} and its subdirectories. "
        f"Set ALPHALEDGER_UNIVERSE_DIR environment variable to specify a custom location."
    )


def load_universe(
    universe_name: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    load_filings: bool = True,
):
    """
    Load a universe by name, optionally loading associated SEC filings.

    Args:
        universe_name: Name or path of the universe file
        start_year: Optional start year for filtering SEC filings
        end_year: Optional end year for filtering SEC filings
        load_filings: If True, automatically fetch or load associated SEC filings.
    """
    path = get_universe_path(universe_name)

    if path.suffix.lower() == ".json":
        return load_from_json(str(path), start_year, end_year, load_filings)
    elif path.suffix.lower() == ".yaml":
        return load_from_yaml(str(path), start_year, end_year, load_filings)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
