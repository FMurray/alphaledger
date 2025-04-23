import json
import yaml
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path
import os
from alphaledger.config import settings
import datetime
import polars as pl
import re

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
    ):
        # Preserve the original name to support hierarchical paths like "sectors/cloud_computing"
        self.raw_name = name
        self.name = self._normalize_name(name)
        self.securities: Dict[str, Security] = {}
        self.start_year = settings.start_year
        self.end_year = settings.end_year
        self.filings_df: Optional[pl.DataFrame] = None
        self._fetcher: Optional[EDGARFetcher] = None

    def _normalize_name(self, name: str) -> str:
        """Normalize the universe name into a filesystem‑safe string.

        Replaces slashes with underscores and converts any character that is
        *not* alphanumeric, hyphen or underscore into an underscore as well
        (e.g. spaces → "_", "%" → "_", etc.).
        """
        name = name.replace("/", "_")
        return re.sub(r"[^A-Za-z0-9_-]", "_", name)

    def _get_fetcher(self) -> EDGARFetcher:
        """Initializes or returns the EDGARFetcher instance."""
        if self._fetcher is None:
            self._fetcher = EDGARFetcher()
        return self._fetcher

    def _get_filings_path(self, file_format="delta") -> Path:
        """Determines the standard path for saving/loading filings for this universe."""
        output_dir = settings.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # If the *raw* universe name contains a "/" treat the part before the
        # first slash as a sub‑folder to group related universes (e.g.
        # "sectors/cloud_computing" →  output/sec_filings_sectors/cloud_computing.delta)
        if "/" in self.raw_name:
            dir_prefix, file_part = self.raw_name.split("/", 1)
            dir_path = output_dir / f"sec_filings_{dir_prefix}"
            dir_path.mkdir(parents=True, exist_ok=True)
            return dir_path / f"{self._normalize_name(file_part)}.{file_format}"

        # Fallback: flat naming (previous behaviour)
        return output_dir / f"sec_filings_{self.name}.{file_format}"

    def fetch_or_load_filings(
        self,
        force_fetch: bool = False,
        max_age_days: int = 7,
        file_format="delta",  # Default to delta
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
                        output_path=settings.output_dir,  # Save directly to output_dir
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

    def get_missing_filings(self) -> List[Dict[str, int]]:
        """Return missing (ticker, year) combinations.

        We expect exactly one 10‑K filing per (ticker, filing_year) within the
        universe's defined year range.  After `fetch_or_load_filings` has
        populated ``self.filings_df``, this helper inspects that DataFrame and
        returns a list of dictionaries like ``{"ticker": "AMZN", "filing_year":
        2021}`` for every combination that is absent.
        """
        # Ensure filings metadata is loaded
        if self.filings_df is None:
            logger.info(
                "Filings not yet loaded – attempting to load via fetch_or_load_filings()."
            )
            # Avoid infinite recursion if fetch_or_load_filings fails
            # Just return potentially missing based on current state or tickers/years
            if self.filings_df is None:  # Check again after potential load attempt
                self.fetch_or_load_filings()

        # If still empty or None after attempt, assume everything is missing
        if self.filings_df is None or self.filings_df.is_empty():
            logger.warning(
                "Filings DataFrame is empty or None, reporting all expected combinations as missing."
            )
            missing = [
                {"ticker": t, "filing_year": y}
                for t in self.get_tickers()
                for y in self.get_filing_years()
            ]
            return missing

        # Build a fast‑lookup set of existing (ticker, year) tuples
        # Ensure 'ticker' and 'filing_year' columns exist
        if (
            "ticker" not in self.filings_df.columns
            or "filing_year" not in self.filings_df.columns
        ):
            logger.warning(
                "filings_df is missing 'ticker' or 'filing_year' column – treating every combination as missing."
            )
            existing_pairs = set()
        else:
            try:
                # Filter out potential nulls before creating pairs
                filtered_df = self.filings_df.filter(
                    pl.col("ticker").is_not_null() & pl.col("filing_year").is_not_null()
                )
                existing_pairs = set(
                    zip(
                        filtered_df["ticker"].to_list(),
                        filtered_df["filing_year"].to_list(),
                    )
                )
            except Exception as e:
                logger.error(
                    f"Error creating existing pairs set: {e}. Reporting all as missing.",
                    exc_info=True,
                )
                existing_pairs = set()

        missing: List[Dict[str, int]] = []
        for ticker in self.get_tickers():
            for yr in self.get_filing_years():
                if (ticker, yr) not in existing_pairs:
                    missing.append({"ticker": ticker, "filing_year": yr})
        return missing

    def sync_filings(self, file_format="delta") -> None:
        """
        Ensures the local filings data matches the expected filings for the universe.

        Loads existing data, identifies missing (ticker, year) combinations,
        fetches only the missing filings, and updates the local data file.

        Args:
            file_format: The format of the filings file ('delta', 'parquet', 'csv').
        """
        logger.info(f"Starting filings sync for universe '{self.name}'...")

        # 1. Ensure base data is loaded (using existing logic)
        # Use a low max_age_days to ensure we load relatively fresh data before syncing
        # Force fetch=False allows using existing file even if old, sync will fill gaps.
        self.fetch_or_load_filings(
            force_fetch=False, max_age_days=3650, file_format=file_format
        )

        # Handle case where fetch/load failed entirely and df is still None
        if self.filings_df is None:
            logger.error(
                f"Initial load/fetch failed for universe '{self.name}'. Cannot sync."
            )
            return

        # 2. Identify missing filings
        # get_missing_filings already handles None/empty self.filings_df after the load attempt
        missing_combinations = self.get_missing_filings()

        if not missing_combinations:
            logger.info(
                f"Filings for universe '{self.name}' are already complete. No sync needed."
            )
            return

        logger.info(
            f"Found {len(missing_combinations)} missing filing combinations to fetch for {self.get_tickers()} years {self.get_filing_years()}."
        )
        if len(missing_combinations) < 10:
            logger.info(f"Missing items: {missing_combinations}")

        # 3. Fetch *only* the missing filings
        fetcher = self._get_fetcher()
        try:
            # NOTE: EDGARFetcher needs a new method like fetch_specific_filings
            # This method should accept a list of dicts like {'ticker': 'T', 'filing_year': Y}
            # and return a Polars DataFrame with the fetched data.
            logger.info(
                f"Attempting to fetch {len(missing_combinations)} missing filings..."
            )
            tickers_in_missing = list(
                set(item["ticker"] for item in missing_combinations)
            )

            # Reuse existing CIK mapping logic if possible
            cache_file_path = settings.output_dir / "cache" / "ticker_to_cik.json"
            ticker_to_cik = load_ticker_to_cik_mapping(
                tickers_in_missing, str(cache_file_path)
            )
            # Filter ticker_to_cik for only those needed in missing_combinations
            ciks_to_fetch = {
                t: cik for t, cik in ticker_to_cik.items() if t in tickers_in_missing
            }
            logger.info(
                f"Found {len(ciks_to_fetch)} CIKs for {len(tickers_in_missing)} missing tickers."
            )

            missing_filings_df = fetcher.fetch_specific_filings(
                missing_combinations, ciks_to_fetch
            )

            if missing_filings_df is None or missing_filings_df.is_empty():
                logger.warning(
                    f"Fetching missing filings returned no data for universe '{self.name}'. Local file will not be updated with new fetches."
                )
                # If fetch returned nothing, no need to proceed with combining/saving
                logger.info(
                    f"Filings sync process finished for universe '{self.name}' (no new data fetched)."
                )
                return

            logger.info(
                f"Successfully fetched {len(missing_filings_df)} missing filings."
            )

            # 4. Combine and save
            # Ensure self.filings_df is a DataFrame before concatenating
            if (
                self.filings_df is None
            ):  # Should not happen due to check above, but safety first
                current_filings_df = pl.DataFrame()
                logger.warning(
                    "self.filings_df was None before combining, starting with empty DataFrame."
                )
            else:
                current_filings_df = self.filings_df

            # Use pl.concat for combining. 'vertical_relaxed' allows combining even if schemas
            # differ slightly (e.g., new columns added), filling missing values with nulls.
            combined_df = pl.concat(
                [current_filings_df, missing_filings_df], how="vertical_relaxed"
            )

            # Drop duplicates based on key identifiers after combining
            # Keep the 'first' entry encountered, assuming initial load is preferred over sync fetch if identical
            key_columns = [
                "ticker",
                "filing_year",
                "accession_number",
            ]  # Adjust if accession_number isn't always present or unique enough
            if all(col in combined_df.columns for col in key_columns):
                combined_df = combined_df.unique(subset=key_columns, keep="first")
            else:
                logger.warning(
                    f"Cannot drop duplicates using keys {key_columns} as one or more are missing. Skipping deduplication."
                )

            self.filings_df = combined_df
            logger.info(f"Combined DataFrame now has {len(self.filings_df)} filings.")

            # 5. Save the updated DataFrame directly using Polars
            filings_path = self._get_filings_path(file_format)
            logger.info(
                f"Saving updated filings dataframe ({len(self.filings_df)} rows) directly to: {filings_path}"
            )
            try:
                filings_path.parent.mkdir(
                    parents=True, exist_ok=True
                )  # Ensure directory exists
                if file_format == "delta":
                    self.filings_df.write_delta(str(filings_path), mode="overwrite")
                elif file_format == "parquet":
                    self.filings_df.write_parquet(str(filings_path))
                elif file_format == "csv":
                    self.filings_df.write_csv(str(filings_path))
                else:
                    logger.error(f"Unsupported file format '{file_format}' for saving.")
                    return  # Don't mark as success if save failed

                logger.info(f"Successfully synced and saved filings to {filings_path}")

            except Exception as e:
                logger.error(
                    f"Error saving updated filings to {filings_path}: {e}",
                    exc_info=True,
                )

        except FileNotFoundError as e:
            # Raised by load_ticker_to_cik_mapping if cache file path is bad
            logger.error(f"File not found during sync setup: {e}", exc_info=True)
            # reraise if needed?
        except Exception as e:
            # Catch potential errors during the hypothetical fetch_specific_filings
            logger.error(
                f"Error during fetching process for missing filings (universe {self.name}): {e}",
                exc_info=True,
            )

        logger.info(f"Filings sync process finished for universe '{self.name}'.")


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

    universe = Universe(name=universe_name)

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
    load_filings: bool = False,
):
    """Loads a Universe object from a config file (YAML preferred, JSON fallback).

    Args:
        universe_name: The name of the universe (without extension).
        start_year: Optional start year for filtering filings (passed to Universe).
        end_year: Optional end year for filtering filings (passed to Universe).
        load_filings: Whether to automatically trigger fetching/loading of filings metadata.
                      Set to False if the calling script (e.g., build_kb) handles this.

    Returns:
        The loaded Universe object.

    Raises:
        FileNotFoundError: If neither YAML nor JSON config file is found.
    """
    universe_path = get_universe_path(universe_name)
    logger.info(f"Loading universe from: {universe_path}")

    if universe_path.suffix == ".yaml":
        return load_from_yaml(str(universe_path), load_filings)
    elif universe_path.suffix == ".json":
        return load_from_json(str(universe_path), load_filings)
    else:
        # This case should ideally not be reached if get_universe_path works correctly
        raise FileNotFoundError(
            f"No valid universe config file found for {universe_name} (checked .yaml and .json)"
        )


if __name__ == "__main__":
    universe = load_universe("sectors/cloud_computing")
    print(universe)
    print(universe.get_missing_filings())
    universe.sync_filings()
    print(universe.get_missing_filings())
