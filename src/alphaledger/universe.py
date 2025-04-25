import json
import yaml
from typing import List, Dict, Optional, Union, Literal, TYPE_CHECKING, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import os
from alphaledger.config import settings
import datetime
import polars as pl
import re
from pydantic import BaseModel, Field, model_validator
from unittest.mock import MagicMock

from alphaledger.sec import EDGARFetcher, load_ticker_to_cik_mapping
from alphaledger import get_logger
from alphaledger.process_xbrl import (
    process_filings_structured,
    TARGET_SCHEMA_NUMERIC_DIRECT_POLARS,
    TARGET_SCHEMA_NUMERIC_POLARS,
)

# Conditional import for type hinting
# if TYPE_CHECKING:
#     from alphaledger.universe import Universe as UniverseType # noqa
#     pass

logger = get_logger(__name__)

# --- Constants ---
EARLIEST_YEAR_PLACEHOLDER = 1995  # Placeholder for "earliest" in range calculations
DEFAULT_YEAR_RANGE = 10  # Default lookback if start_year is None


class YearRange(BaseModel):
    """Defines and validates a range of years for analysis."""

    start: Union[int, Literal["earliest"], None] = Field(
        default=None,
        description=f"Start year (int), 'earliest', or None (defaults to {DEFAULT_YEAR_RANGE} years ago).",
    )
    end: Union[int, Literal["latest"], None] = Field(
        default=None,
        description="End year (int), 'latest', or None (defaults to datetime.datetime.now().year).",
    )

    @model_validator(mode="after")
    def set_defaults_and_validate(self) -> "YearRange":
        """Set default years and validate the range."""
        current_year = datetime.datetime.now().year
        start_year = self.start
        end_year = self.end

        # Set defaults if None
        if start_year is None:
            start_year = current_year - DEFAULT_YEAR_RANGE
            logger.debug(f"Defaulting start_year to {start_year}")
        if end_year is None:
            end_year = current_year
            logger.debug(f"Defaulting end_year to {end_year}")

        if isinstance(start_year, int) and isinstance(end_year, int):
            if start_year > end_year:
                logger.warning(
                    f"Start year {start_year} is after end year {end_year}. Adjusting end year to match start year."
                )
                end_year = start_year

        # Assign potentially modified values back
        self.start = start_year
        self.end = end_year
        return self

    def get_filing_years(self) -> List[int]:
        """Get the concrete list of integer years based on the range."""
        current_year = datetime.datetime.now().year

        if self.start == "earliest":
            resolved_start = EARLIEST_YEAR_PLACEHOLDER
        elif isinstance(self.start, int):
            resolved_start = self.start
        else:  # Should have been defaulted to int
            logger.error(
                f"Unexpected start year type: {self.start}, defaulting to {current_year - DEFAULT_YEAR_RANGE}"
            )
            resolved_start = current_year - DEFAULT_YEAR_RANGE

        # Resolve end year
        if self.end == "latest":
            resolved_end = current_year
        elif isinstance(self.end, int):
            resolved_end = self.end
        else:  # Should have been defaulted to int
            logger.error(
                f"Unexpected end year type: {self.end}, defaulting to {current_year}"
            )
            resolved_end = current_year

        if resolved_start > resolved_end:
            logger.warning(
                f"Resolved start year {resolved_start} is after resolved end year {resolved_end}. Returning empty list."
            )
            return []

        return list(range(resolved_start, resolved_end + 1))

    def __str__(self) -> str:
        """String representation of the range."""
        start_str = str(self.start) if self.start is not None else "None"
        end_str = str(self.end) if self.end is not None else "None"
        return f"[{start_str}-{end_str}]"


@dataclass
class Security:
    """Represents a publicly traded security."""

    ticker: str
    name: str
    exchange: str
    # universe: "UniverseType"  # Reference back removed
    sector: Optional[str] = None
    industry: Optional[str] = None
    currency: str = "USD"
    country: str = "US"
    custom_sector: Optional[str] = None
    custom_industry: Optional[str] = None
    subsector: Optional[str] = None
    theme: Optional[List[str]] = None

    def __str__(self) -> str:
        return f"{self.ticker} - {self.name} ({self.exchange})"


class Universe:
    """
    A collection of securities that defines the investment universe.

    Loads its definition (name, securities) eagerly from a YAML or JSON
    file upon instantiation. Initializes a LazyFrame (`filings_lf`)
    pointing to the expected filings *metadata* file if it exists.

    Use `collect_filings()` to create or update the filings *metadata* file.
    Use `get_filings_lazy()` to access the filings *metadata* lazily.
    Numeric/text facts require separate processing & storage.
    """

    DEFAULT_FILE_FORMAT = "delta"

    def __init__(
        self,
        universe_name_or_path: str,
        start_year: Union[int, Literal["earliest"], None] = None,
        end_year: Union[int, Literal["latest"], None] = None,
        file_format: str = DEFAULT_FILE_FORMAT,
    ):
        """
        Initializes the Universe by loading its definition from a file and
        setting up a lazy reference to its filings *metadata* file if it exists.

        Args:
            universe_name_or_path: The name (e.g., "my_universe", "sectors/cloud")
                or full path to the definition file (.yaml or .json).
            start_year: Optional start year for the analysis period.
            end_year: Optional end year for the analysis period.
            file_format: The expected format of the filings *metadata* file ('delta', 'parquet', 'csv').

        Raises:
            FileNotFoundError: If the definition file cannot be found.
            ValueError: If the loaded definition is invalid.
        """
        self.raw_name: str = universe_name_or_path
        self.name: str
        self.definition_path: Path
        self.securities: Dict[str, Security] = {}
        self.year_range = YearRange(start=start_year, end=end_year)
        self.filings_lf: Optional[pl.LazyFrame] = (
            None  # Will hold LazyFrame if metadata file exists
        )
        self._fetcher: Optional[EDGARFetcher] = None
        self.file_format = file_format  # Store the format

        # Eagerly load definition (populates self.name, self.securities, etc.)
        self._load_definition(universe_name_or_path)

        # Determine filings metadata path based on loaded/normalized name
        filings_path = self._get_filings_path(self.file_format)

        # Initialize LazyFrame if the filings metadata file already exists
        if filings_path.exists():
            logger.info(
                f"Found existing filings metadata file: {filings_path}. Initializing LazyFrame."
            )
            try:
                self.filings_lf = self._scan_filings_file(
                    filings_path, self.file_format
                )
                logger.debug(
                    f"Successfully initialized filings_lf for {self.name}. Schema: {self.filings_lf.schema}"
                )
            except Exception as e:
                logger.error(
                    f"Error scanning existing filings metadata file {filings_path}: {e}. Treating as non-existent.",
                    exc_info=True,
                )
                self.filings_lf = None  # Set to None if scan fails
        else:
            logger.info(
                f"Filings metadata file not found at expected path: {filings_path}. Run collect_filings() to create it."
            )
            self.filings_lf = None

        logger.info(
            f"Initialized Universe '{self.name}' from {self.definition_path} "
            f"with {len(self.securities)} securities. Filings metadata status: {'Detected' if self.filings_lf is not None else 'Not Found'}."
        )

    def _scan_filings_file(self, path: Path, file_format: str) -> pl.LazyFrame:
        """Scans the filings metadata file based on format, returning a LazyFrame."""
        if file_format == "delta":
            return pl.scan_delta(str(path))
        elif file_format == "parquet":
            return pl.scan_parquet(str(path))
        elif file_format == "csv":
            return pl.scan_csv(str(path), infer_schema_length=1000)
        else:
            raise ValueError(f"Unsupported file format '{file_format}' for scanning.")

    def _find_definition_file(self, universe_name_or_path: str) -> Path:
        """Finds the path to a universe definition file (YAML preferred, JSON fallback)."""
        universe_dir = settings.universe_dir

        if os.path.isabs(universe_name_or_path):
            path = Path(universe_name_or_path)
            if path.exists() and path.is_file() and path.suffix in [".yaml", ".json"]:
                return path
            raise FileNotFoundError(
                f"Universe definition file not found at absolute path: {path}"
            )

        name = universe_name_or_path
        potential_rel_path = Path(name)
        preferred_ext = ".yaml"
        fallback_ext = ".json"

        if potential_rel_path.suffix in [preferred_ext, fallback_ext]:
            abs_path = (universe_dir / potential_rel_path).resolve()
            if abs_path.exists() and abs_path.is_file():
                return abs_path

        path_without_ext = potential_rel_path.with_suffix("")
        yaml_path = (
            (universe_dir / path_without_ext).with_suffix(preferred_ext).resolve()
        )
        if yaml_path.exists() and yaml_path.is_file():
            return yaml_path

        json_path = (
            (universe_dir / path_without_ext).with_suffix(fallback_ext).resolve()
        )
        if json_path.exists() and json_path.is_file():
            return json_path

        raise FileNotFoundError(
            f"Universe definition '{universe_name_or_path}' not found. "
            f"Searched relative to '{universe_dir}' for '{name}.yaml' and '{name}.json'. "
            f"Set ALPHALEDGER_UNIVERSE_DIR environment variable if needed."
        )

    def _load_definition(self, universe_name_or_path: str):
        """Loads securities from the definition file."""
        self.definition_path = self._find_definition_file(universe_name_or_path)
        logger.info(f"Loading universe definition from: {self.definition_path}")

        try:
            with open(self.definition_path, "r") as f:
                if self.definition_path.suffix == ".yaml":
                    data = yaml.safe_load(f)
                elif self.definition_path.suffix == ".json":
                    data = json.load(f)
                else:
                    raise ValueError(
                        f"Unsupported file type: {self.definition_path.suffix}"
                    )

            if not isinstance(data, dict):
                raise ValueError(
                    f"Invalid format: Root element must be a dictionary in {self.definition_path}"
                )

            file_name_attr = data.get("name")
            if file_name_attr:
                self.raw_name = file_name_attr
                self.name = self._normalize_name(file_name_attr)
            else:
                self.name = self._normalize_name(self.definition_path.stem)

            securities_data = data.get("securities")
            if securities_data is None:
                logger.warning(
                    f"No 'securities' key found in {self.definition_path}. Universe will be empty."
                )
                self.securities = {}
            elif not isinstance(securities_data, list):
                raise ValueError(
                    f"Invalid format: 'securities' must be a list in {self.definition_path}"
                )
            else:
                self.securities = {}
                for i, sec_data in enumerate(securities_data):
                    if not isinstance(sec_data, dict):
                        logger.warning(
                            f"Skipping invalid security entry at index {i} (not a dict) in {self.definition_path}"
                        )
                        continue
                    try:
                        # Create Security without universe link initially
                        security = Security(
                            ticker=sec_data["ticker"],
                            name=sec_data["name"],
                            exchange=sec_data["exchange"],
                            # universe=self, # Removed link
                            sector=sec_data.get("sector"),
                            industry=sec_data.get("industry"),
                            currency=sec_data.get("currency", "USD"),
                            country=sec_data.get("country", "US"),
                            custom_sector=sec_data.get("custom_sector"),
                            custom_industry=sec_data.get("custom_industry"),
                            subsector=sec_data.get("subsector"),
                            theme=sec_data.get("theme"),
                        )
                        if security.ticker in self.securities:
                            logger.warning(
                                f"Duplicate ticker '{security.ticker}' found in {self.definition_path}. Overwriting previous entry."
                            )
                        self.securities[security.ticker] = security
                    except KeyError as e:
                        logger.warning(
                            f"Skipping security entry at index {i} due to missing key: {e} in {sec_data}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Skipping security entry at index {i} due to unexpected error: {e} in {sec_data}",
                            exc_info=True,
                        )

        except FileNotFoundError:
            raise
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ValueError(
                f"Error parsing definition file {self.definition_path}: {e}"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Unexpected error loading definition from {self.definition_path}: {e}"
            ) from e

    def _normalize_name(self, name: str) -> str:
        """Normalize the universe name into a filesystem‑safe string."""
        name = name.replace("/", "_")
        return re.sub(r"[^A-Za-z0-9_-]", "_", name)

    def _get_fetcher(self) -> EDGARFetcher:
        """Initializes or returns the EDGARFetcher instance."""
        if self._fetcher is None:
            self._fetcher = EDGARFetcher()
        return self._fetcher

    def _get_filings_path(self, file_format: Optional[str] = None) -> Path:
        """Determines the standard path for saving/loading filings *metadata* for this universe."""
        fmt = file_format if file_format is not None else self.file_format
        output_dir = settings.output_dir / "sec_filings"
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{self.name}.{fmt}"

        path_parts = []
        if isinstance(self.raw_name, str) and "/" in self.raw_name:
            path_parts = self.raw_name.split("/")[:-1]

        if path_parts:
            subdir = output_dir.joinpath(*path_parts)
            subdir.mkdir(parents=True, exist_ok=True)
            return subdir / filename
        else:
            return output_dir / filename

    def _identify_missing_filing_combinations(
        self, existing_lf: Optional[pl.LazyFrame]
    ) -> List[Dict[str, Union[str, int]]]:
        """
        Identifies (ticker, year) combinations missing from the filings *metadata*.
        Compares against the universe's defined tickers and year range.
        """
        expected_years = self.get_filing_years()
        tickers = self.get_tickers()

        if not expected_years or not tickers:
            logger.warning(
                f"No tickers or valid filing years for universe '{self.name}'. Cannot determine missing filings."
            )
            return []

        expected_pairs: Set[Tuple[str, int]] = set(
            (t, y) for t in tickers for y in expected_years
        )

        existing_pairs: Set[Tuple[str, int]] = set()
        if existing_lf is not None:
            if "ticker" in existing_lf.schema and "filing_year" in existing_lf.schema:
                try:
                    logger.debug(
                        f"Collecting ticker and filing_year from existing metadata LazyFrame for '{self.name}' to check for missing..."
                    )
                    existing_data = existing_lf.select(
                        ["ticker", "filing_year"]
                    ).collect()
                    logger.debug(
                        f"Collected {len(existing_data)} metadata rows for checking."
                    )

                    filtered_df = existing_data.filter(
                        pl.col("ticker").is_not_null()
                        & pl.col("filing_year").is_not_null()
                    )
                    existing_pairs = set(
                        zip(
                            filtered_df["ticker"].to_list(),
                            filtered_df["filing_year"]
                            .cast(pl.Int64, strict=False)
                            .to_list(),
                        )
                    )
                except Exception as e:
                    logger.error(
                        f"Error collecting or processing existing pairs from metadata LazyFrame for '{self.name}': {e}. Assuming all are missing for check.",
                        exc_info=True,
                    )
                    existing_pairs = set()
            else:
                logger.warning(
                    f"Existing filings metadata LazyFrame for '{self.name}' is missing 'ticker' or 'filing_year' column. Assuming all expected filings are missing."
                )
        else:
            logger.debug(
                f"No existing filings metadata LazyFrame for '{self.name}'. Assuming all expected filings are missing."
            )

        missing_pairs = expected_pairs - existing_pairs
        missing_list = [
            {"ticker": t, "filing_year": y} for t, y in sorted(list(missing_pairs))
        ]
        return missing_list

    def collect_filings(self, force_check: bool = False) -> pl.LazyFrame:
        """
        Ensures the local filings *metadata* file (e.g., Delta table) is created and complete.

        Checks for missing (ticker, year) combinations based on the current `filings_lf`.
        Fetches *metadata* for only the missing filings from SEC EDGAR, combines the data,
        and saves/overwrites the complete metadata set back to the local file.
        Updates `self.filings_lf` to point to the latest version of the metadata file.

        Args:
            force_check: If True, always check for missing filings metadata even if `self.filings_lf`
                         already points to an existing file.
        """
        logger.info(
            f"Ensuring filings metadata file is available and complete for '{self.name}' ({self.year_range})..."
        )
        filings_path = self._get_filings_path(self.file_format)
        fetcher = self._get_fetcher()

        run_check = True
        if self.filings_lf is not None and not force_check:
            logger.info(
                f"Filings metadata file already detected for '{self.name}' at {filings_path}. Use force_check=True to re-verify completeness."
            )
            run_check = False

        missing_combinations = []
        if run_check:
            missing_combinations = self._identify_missing_filing_combinations(
                self.filings_lf
            )

        if not missing_combinations:
            if run_check:
                logger.info(
                    f"Filings metadata for '{self.name}' at {filings_path} is already complete."
                )
            if self.filings_lf is None and filings_path.exists():
                logger.info(
                    f"Initializing LazyFrame for already complete metadata file at {filings_path}."
                )
                try:
                    self.filings_lf = self._scan_filings_file(
                        filings_path, self.file_format
                    )
                    return self.filings_lf
                except Exception as e:
                    logger.error(
                        f"Failed to scan presumably complete metadata file {filings_path}: {e}",
                        exc_info=True,
                    )
            return  # Exit if complete

        logger.info(
            f"Found {len(missing_combinations)} missing filing metadata combinations to fetch for {self.get_tickers()} years {self.get_filing_years()}."
        )
        if len(missing_combinations) < 20:
            logger.info(f"Missing items: {missing_combinations}")

        try:
            logger.info(
                f"Attempting to fetch metadata for {len(missing_combinations)} missing filings..."
            )
            tickers_in_missing = list(
                set(item["ticker"] for item in missing_combinations)
            )
            cache_file_path = settings.output_dir / "cache" / "ticker_to_cik.json"
            ticker_to_cik = load_ticker_to_cik_mapping(
                tickers_in_missing, str(cache_file_path)
            )
            ciks_to_fetch = {
                t: cik for t, cik in ticker_to_cik.items() if t in tickers_in_missing
            }
            logger.info(
                f"Using {len(ciks_to_fetch)} CIKs for {len(tickers_in_missing)} tickers."
            )

            # Ensure the fetcher can fetch metadata for specific combinations
            # Assuming fetch_specific_filings returns a DataFrame with metadata columns
            if not hasattr(fetcher, "fetch_specific_filings"):
                logger.error(
                    "EDGARFetcher missing 'fetch_specific_filings' method. Cannot fetch missing metadata."
                )
                return

            # This method MUST return metadata, not parsed facts
            missing_metadata_df = fetcher.fetch_specific_filings(
                missing_combinations, ciks_to_fetch
            )

            if missing_metadata_df is None or missing_metadata_df.is_empty():
                logger.warning(
                    f"Fetching missing filing metadata returned no data for '{self.name}'. Local metadata file will not be updated."
                )
                return

            logger.info(
                f"Successfully fetched metadata for {len(missing_metadata_df)} missing filings."
            )

            existing_df = None
            if self.filings_lf is not None:
                logger.info(
                    f"Collecting existing metadata from LazyFrame for '{self.name}' to combine with fetched data..."
                )
                try:
                    existing_df = self.filings_lf.collect()
                    logger.info(f"Collected {len(existing_df)} existing metadata rows.")
                except Exception as e:
                    logger.error(
                        f"Failed to collect existing metadata from LazyFrame for '{self.name}': {e}. Proceeding with fetched data only.",
                        exc_info=True,
                    )
                    existing_df = pl.DataFrame()
            else:
                existing_df = pl.DataFrame()

            logger.info(
                f"Combining {len(existing_df)} existing metadata rows with {len(missing_metadata_df)} newly fetched metadata rows."
            )
            combined_df = pl.concat(
                [existing_df, missing_metadata_df], how="vertical_relaxed"
            )

            key_columns = ["ticker", "filing_year", "accession_number"]
            if all(col in combined_df.columns for col in key_columns):
                initial_count = len(combined_df)
                combined_df = combined_df.unique(
                    subset=key_columns, keep="first", maintain_order=True
                )
                dedup_count = initial_count - len(combined_df)
                if dedup_count > 0:
                    logger.info(f"Removed {dedup_count} duplicate metadata entries.")
            else:
                logger.warning(
                    f"Cannot drop duplicates using keys {key_columns}. Skipping deduplication."
                )

            logger.info(
                f"Combined metadata DataFrame ready with {len(combined_df)} entries."
            )

            self._save_filings_df(combined_df, filings_path, self.file_format)

            logger.info(
                f"Updating LazyFrame reference to point to updated metadata file: {filings_path}"
            )
            try:
                self.filings_lf = self._scan_filings_file(
                    filings_path, self.file_format
                )
                logger.debug(
                    f"Successfully updated filings_lf for '{self.name}'. New schema: {self.filings_lf.schema}"
                )
            except Exception as e:
                logger.error(
                    f"Error scanning newly saved metadata file {filings_path}: {e}. LazyFrame set to None.",
                    exc_info=True,
                )
                self.filings_lf = None

        except FileNotFoundError as e:
            logger.error(f"File not found during CIK mapping: {e}", exc_info=True)
        except Exception as e:
            logger.error(
                f"Error during fetching/combining metadata process for '{self.name}': {e}",
                exc_info=True,
            )

        status = (
            f"LazyFrame updated ({len(combined_df)} metadata rows)"
            if "combined_df" in locals()
            else "LazyFrame status unchanged due to error"
        )
        logger.info(
            f"Filings metadata availability check/update finished for '{self.name}'. {status}"
        )

    def _save_filings_df(self, df: pl.DataFrame, path: Path, file_format: str):
        """Helper to save the filings *metadata* dataframe."""
        logger.info(
            f"Saving updated filings metadata dataframe ({len(df)} rows) to: {path} (format: {file_format})"
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if file_format == "delta":
                if path.is_file():
                    path.unlink()
                df.write_delta(str(path), mode="overwrite")
            elif file_format == "parquet":
                if path.is_dir():
                    logger.warning(
                        f"Target path {path} is a directory. Parquet write might behave unexpectedly. Consider removing dir first."
                    )
                df.write_parquet(str(path))
            elif file_format == "csv":
                df.write_csv(str(path))
            else:
                logger.error(f"Unsupported file format '{file_format}' for saving.")
                return

            logger.info(f"Successfully saved filings metadata to {path}")

        except Exception as e:
            logger.error(
                f"Error saving updated filings metadata to {path}: {e}", exc_info=True
            )

    def get_filings_lazy(self) -> Optional[pl.LazyFrame]:
        """
        Returns a LazyFrame pointing to the SEC filings *metadata* file for this universe,
        if the file exists and was successfully scanned during initialization or
        after `collect_filings()`.

        Returns None if the file does not exist or could not be scanned.
        Does NOT trigger loading data into memory or fetching from network.
        """
        if self.filings_lf is None:
            logger.debug(
                f"Filings metadata LazyFrame not available for '{self.name}'. File may be missing or failed initial scan."
            )
        return self.filings_lf

    # --- Basic Accessors ---
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
        """Get the range of years to consider for SEC filings using the YearRange model."""
        return self.year_range.get_filing_years()

    def __str__(self) -> str:
        time_range = str(self.year_range)
        filings_status = (
            "(Filings metadata detected)"
            if self.filings_lf is not None
            else "(Filings metadata file not found)"
        )
        return f"{self.name} {time_range} ({len(self)} securities) {filings_status}"

    def _get_numeric_facts_path(self) -> Path:
        """Determines the standard path for caching numeric facts for the entire universe."""
        # Use delta format for numeric facts cache, regardless of metadata format.
        fmt = "delta"
        output_dir = settings.output_dir / "numeric_facts"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.name}.{fmt}"  # Filename is the normalized universe name
        return output_dir / filename

    def get_numeric_facts(self) -> Optional[pl.LazyFrame]:
        """
        Lazily retrieves numeric facts for the entire universe.

        Checks a local cache path first (e.g., output/numeric_facts/universe_name.delta).
        If facts are not cached, it ensures filings metadata is available,
        processes all XBRL filings for the universe, saves the combined results
        to the cache, and then returns a LazyFrame pointing to the cache.

        Returns:
            A Polars LazyFrame pointing to the universe's numeric facts data.
            Returns None if the essential filings metadata (`filings_lf`) cannot be
            obtained (e.g., after running collect_filings).
            Returns an empty LazyFrame if no filings are found or processing fails.
        """
        numeric_facts_path = self._get_numeric_facts_path()

        # 1. Check cache
        if numeric_facts_path.exists() and numeric_facts_path.is_dir():
            logger.info(
                f"Cache hit: Found existing numeric facts for universe '{self.name}' at {numeric_facts_path}. Returning LazyFrame."
            )
            try:
                lf = pl.scan_delta(str(numeric_facts_path))
                # Simple check: does it have expected columns?
                # Use the DIRECT schema for checking now (no section_name)
                expected_cols = set(TARGET_SCHEMA_NUMERIC_DIRECT_POLARS.keys())
                expected_cols.add("ticker")  # Ticker should be present
                expected_cols.update(["filing_date", "report_date"])
                if expected_cols.issubset(lf.columns):
                    return lf
                else:
                    logger.warning(
                        f"Cached file {numeric_facts_path} has unexpected schema ({lf.columns}). Expected {expected_cols}. Will regenerate."
                    )
            except Exception as e:
                logger.error(
                    f"Error scanning cached delta table for universe '{self.name}' at {numeric_facts_path}: {e}. Will attempt to regenerate.",
                    exc_info=True,
                )
                # If scanning fails or schema mismatch, proceed to regenerate
        else:
            logger.info(
                f"Cache miss: No cached numeric facts found for universe '{self.name}' at {numeric_facts_path}. Processing required."
            )

        # 2. Ensure filings metadata is available (essential for processing)
        # Use get_filings_lazy which returns None if not available
        filings_lf = self.get_filings_lazy()
        if filings_lf is None:
            logger.warning(
                f"Filings metadata LazyFrame not available for '{self.name}'. "
                f"Run collect_filings() first. Cannot process numeric facts."
            )
            # Cannot proceed without metadata
            return None

        logger.info(f"Processing numeric facts for universe '{self.name}'...")

        # 3. Collect *all* necessary filing metadata for the universe
        try:
            # Select only needed columns for efficiency, including dates
            required_cols = [
                "ticker",
                "xbrl_instance_url",
                "filing_date",
                "report_date",
            ]
            if not all(col in filings_lf.columns for col in required_cols):
                logger.error(
                    f"Filings metadata is missing required columns for fact processing. Needed: {required_cols}. Available: {filings_lf.columns}"
                )
                # Return None because we cannot proceed correctly without dates
                return None

            universe_filings_df = filings_lf.select(required_cols).collect()
        except Exception as e:
            logger.error(
                f"Error collecting filings metadata for universe '{self.name}': {e}",
                exc_info=True,
            )
            # Define schema for empty LF, including ticker and dates (use DIRECT schema)
            empty_schema = {
                **TARGET_SCHEMA_NUMERIC_DIRECT_POLARS,
                "ticker": pl.Utf8,
                "filing_date": pl.Date,
                "report_date": pl.Date,
            }
            return pl.LazyFrame(schema=empty_schema)

        if universe_filings_df.is_empty():
            logger.warning(
                f"No filings metadata found for universe '{self.name}' within the specified year range {self.year_range}."
            )
            # Create an empty DataFrame with full direct schema and save it to the cache path
            empty_schema = {
                **TARGET_SCHEMA_NUMERIC_DIRECT_POLARS,
                "ticker": pl.Utf8,
                "filing_date": pl.Date,
                "report_date": pl.Date,
            }
            empty_df = pl.DataFrame(schema=empty_schema)
            try:
                empty_df.write_delta(str(numeric_facts_path), mode="overwrite")
                logger.info(
                    f"Saved empty placeholder for universe '{self.name}' facts at {numeric_facts_path}"
                )
                return empty_df.lazy()
            except Exception as e:
                logger.error(
                    f"Failed to write empty placeholder delta table for universe '{self.name}' at {numeric_facts_path}: {e}",
                    exc_info=True,
                )
                return empty_df.lazy()  # Return empty lazy frame anyway

        logger.info(
            f"Found {len(universe_filings_df)} filings for universe '{self.name}'. Processing XBRL..."
        )

        # 4. Process all URLs for the universe using the direct method
        try:
            numeric_facts_df, _ = process_filings_structured(universe_filings_df)
            logger.info(
                f"Successfully processed facts for universe '{self.name}'. Found {len(numeric_facts_df)} total numeric facts."
            )

            # 5. Save the combined DataFrame to cache
            if not numeric_facts_df.is_empty():
                # Ensure ticker and date columns exist before saving
                expected_cols = set(TARGET_SCHEMA_NUMERIC_DIRECT_POLARS.keys()).union(
                    {"ticker", "filing_date", "report_date"}
                )
                if not expected_cols.issubset(numeric_facts_df.columns):
                    logger.error(
                        f"Processed facts DataFrame is missing expected columns. Expected: {expected_cols}. Got: {numeric_facts_df.columns}. Cannot save correctly."
                    )
                    # Return an empty LF as the result is incomplete.
                    empty_schema = {
                        **TARGET_SCHEMA_NUMERIC_DIRECT_POLARS,
                        "ticker": pl.Utf8,
                        "filing_date": pl.Date,
                        "report_date": pl.Date,
                    }
                    return pl.LazyFrame(schema=empty_schema)

                try:
                    numeric_facts_df.write_delta(
                        str(numeric_facts_path), mode="overwrite"
                    )
                    logger.info(
                        f"Saved numeric facts for universe '{self.name}' to cache: {numeric_facts_path}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to write numeric facts delta table for universe '{self.name}' to {numeric_facts_path}: {e}",
                        exc_info=True,
                    )
                    # Proceed to return the lazy frame even if saving failed
            else:
                logger.warning(
                    f"Processing for universe '{self.name}' yielded no numeric facts. Saving empty placeholder."
                )
                # Save empty placeholder if processing resulted in nothing
                empty_schema = {
                    **TARGET_SCHEMA_NUMERIC_DIRECT_POLARS,
                    "ticker": pl.Utf8,
                    "filing_date": pl.Date,
                    "report_date": pl.Date,
                }
                empty_df = pl.DataFrame(schema=empty_schema)
                try:
                    empty_df.write_delta(str(numeric_facts_path), mode="overwrite")
                except Exception as e:
                    logger.error(
                        f"Failed to write empty placeholder delta after processing for universe '{self.name}': {e}",
                        exc_info=True,
                    )

            # 6. Return LazyFrame pointing to the (newly created) cache
            try:
                return pl.scan_delta(str(numeric_facts_path))
            except Exception as e:
                logger.error(
                    f"Failed to scan newly created/updated delta table for '{self.name}' at {numeric_facts_path}: {e}. Returning in-memory LF.",
                    exc_info=True,
                )
                # Fallback: return the lazy version of the processed DF if scan fails
                # Ensure expected columns are present in fallback
                expected_cols = set(TARGET_SCHEMA_NUMERIC_DIRECT_POLARS.keys()).union(
                    {"ticker", "filing_date", "report_date"}
                )
                if not expected_cols.issubset(numeric_facts_df.columns):
                    empty_schema = {
                        **TARGET_SCHEMA_NUMERIC_DIRECT_POLARS,
                        "ticker": pl.Utf8,
                        "filing_date": pl.Date,
                        "report_date": pl.Date,
                    }
                    return pl.LazyFrame(schema=empty_schema)
                else:
                    return numeric_facts_df.lazy()

        except Exception as e:
            logger.error(
                f"Error during XBRL processing pipeline for universe '{self.name}': {e}",
                exc_info=True,
            )
            # Return empty LF if XBRL processing itself failed
            empty_schema = {
                **TARGET_SCHEMA_NUMERIC_DIRECT_POLARS,
                "ticker": pl.Utf8,
                "filing_date": pl.Date,
                "report_date": pl.Date,
            }
            return pl.LazyFrame(schema=empty_schema)

    # --- Standalone Load/Save for Definition Files (Optional) ---

    def get_security_numeric_facts(self, ticker: str) -> Optional[pl.LazyFrame]:
        """
        Lazily retrieves numeric facts for a specific security.

        Checks a local cache path first. If facts are not cached,
        it fetches the required filing metadata, processes the XBRL,
        saves the results to the cache, and then returns a LazyFrame.

        Args:
            ticker: The ticker symbol of the security.

        Returns:
            A Polars LazyFrame pointing to the numeric facts data, or None if the
            security is not found or initial metadata is missing/unavailable.
            Returns an empty LazyFrame if no filings are found or processing fails.
        """
        if ticker not in self.securities:
            logger.error(f"Ticker '{ticker}' not found in universe '{self.name}'.")
            return None

        numeric_facts_path = self._get_numeric_facts_path()

        # 1. Check cache
        if (
            numeric_facts_path.exists() and numeric_facts_path.is_dir()
        ):  # Delta tables are dirs
            logger.info(
                f"Cache hit: Found existing numeric facts for {ticker} at {numeric_facts_path}. Returning LazyFrame."
            )
            try:
                return pl.scan_delta(str(numeric_facts_path))
            except Exception as e:
                logger.error(
                    f"Error scanning cached delta table for {ticker} at {numeric_facts_path}: {e}. Will attempt to regenerate.",
                    exc_info=True,
                )
                # If scanning fails, proceed to regenerate
        else:
            logger.info(
                f"Cache miss: No cached numeric facts found for {ticker} at {numeric_facts_path}. Processing required."
            )

        # 2. Check if filings metadata is available (needed for processing)
        if self.filings_lf is None:
            logger.warning(
                f"Filings metadata LazyFrame not available for '{self.name}'. "
                f"Run collect_filings() first. Cannot fetch facts for {ticker}."
            )
            # Cannot proceed without metadata
            return None  # Return None as we cannot even attempt processing

        logger.info(f"Fetching numeric facts for {ticker} in universe '{self.name}'...")

        # 3. Filter metadata for the specific ticker
        try:
            ticker_filings_lf = self.filings_lf.filter(pl.col("ticker") == ticker)
            ticker_filings_df = ticker_filings_lf.collect()
        except Exception as e:
            logger.error(
                f"Error collecting filings metadata for ticker '{ticker}': {e}",
                exc_info=True,
            )
            # Return empty LF as processing failed before XBRL step
            return pl.LazyFrame(schema=TARGET_SCHEMA_NUMERIC_POLARS)

        if ticker_filings_df.is_empty():
            logger.warning(
                f"No filings metadata found for ticker '{ticker}' within the specified year range {self.year_range} for universe '{self.name}'."
            )
            # Create an empty DataFrame and save it to the cache path
            empty_df = pl.DataFrame(schema=TARGET_SCHEMA_NUMERIC_POLARS)
            try:
                empty_df.write_delta(str(numeric_facts_path), mode="overwrite")
                logger.info(
                    f"Saved empty placeholder for {ticker} facts at {numeric_facts_path}"
                )
                return empty_df.lazy()
            except Exception as e:
                logger.error(
                    f"Failed to write empty placeholder delta table for {ticker} at {numeric_facts_path}: {e}",
                    exc_info=True,
                )
                return empty_df.lazy()  # Return empty lazy frame anyway

        logger.info(
            f"Found {len(ticker_filings_df)} filings for {ticker}. Processing XBRL..."
        )

        # 4. Process the URLs using the function from process_xbrl
        try:
            numeric_facts_df, _ = process_filings_structured(ticker_filings_df)
            logger.info(
                f"Successfully processed facts for {ticker}. Found {len(numeric_facts_df)} numeric facts."
            )

            # 5. Save to cache
            if not numeric_facts_df.is_empty():
                try:
                    numeric_facts_df.write_delta(
                        str(numeric_facts_path), mode="overwrite"
                    )
                    logger.info(
                        f"Saved numeric facts for {ticker} to cache: {numeric_facts_path}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to write numeric facts delta table for {ticker} to {numeric_facts_path}: {e}",
                        exc_info=True,
                    )
                    # Proceed to return the lazy frame even if saving failed
            else:
                logger.warning(
                    f"Processing for {ticker} yielded no numeric facts. Saving empty placeholder."
                )
                # Save empty placeholder if processing resulted in nothing
                empty_df = pl.DataFrame(schema=TARGET_SCHEMA_NUMERIC_POLARS)
                try:
                    empty_df.write_delta(str(numeric_facts_path), mode="overwrite")
                except Exception as e:
                    logger.error(
                        f"Failed to write empty placeholder delta after processing for {ticker}: {e}",
                        exc_info=True,
                    )

            # 6. Return LazyFrame pointing to the (newly created) cache
            # Use scan_delta again to ensure we load exactly what was written (or the empty placeholder)
            try:
                return pl.scan_delta(str(numeric_facts_path))
            except Exception as e:
                logger.error(
                    f"Failed to scan newly created/updated delta table for {ticker} at {numeric_facts_path}: {e}. Returning in-memory LF.",
                    exc_info=True,
                )
                # Fallback: return the lazy version of the processed DF if scan fails
                return numeric_facts_df.lazy()

        except Exception as e:
            logger.error(
                f"Error processing XBRL filings for ticker '{ticker}': {e}",
                exc_info=True,
            )
            # Return empty LF if XBRL processing itself failed
            return pl.LazyFrame(schema=TARGET_SCHEMA_NUMERIC_POLARS)


# --- Standalone Load/Save for Definition Files (Optional) ---


def save_universe_definition(
    universe: "Universe", filepath: Optional[str] = None, format: str = "yaml"
) -> None:
    """Saves the universe definition (securities) to a file."""
    if filepath is None:
        if hasattr(universe, "definition_path") and universe.definition_path:
            original_suffix = universe.definition_path.suffix
            save_format = format if format else original_suffix.lstrip(".")
            save_path = universe.definition_path.with_suffix(f".{save_format}")
        else:
            raise ValueError(
                "Filepath must be provided to save universe definition if not loaded from a file."
            )
    else:
        save_path = Path(filepath)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "name": universe.raw_name,
        "securities": [
            {
                "ticker": s.ticker,
                "name": s.name,
                "exchange": s.exchange,
                "sector": s.sector,
                "industry": s.industry,
                "currency": s.currency,
                "country": s.country,
                "custom_sector": s.custom_sector,
                "custom_industry": s.custom_industry,
                "subsector": s.subsector,
                "theme": s.theme,
            }
            for s in sorted(universe.get_all_securities(), key=lambda x: x.ticker)
        ],
    }

    logger.info(f"Saving universe definition for '{universe.name}' to {save_path}")
    with open(save_path, "w") as f:
        if save_path.suffix == ".yaml":
            yaml.dump(data, f, sort_keys=False, default_flow_style=False)
        elif save_path.suffix == ".json":
            json.dump(data, f, indent=2)
        else:
            raise ValueError(
                f"Unsupported save format based on filepath extension: {save_path.suffix}"
            )


# --- Example Usage ---
if __name__ == "__main__":
    # Setup: Create dummy files for example
    temp_dir = Path("./temp_universe_dir_example_lazy")
    temp_dir.mkdir(parents=True, exist_ok=True)
    example_uni_path = temp_dir / "example_uni.yaml"
    sub_dir = temp_dir / "sectors"
    sub_dir.mkdir(parents=True, exist_ok=True)
    cloud_uni_path = sub_dir / "cloud.yaml"

    example_data = {
        "name": "Example Universe",
        "securities": [
            {
                "ticker": "EX1",
                "name": "Example One",
                "exchange": "NYSE",
                "sector": "Industrials",
            },
            {
                "ticker": "EX2",
                "name": "Example Two",
                "exchange": "NASDAQ",
                "sector": "Tech",
            },
        ],
    }
    cloud_data = {
        "securities": [
            {"ticker": "MSFT", "name": "Microsoft", "exchange": "NASDAQ"},
            {"ticker": "AMZN", "name": "Amazon", "exchange": "NASDAQ"},
        ]
    }

    with open(example_uni_path, "w") as f:
        yaml.dump(example_data, f)
    with open(cloud_uni_path, "w") as f:
        yaml.dump(cloud_data, f)

    # --- Mock Settings ---
    original_universe_dir = settings.universe_dir
    original_output_dir = settings.output_dir
    settings.universe_dir = temp_dir
    settings.output_dir = temp_dir / "output"
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"--- Using temporary universe directory: {settings.universe_dir} ---")
    print(f"--- Using temporary output directory: {settings.output_dir} ---")

    try:
        print(
            "\n--- 1. Initialize Universe (Definition loaded, Filings *Metadata* LazyFrame status checked) ---"
        )
        example_universe = Universe(
            "example_uni", start_year=2021, end_year="latest", file_format="delta"
        )
        print(example_universe)
        assert example_universe.filings_lf is None

        print("\n--- 2. Try getting *metadata* before file exists ---")
        meta_lf = example_universe.get_filings_lazy()
        print(f"get_filings_lazy() returns: {meta_lf}")
        assert meta_lf is None

        # --- 3. Run collect_filings (MOCK - Creates the *metadata* file) ---
        print("\n--- 3. Run collect_filings (Mocked Fetch - Creates Metadata File) ---")
        try:
            mock_fetcher = MagicMock(spec=EDGARFetcher)

            def mock_fetch_metadata(combinations, ciks):
                print(
                    f" -> MOCK fetcher called for metadata of {len(combinations)} combinations."
                )
                if not combinations:
                    return pl.DataFrame()
                # Simulate fetching metadata (URLs, dates, etc.)
                return pl.DataFrame(
                    [
                        {
                            "ticker": combo["ticker"],
                            "filing_year": combo["filing_year"],
                            "accession_number": f"FAKE_{combo['ticker']}_{combo['filing_year']}",
                            "report_date": datetime.date(
                                combo["filing_year"] + 1, 3, 1
                            ),
                            "filing_date": datetime.date(
                                combo["filing_year"] + 1, 2, 15
                            ),
                            "xbrl_instance_url": f"http://fake.sec.gov/{combo['ticker']}_{combo['filing_year']}.xml",
                        }
                        for combo in combinations
                    ]
                )

            mock_fetcher.fetch_specific_filings.side_effect = (
                mock_fetch_metadata  # Now mocking metadata fetch
            )
            example_universe._fetcher = mock_fetcher

            example_universe.collect_filings()
            print(example_universe)
            assert (
                example_universe.filings_lf is not None
            )  # LazyFrame for metadata should exist

        except ImportError:
            print("Skipping ensure_filings test: Mocking library not available.")
        except Exception as e:
            print(
                f"An error occurred during collect_filings mock run: {e}",
                exc_info=True,
            )

        # --- 4. Try getting *metadata* AFTER file should exist ---
        print("\n--- 4. Try getting *metadata* AFTER file should exist ---")
        if example_universe.filings_lf is not None:
            try:
                meta_lf_after = example_universe.get_filings_lazy()
                print(f"Metadata LazyFrame schema: {meta_lf_after.schema}")
                print(f"Collecting metadata for {example_universe.name}...")
                meta_df = meta_lf_after.collect()
                print(f"Collected metadata ({len(meta_df)} rows):")
                print(meta_df)
                assert not meta_df.is_empty()
                assert "xbrl_instance_url" in meta_df.columns

            except Exception as e:
                print(
                    f"Error getting/collecting metadata after ensure_filings: {e}",
                    exc_info=True,
                )
        else:
            print(
                "Cannot test getting metadata: filings_lf is still None after collect_filings."
            )

        # --- 5. Initialize new instance - should detect existing *metadata* file ---
        print(
            "\n--- 5. Initialize NEW Universe instance (should detect existing metadata file) ---"
        )
        example_universe_2 = Universe(
            "example_uni", start_year=2021, end_year="latest", file_format="delta"
        )
        print(example_universe_2)
        assert example_universe_2.filings_lf is not None

        # --- NOTE: Accessing numeric facts is now a separate step ---
        print(
            "\n--- Accessing processed numeric facts (Requires separate processing step) ---"
        )
        # 1. Get metadata (e.g., URLs) from universe.get_filings_lazy().collect()
        # 2. Pass URLs to a function like process_filing_urls from process_xbrl.py
        # 3. That function parses XBRL and saves numeric facts (e.g., to numeric_facts.delta)
        # 4. Load and query the separate numeric_facts.delta file:
        #    numeric_facts = pl.scan_delta("path/to/numeric_facts.delta")
        #    ibm_facts = numeric_facts.filter(pl.col('ticker') == 'IBM').collect()

    except FileNotFoundError as e:
        print(f"Error finding universe definition: {e}")
    except ValueError as e:
        print(f"Error loading universe definition: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        # --- Restore Original Settings & Clean up ---
        settings.universe_dir = original_universe_dir
        settings.output_dir = original_output_dir
        print(f"\n--- Restored universe directory: {settings.universe_dir} ---")
        print(f"--- Restored output directory: {settings.output_dir} ---")
        try:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Cleaned up temporary directory: {temp_dir}")
        except ImportError:
            print(f"Could not import shutil to clean up {temp_dir}")
