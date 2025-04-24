import json
import yaml
from typing import List, Dict, Optional, Union, Literal, TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path
import os
from alphaledger.config import settings
import datetime
import polars as pl
import re
from pydantic import BaseModel, Field, model_validator

from alphaledger.sec import EDGARFetcher, load_ticker_to_cik_mapping
from alphaledger import get_logger

# Conditional import for type hinting
if TYPE_CHECKING:
    from alphaledger.universe import Universe as UniverseType

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
                end_year = (
                    start_year  # Or raise validation error? Let's adjust for now.
                )

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
            return []  # Or maybe just [resolved_end]? Empty seems safer.

        return list(range(resolved_start, resolved_end + 1))

    def __str__(self) -> str:
        """String representation of the range."""
        start_str = str(self.start) if self.start is not None else "None"
        end_str = str(self.end) if self.end is not None else "None"
        return f"[{start_str}-{end_str}]"


@dataclass
class Security:
    """Represents a publicly traded security, linked to its parent Universe."""

    ticker: str
    name: str
    exchange: str
    universe: "UniverseType"  # Reference back to the parent Universe
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

    def get_numeric_facts(self) -> pl.LazyFrame:
        """
        Returns a LazyFrame containing the numeric facts for this security
        from the parent Universe's filings data.

        Raises:
            RuntimeError: If the parent Universe's filings have not been loaded.
        """
        if self.universe.filings_df is None:
            raise RuntimeError(
                f"Filings not loaded for universe '{self.universe.name}'. "
                f"Call Universe.ensure_filings_available() first."
            )

        lf = self.universe.filings_df.lazy()
        filtered_lf = lf.filter(pl.col("ticker") == self.ticker)

        # Select columns with numeric types (integers or floats)
        numeric_cols = [
            col for col in filtered_lf.columns if filtered_lf.schema[col].is_numeric()
        ]
        # Optionally, always include key identifiers even if not numeric
        # key_cols = ['ticker', 'filing_year', 'accession_number'] # Example
        # final_cols = list(set(numeric_cols + key_cols)) # Combine and unique

        numeric_lf = filtered_lf.select(numeric_cols)

        return numeric_lf


class Universe:
    """
    A collection of securities that defines the investment universe.

    Loads its definition (name, securities) eagerly from a YAML or JSON
    file upon instantiation based on the provided name/path. Filing data
    is loaded explicitly via `ensure_filings_available()`.
    """

    def __init__(
        self,
        universe_name_or_path: str,
        start_year: Union[int, Literal["earliest"], None] = None,
        end_year: Union[int, Literal["latest"], None] = None,
    ):
        """
        Initializes the Universe by loading its definition from a file.

        Args:
            universe_name_or_path: The name of the universe (e.g., "my_universe")
                or the full path to the definition file (.yaml or .json).
                If a name is given, it searches in the configured universe directory.
            start_year: Optional start year for the analysis period.
            end_year: Optional end year for the analysis period.

        Raises:
            FileNotFoundError: If the definition file cannot be found.
            ValueError: If the loaded definition is invalid.
        """
        self.raw_name: str = universe_name_or_path  # Store the original identifier
        self.name: str
        self.definition_path: Path
        self.securities: Dict[str, Security] = {}
        self.year_range = YearRange(start=start_year, end=end_year)
        self.filings_df: Optional[pl.DataFrame] = None
        self._fetcher: Optional[EDGARFetcher] = None

        self._load_definition(universe_name_or_path)  # Eagerly load definition

        logger.info(
            f"Initialized Universe '{self.name}' from {self.definition_path} "
            f"with {len(self.securities)} securities and year range: {self.year_range}"
        )

    def _find_definition_file(self, universe_name_or_path: str) -> Path:
        """
        Finds the path to a universe definition file (YAML or JSON).

        Args:
            universe_name_or_path: Name or path provided by the user.

        Returns:
            Path object to the found file.

        Raises:
            FileNotFoundError: If the file cannot be found.
        """
        universe_dir = settings.universe_dir

        # Check if it's an absolute path
        if os.path.isabs(universe_name_or_path):
            path = Path(universe_name_or_path)
            if path.exists() and path.is_file() and path.suffix in [".yaml", ".json"]:
                return path
            raise FileNotFoundError(
                f"Universe definition file not found at absolute path: {path}"
            )

        # If not absolute, treat as name and search in universe_dir
        name = universe_name_or_path
        has_extension = name.endswith(".yaml") or name.endswith(".json")
        name_without_ext = name.split(".")[0] if has_extension else name
        preferred_ext = ".yaml"  # Prefer YAML
        fallback_ext = ".json"

        # Handle potential subdirectories in the name (e.g., "sectors/cloud")
        potential_path_parts = name_without_ext.split("/")
        base_name = potential_path_parts[-1]
        sub_dirs = potential_path_parts[:-1]
        search_dir = universe_dir.joinpath(*sub_dirs)

        # Check for exact name with extension if provided
        if has_extension:
            exact_path = search_dir / name
            if exact_path.exists() and exact_path.is_file():
                return exact_path

        # Check for preferred extension (.yaml)
        yaml_path = search_dir / f"{base_name}{preferred_ext}"
        if yaml_path.exists() and yaml_path.is_file():
            return yaml_path

        # Check for fallback extension (.json)
        json_path = search_dir / f"{base_name}{fallback_ext}"
        if json_path.exists() and json_path.is_file():
            return json_path

        # If not found directly, search recursively (less efficient, keep simple for now)
        # Simple search logic:
        # Check direct path variations first
        # for ext in [preferred_ext, fallback_ext]:
        #     p = universe_dir / f"{name_without_ext}{ext}"
        #     if p.exists() and p.is_file():
        #         return p

        # # If name includes slashes, check that path
        # if '/' in name_without_ext:
        #     p_sub = universe_dir / name_without_ext
        #     for ext in [preferred_ext, fallback_ext]:
        #         p = p_sub.with_suffix(ext)
        #         if p.exists() and p.is_file():
        #             return p

        raise FileNotFoundError(
            f"Universe definition '{universe_name_or_path}' not found. "
            f"Searched for {yaml_path} and {json_path}. "
            f"Set ALPHALEDGER_UNIVERSE_DIR environment variable if needed."
        )

    def _load_definition(self, universe_name_or_path: str):
        """Loads securities from the definition file found by _find_definition_file."""
        self.definition_path = self._find_definition_file(universe_name_or_path)
        logger.info(f"Loading universe definition from: {self.definition_path}")

        try:
            with open(self.definition_path, "r") as f:
                if self.definition_path.suffix == ".yaml":
                    data = yaml.safe_load(f)
                elif self.definition_path.suffix == ".json":
                    data = json.load(f)
                else:
                    # Should not happen if _find_definition_file works
                    raise ValueError(
                        f"Unsupported file type: {self.definition_path.suffix}"
                    )

            if not isinstance(data, dict):
                raise ValueError(
                    f"Invalid format: Root element must be a dictionary in {self.definition_path}"
                )

            # Use name from file if present, otherwise derive from path/input
            # Normalize the final name for filesystem safety
            file_name_attr = data.get("name")
            if file_name_attr:
                self.raw_name = file_name_attr  # Prefer name from file if exists
                self.name = self._normalize_name(file_name_attr)
            else:
                # If no name in file, use the input name/path stem, keep raw_name as input
                self.name = self._normalize_name(self.definition_path.stem)
                # Keep self.raw_name as the originally passed identifier

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
                        security = Security(
                            ticker=sec_data["ticker"],
                            name=sec_data["name"],
                            exchange=sec_data["exchange"],
                            universe=self,  # Link back to this universe instance
                            sector=sec_data.get("sector"),
                            industry=sec_data.get("industry"),
                            currency=sec_data.get("currency", "USD"),
                            country=sec_data.get("country", "US"),
                            # Add other optional fields here if needed
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

        except (
            FileNotFoundError
        ):  # Already handled by _find_definition_file, but double check
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
        # If name came from file path like "sectors/cloud", normalize that
        name = name.replace("/", "_")
        # Convert any character that is *not* alphanumeric, hyphen or underscore into an underscore
        return re.sub(r"[^A-Za-z0-9_-]", "_", name)

    def _get_fetcher(self) -> EDGARFetcher:
        """Initializes or returns the EDGARFetcher instance."""
        if self._fetcher is None:
            self._fetcher = EDGARFetcher()
        return self._fetcher

    def _get_filings_path(self, file_format="delta") -> Path:
        """Determines the standard path for saving/loading filings for this universe."""
        output_dir = settings.output_dir / "sec_filings"  # Standard subfolder
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use normalized name for the filename
        filename = f"{self.name}.{file_format}"

        # If the *raw* name had slashes, use them to create subdirectories
        # Check raw_name used *during initialization* for path structure
        # Use self.raw_name which stores the original identifier passed to __init__
        # or the name from the definition file if it contained slashes.
        path_parts = []
        if isinstance(self.raw_name, str) and "/" in self.raw_name:
            # Handle case where name came from file and had slashes: data.get("name", ...)
            if self.definition_path and self.definition_path.name != self.raw_name:
                # Name came from file, raw_name holds that file-defined name
                path_parts = self.raw_name.split("/")[
                    :-1
                ]  # Use directory parts from raw_name
            else:
                # Name came from user input path like "sectors/cloud"
                input_parts = self.raw_name.split("/")
                path_parts = input_parts[:-1]  # Use directory parts from input

        if path_parts:
            subdir = output_dir.joinpath(*path_parts)
            subdir.mkdir(parents=True, exist_ok=True)
            return subdir / filename
        else:
            # Flat structure in sec_filings directory
            return output_dir / filename

    def ensure_filings_available(self, file_format="delta") -> None:
        # TODO: Implement combined load/fetch/sync logic
        raise NotImplementedError("ensure_filings_available is not yet implemented")

    def get_filings(self) -> Optional[pl.DataFrame]:
        """
        Returns the DataFrame of SEC filings associated with this universe,
        if they have been loaded by `ensure_filings_available()`.

        Returns None if filings are not loaded. Does NOT trigger loading.
        """
        if self.filings_df is None:
            logger.debug(
                "Filings DataFrame not loaded. Call ensure_filings_available() first."
            )
        return self.filings_df

    # Keep basic accessors
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
            "(Filings loaded)"
            if self.filings_df is not None
            else "(Filings not loaded)"
        )
        # Use self.name (normalized) for display consistency
        return f"{self.name} {time_range} ({len(self)} securities) {filings_status}"

    def _get_missing_filing_combinations(self) -> List[Dict[str, Union[str, int]]]:
        """
        Identifies (ticker, year) combinations missing from the loaded filings_df.
        Helper for ensure_filings_available. Assumes self.filings_df might be None or empty.
        """
        expected_years = self.get_filing_years()
        tickers = self.get_tickers()

        if not expected_years or not tickers:
            logger.warning(
                f"No tickers or valid filing years for universe '{self.name}'. Cannot determine missing filings."
            )
            return []

        # Build set of expected pairs
        expected_pairs = set((t, y) for t in tickers for y in expected_years)

        # Build set of existing pairs from self.filings_df (if loaded)
        existing_pairs = set()
        if self.filings_df is not None and not self.filings_df.is_empty():
            # Ensure required columns exist
            if (
                "ticker" in self.filings_df.columns
                and "filing_year" in self.filings_df.columns
            ):
                try:
                    # Filter out potential nulls before creating pairs
                    filtered_df = self.filings_df.filter(
                        pl.col("ticker").is_not_null()
                        & pl.col("filing_year").is_not_null()
                    )
                    # Ensure filing_year is int for comparison if needed (might be float/string)
                    # It's safer to compare tuples directly if types match expected set
                    existing_pairs = set(
                        zip(
                            filtered_df["ticker"].to_list(),
                            # Assuming filing_year in DataFrame is compatible with int from get_filing_years()
                            filtered_df["filing_year"].cast(pl.Int64).to_list(),
                        )
                    )
                except Exception as e:
                    logger.error(
                        f"Error creating existing pairs set from filings_df: {e}. Assuming all are missing.",
                        exc_info=True,
                    )
                    # Fallback to assuming nothing exists if error processing df
                    existing_pairs = set()
            else:
                logger.warning(
                    "filings_df is missing 'ticker' or 'filing_year' column. Assuming all expected filings are missing."
                )

        missing_pairs = expected_pairs - existing_pairs

        missing_list = [
            {"ticker": t, "filing_year": y} for t, y in sorted(list(missing_pairs))
        ]

        return missing_list


# --- Removed Functions ---
# load_from_json, load_from_yaml, save_to_json, save_to_yaml, get_universe_path, load_universe
# These are replaced by the new Universe.__init__ logic and potential future save methods on Universe instance.


# --- Save/Load Functions (Consider moving to instance methods or separate utility) ---
def save_universe_definition(
    universe: Universe, filepath: Optional[str] = None, format: str = "yaml"
) -> None:
    """Saves the universe definition (securities) to a file."""
    if filepath is None:
        # Default save path based on universe name/structure? Needs thought.
        # For now, require explicit path or maybe use self.definition_path?
        if hasattr(universe, "definition_path"):
            save_path = universe.definition_path.with_suffix(f".{format}")
        else:
            raise ValueError("Filepath must be provided to save universe definition.")
    else:
        save_path = Path(filepath)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        # Use raw_name if it contains slashes, otherwise normalized name? Be consistent.
        # Let's use raw_name to preserve original structure/intent if possible.
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
            for s in universe.get_all_securities()  # Assumes get_all_securities exists
        ],
    }

    logger.info(f"Saving universe definition for '{universe.name}' to {save_path}")
    with open(save_path, "w") as f:
        if format == "yaml":
            yaml.dump(data, f, sort_keys=False, default_flow_style=False)
        elif format == "json":
            json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported save format: {format}")


# --- Example Usage (Update for new API) ---
if __name__ == "__main__":
    # Setup: Assume 'temp_universe_dir' exists and contains definition files
    # Create dummy files for example if needed
    temp_dir = Path("./temp_universe_dir")
    temp_dir.mkdir(exist_ok=True)
    dummy_yaml = temp_dir / "example_uni.yaml"
    dummy_data = {
        "name": "Example Universe",
        "securities": [
            {"ticker": "EX1", "name": "Example One", "exchange": "NYSE"},
            {
                "ticker": "EX2",
                "name": "Example Two",
                "exchange": "NASDAQ",
                "sector": "Tech",
            },
        ],
    }
    with open(dummy_yaml, "w") as f:
        yaml.dump(dummy_data, f)

    # Monkeypatch settings for the example
    original_universe_dir = settings.universe_dir
    settings.universe_dir = temp_dir
    print(f"Using temporary universe directory: {settings.universe_dir}")

    print("\n--- Initialize Universe (loads definition) ---")
    try:
        # Initialize by name (searches in settings.universe_dir)
        example_universe = Universe("example_uni", start_year=2020, end_year=2022)
        print(example_universe)
        print(f"Tickers loaded: {example_universe.get_tickers()}")
        sec1 = example_universe.get_security("EX1")
        if sec1:
            print(f"Got security: {sec1}")
            # Test get_numeric_facts (will fail until ensure_filings_available is implemented and called)
            try:
                facts = sec1.get_numeric_facts()
                print(f"Numeric facts schema for EX1: {facts.schema}")
                # print(facts.collect()) # Uncomment to see data after filings are loaded
            except RuntimeError as e:
                print(f"Could not get numeric facts yet: {e}")
        else:
            print("Could not find security EX1")

        # Initialize by full path
        # path_universe = Universe(str(dummy_yaml.resolve()))
        # print(path_universe)

        # Test missing filings logic (requires filings_df to be populated or None)
        print("\n--- Missing Filings (Before Load) ---")
        missing = example_universe._get_missing_filing_combinations()
        print(
            f"Missing combinations initially ({len(missing)}): {missing[:5]}..."
        )  # Show first 5

        # TODO: Add example call to ensure_filings_available() when implemented
        # print("\n--- Calling ensure_filings_available() ---")
        # example_universe.ensure_filings_available()
        # print(example_universe) # Status should update
        # print("\n--- Missing Filings (After Load) ---")
        # missing_after = example_universe._get_missing_filing_combinations()
        # print(f"Missing combinations after load ({len(missing_after)}): {missing_after}")
        # facts_after = sec1.get_numeric_facts()
        # print(f"Numeric facts for EX1 after load:\n{facts_after.collect()}")

        # Test saving
        # save_universe_definition(example_universe, format="json") # Saves to example_uni.json

    except FileNotFoundError as e:
        print(f"Error finding universe definition: {e}")
    except ValueError as e:
        print(f"Error loading universe definition: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        # Clean up dummy file/dir and restore settings
        # dummy_yaml.unlink(missing_ok=True)
        # temp_dir.rmdir()
        settings.universe_dir = original_universe_dir
        print(f"\nRestored universe directory: {settings.universe_dir}")

# Keep _get_missing_filings definition at the end for now
# ... (rest of the file, including _get_missing_filing_combinations if it was moved)
