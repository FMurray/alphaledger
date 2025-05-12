import requests
import time
import os
import json
import polars as pl
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Union, TYPE_CHECKING, Tuple
from datetime import datetime
from alphaledger.config import settings
from alphaledger import get_logger
from pathlib import Path
from dataclasses import dataclass
from datetime import date
import deltalake

# --- Add TYPE_CHECKING block for Universe import ---
if TYPE_CHECKING:
    from alphaledger.universe import Universe

logger = get_logger(__name__)


@dataclass
class CoreFilingMetadata:
    ticker: str
    filing_date: date
    filing_type: str
    filing_id: str
    cik: str
    accession_number: str


def download_ticker_to_cik_mapping(output_path: Optional[str] = None) -> Dict[str, str]:
    """
    Download the ticker to CIK mapping from SEC's website.

    Args:
        output_path: Optional path to save the mapping to disk

    Returns:
        Dictionary mapping tickers to CIK numbers
    """
    url = "https://www.sec.gov/include/ticker.txt"
    headers = {"User-Agent": settings.sec_user_agent}  # Use settings for user agent

    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise exception for non-200 status codes

    mapping = {}
    for line in response.text.strip().split("\n"):
        parts = line.strip().split("\t")
        if len(parts) == 2:
            ticker_val, cik_val = parts
            mapping[ticker_val.upper()] = cik_val.zfill(10)  # Ensure CIK is 10 digits

    # Save mapping to file if requested
    if output_path:
        # Ensure output_path is a Path object before using parent
        output_path_obj = Path(output_path)
        os.makedirs(output_path_obj.parent, exist_ok=True)
        with open(output_path_obj, "w") as f:
            json.dump(mapping, f, indent=2)

    return mapping


def load_ticker_to_cik_mapping(
    tickers: List[str], cache_file: str = "ticker_to_cik.json"
) -> Dict[str, str]:
    """
    Load CIK numbers for the given tickers.

    Args:
        tickers: List of ticker symbols
        cache_file: Path to cache the mapping

    Returns:
        Dictionary mapping tickers to CIK numbers
    """
    # If cache exists and is recent (less than 30 days old), use it
    use_cache = False
    if os.path.exists(cache_file):
        file_age = datetime.now().timestamp() - os.path.getmtime(cache_file)
        if file_age < 30 * 24 * 60 * 60:  # 30 days in seconds
            use_cache = True

    # Load the full mapping
    if use_cache:
        try:
            with open(cache_file, "r") as f:
                full_mapping = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Cache file {cache_file} is corrupted. Re-downloading.")
            full_mapping = download_ticker_to_cik_mapping(
                cache_file
            )  # Re-download if corrupt
    else:
        full_mapping = download_ticker_to_cik_mapping(cache_file)

    # Filter to just the tickers we need
    result = {}
    missing_tickers = []
    for ticker_sym in tickers:
        upper_ticker = ticker_sym.upper()
        if upper_ticker in full_mapping:
            result[ticker_sym] = full_mapping[upper_ticker]
        else:
            missing_tickers.append(ticker_sym)

    if missing_tickers:
        logger.warning(
            f"Could not find CIKs for the following tickers: {', '.join(missing_tickers)}"
        )

    return result


# Define a basic schema for the raw combined data from SEC JSON
RAW_FILING_SCHEMA = {
    "accessionNumber": pl.Utf8,
    "filingDate": pl.Utf8,
    "reportDate": pl.Utf8,
    "form": pl.Utf8,
    "primaryDocument": pl.Utf8,
    "primaryDocDescription": pl.Utf8,
    # SEC includes other fields like 'act', 'fileNumber', 'filmNumber', 'items',
    # 'size', 'isXBRL', 'isInlineXBRL', 'acceptanceDateTime'.
    # These will be included if present due to strict=False when creating DataFrame.
}


class EDGARFetcher:
    def __init__(self, user_agent=None, cache_dir=settings.output_dir / "cache"):
        """
        Initialize the EDGAR fetcher with your contact information.

        Args:
            user_agent (str, optional): Your name, email, and organization as required by SEC
                                       If None, uses the value from settings
            cache_dir (str, optional): Directory to store the XBRL cache
        """
        # Use provided user_agent or fall back to settings
        self.headers = {"User-Agent": user_agent or settings.sec_user_agent}
        self.base_url = settings.sec_base_url
        # Respect SEC rate limits (10 requests per second)
        self.request_interval = settings.sec_request_interval
        # Set up cache directory
        self.cache_dir = Path(
            cache_dir or settings.output_dir / "cache"
        )  # Ensure Path object

        # Initialize HTTP cache for XBRL parsing
        from xbrl.cache import HttpCache

        # Convert Path object to string to avoid the endswith() error
        # HttpCache might expect a string path
        cache_dir_str = str(self.cache_dir)
        self.http_cache = HttpCache(cache_dir_str)
        self.http_cache.set_headers({"User-Agent": self.headers["User-Agent"]})

        # Cache for raw combined DataFrames per CIK (cik_formatted -> pl.DataFrame)
        self._cik_raw_data_store: Dict[str, pl.DataFrame] = {}

        # Revised Cache: Store final filtered results per (CIK, year/range)
        self._search_results_cache: Dict[
            Tuple[str, Optional[int], Optional[int], Optional[int]], pl.DataFrame
        ] = {}

    def _get_json_for_cik_and_url(
        self, cik_formatted: str, custom_url_part: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Low-level helper to fetch and parse a specific JSON file from SEC for a CIK.
        Uses HttpCache for network requests.
        cik_formatted: 10-digit CIK string.
        custom_url_part: e.g., 'CIKxxxxxxxxxx-submissions-001.json' or None for main CIKxxx.json.
        """
        if custom_url_part:
            # Ensure custom_url_part is just the filename, not a full path
            url_filename = Path(custom_url_part).name
            # Construct URL for historical archive files
            # Example: https://data.sec.gov/submissions/CIK0000320193-submissions-001.json
            url = f"https://data.sec.gov/submissions/{url_filename}"

        else:
            # Construct URL for the main CIK index file
            # Example: https://data.sec.gov/submissions/CIK0000320193.json
            url = f"https://data.sec.gov/submissions/CIK{cik_formatted}.json"

        logger.debug(f"Fetching/caching JSON from: {url}")
        time.sleep(self.request_interval)
        try:
            local_file_path_str = self.http_cache.cache_file(url)
            if not local_file_path_str:  # cache_file can return None on error
                logger.error(
                    f"HttpCache failed to retrieve or cache file for URL: {url}"
                )
                return None
            with open(local_file_path_str, "rb") as f:
                response_bytes = f.read()
            response_str = response_bytes.decode("utf-8")
            data = json.loads(response_str)
            logger.debug(
                f"Successfully parsed JSON from {url} (via cache: {local_file_path_str})"
            )
            return data
        except Exception as e:
            logger.error(f"Error fetching/parsing JSON from {url}: {e}", exc_info=True)
            return None

    # Helper function to extract filing dicts from potentially columnar JSON
    def _extract_filings_from_json_data(
        self, filings_data_json: Optional[Dict], cik_formatted: str
    ) -> List[Dict]:
        """Safely extracts a list of filing dictionaries from SEC JSON structures."""
        filings_list = []
        if not isinstance(filings_data_json, dict):
            logger.debug(
                f"Expected dict for filings_data_json for CIK {cik_formatted}, got {type(filings_data_json)}. Skipping."
            )
            return filings_list

        # Data might be directly under keys in filings_data_json (common for historical archives)
        # or nested under filings_data_json['filings']['recent'] (common for main CIK index)

        # Try to access data assuming it might be columnar directly or in a known nested structure
        data_to_process = None
        if "accessionNumber" in filings_data_json and isinstance(
            filings_data_json["accessionNumber"], list
        ):
            data_to_process = filings_data_json  # Assume current level is columnar
        elif "filings" in filings_data_json and isinstance(
            filings_data_json["filings"], dict
        ):
            recent_filings = filings_data_json["filings"].get("recent")
            if (
                isinstance(recent_filings, dict)
                and "accessionNumber" in recent_filings
                and isinstance(recent_filings["accessionNumber"], list)
            ):
                data_to_process = recent_filings  # Process 'recent' section

        if data_to_process:
            try:
                num_filings = len(data_to_process.get("accessionNumber", []))
                keys = list(data_to_process.keys())
                for i in range(num_filings):
                    filing_dict = {}
                    for key in keys:
                        if (
                            key in data_to_process
                            and isinstance(data_to_process[key], list)
                            and i < len(data_to_process[key])
                        ):
                            filing_dict[key] = data_to_process[key][i]
                        else:
                            filing_dict[key] = None  # Ensure all keys are present
                    filings_list.append(filing_dict)
            except Exception as e:
                logger.error(
                    f"Error processing columnar JSON data for CIK {cik_formatted}: {e}",
                    exc_info=True,
                )
        elif isinstance(filings_data_json, list):  # Fallback if root is a list of dicts
            filings_list.extend(filings_data_json)
        else:
            logger.debug(
                f"Could not determine columnar data in filings_data_json for CIK {cik_formatted}. Keys: {list(filings_data_json.keys())}"
            )

        return filings_list

    def _get_all_filings_for_cik_raw(self, cik_formatted: str) -> pl.DataFrame:
        """
        Fetches all filing entries for a CIK (recent and historical)
        and returns them as a single, unfiltered Polars DataFrame.
        Uses _cik_raw_data_store.
        """
        if cik_formatted in self._cik_raw_data_store:
            logger.debug(f"Cache hit for raw combined filings for CIK {cik_formatted}.")
            return self._cik_raw_data_store[cik_formatted]

        logger.debug(f"Fetching raw combined filings for CIK {cik_formatted}.")
        all_filings_list: List[Dict] = []

        # 1. Get the main CIK index file (e.g., CIK0000320193.json)
        main_cik_index_json = self._get_json_for_cik_and_url(cik_formatted, None)

        if not main_cik_index_json:
            logger.warning(f"Could not retrieve main CIK index for {cik_formatted}.")
            self._cik_raw_data_store[cik_formatted] = pl.DataFrame(
                schema=RAW_FILING_SCHEMA
            )
            return self._cik_raw_data_store[cik_formatted]

        # 2. Process "recent" filings from the main CIK index
        if "filings" in main_cik_index_json and isinstance(
            main_cik_index_json["filings"], dict
        ):
            recent_filings_data = main_cik_index_json["filings"].get("recent")
            if recent_filings_data:
                logger.debug(f"Processing 'recent' filings for CIK {cik_formatted}")
                all_filings_list.extend(
                    self._extract_filings_from_json_data(
                        recent_filings_data, cik_formatted
                    )
                )

        # 3. Process "historical" archive files listed in the main CIK index
        if "filings" in main_cik_index_json and isinstance(
            main_cik_index_json["filings"], dict
        ):
            historical_files_info = main_cik_index_json["filings"].get("files", [])
            logger.debug(
                f"Found {len(historical_files_info)} historical archive files for CIK {cik_formatted}."
            )
            for file_info in historical_files_info:
                archive_file_name = file_info.get("name")
                if not archive_file_name:
                    logger.warning(
                        f"Skipping historical archive with no name for CIK {cik_formatted}: {file_info}"
                    )
                    continue

                logger.debug(
                    f"Fetching historical archive: {archive_file_name} for CIK {cik_formatted}"
                )
                historical_archive_json = self._get_json_for_cik_and_url(
                    cik_formatted, custom_url_part=archive_file_name
                )
                if historical_archive_json:
                    all_filings_list.extend(
                        self._extract_filings_from_json_data(
                            historical_archive_json, cik_formatted
                        )
                    )

        # 4. Create DataFrame
        if not all_filings_list:
            logger.info(
                f"No filings found for CIK {cik_formatted} after processing recent and historical."
            )
            raw_df = pl.DataFrame(schema=RAW_FILING_SCHEMA)
        else:
            logger.info(
                f"Combined {len(all_filings_list)} raw filing records for CIK {cik_formatted}."
            )
            try:
                raw_df = pl.DataFrame(
                    all_filings_list, schema_overrides=RAW_FILING_SCHEMA, strict=False
                )
            except Exception as e:
                logger.error(
                    f"Error creating raw DataFrame for CIK {cik_formatted}: {e}",
                    exc_info=True,
                )
                raw_df = pl.DataFrame(schema=RAW_FILING_SCHEMA)

        self._cik_raw_data_store[cik_formatted] = raw_df
        return raw_df

    def get_filings_for_cik_ticker_year(
        self, cik: str, ticker: str, year: int
    ) -> pl.DataFrame:
        """
        Public method to get 10-K filings for a specific CIK, Ticker, and Year.
        """
        cik_formatted = str(cik).zfill(10)

        # Step 1: Get all raw filings for the CIK (uses cache)
        combined_df = self._get_all_filings_for_cik_raw(cik_formatted)

        if combined_df.is_empty():
            logger.debug(
                f"No raw filings data found for CIK {cik_formatted} to filter for year {year}."
            )
            return pl.DataFrame()

        # Step 2: Filter the raw DataFrame
        try:
            # Ensure 'form' column exists
            if "form" not in combined_df.columns:
                logger.warning(
                    f"Raw DataFrame for CIK {cik_formatted} missing 'form' column. Cannot filter 10-Ks."
                )
                return pl.DataFrame()

            filtered_df = combined_df.filter(pl.col("form") == "10-K")
            if filtered_df.is_empty():
                logger.debug(
                    f"No 10-K forms found for CIK {cik_formatted} in raw data."
                )
                return pl.DataFrame()

            # Ensure 'filingDate' column exists for year filtering
            if "filingDate" not in filtered_df.columns:
                logger.warning(
                    f"10-K DataFrame for CIK {cik_formatted} missing 'filingDate' column. Cannot filter by year."
                )
                return pl.DataFrame()

            # Process dates and derive filing_year
            # Handle potential errors during date parsing or year extraction
            date_processed_df = filtered_df.with_columns(
                pl.col("filingDate")
                .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
                .alias("filing_date_dt_temp")
            ).with_columns(
                pl.col("filing_date_dt_temp")
                .dt.year()
                .cast(pl.Int64, strict=False)
                .alias("filing_year_temp")
            )

            # Filter by the requested year
            year_filtered_df = date_processed_df.filter(
                pl.col("filing_year_temp") == year
            )

            if year_filtered_df.is_empty():
                logger.debug(
                    f"No 10-K filings found for CIK {cik_formatted}, Ticker {ticker} for year {year}."
                )
                return pl.DataFrame()

            # Add other derived columns
            final_df = year_filtered_df.with_columns(
                [
                    pl.lit(cik_formatted).alias("cik"),
                    pl.lit(ticker).alias("ticker"),
                    pl.col("filingDate")
                    .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
                    .alias("filing_date_dt"),  # Re-add for final schema
                    pl.col("reportDate")
                    .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
                    .alias("report_date_dt"),
                    pl.col("filing_year_temp").alias(
                        "filing_year"
                    ),  # Rename temp year column
                    pl.lit(datetime.now()).alias("processed_datetime"),
                ]
            )

            # Corrected CIK formatting for URL path
            final_df = final_df.with_columns(
                (
                    pl.lit(self.base_url + "edgar/data/")
                    + pl.col("cik").cast(pl.Int64).cast(pl.Utf8)
                    + "/"  # Corrected: remove leading zeros for path
                    + pl.col("accessionNumber").str.replace_all("-", "")
                    + "/"
                    + pl.col("accessionNumber")
                    + "-index.html"
                ).alias("documents_url"),
                pl.when(
                    pl.col("primaryDocument").is_not_null()
                    & (pl.col("primaryDocument") != "")
                )
                .then(
                    pl.lit(self.base_url + "edgar/data/")
                    + pl.col("cik").cast(pl.Int64).cast(pl.Utf8)
                    + "/"  # Corrected: remove leading zeros for path
                    + pl.col("accessionNumber").str.replace_all("-", "")
                    + "/"
                    + pl.col("primaryDocument")
                )
                .otherwise(None)
                .alias("xbrl_instance_url"),
            )

            # Select and reorder final columns
            # Must match TARGET_METADATA_SCHEMA from universe.py
            final_columns_ordered = [
                "ticker",
                "cik",
                "form",
                "accessionNumber",
                "filingDate",
                "filing_date_dt",
                "filing_year",
                "reportDate",
                "report_date_dt",
                "primaryDocument",
                "primaryDocDescription",
                "documents_url",
                "xbrl_instance_url",
                "processed_datetime",
            ]

            # Ensure all expected columns are present, add nulls if not
            for col_name in final_columns_ordered:
                if col_name not in final_df.columns:
                    if col_name in ["filing_date_dt", "report_date_dt"]:
                        final_df = final_df.with_columns(
                            pl.lit(None, dtype=pl.Date).alias(col_name)
                        )
                    elif col_name == "filing_year":
                        final_df = final_df.with_columns(
                            pl.lit(None, dtype=pl.Int64).alias(col_name)
                        )
                    else:
                        final_df = final_df.with_columns(
                            pl.lit(None, dtype=pl.Utf8).alias(col_name)
                        )

            return final_df.select(final_columns_ordered)

        except Exception as e:
            logger.error(
                f"Error filtering/processing for CIK {cik_formatted}, Ticker {ticker}, Year {year}: {e}",
                exc_info=True,
            )
            return pl.DataFrame()

    def fetch_specific_filings(
        self,
        filing_combinations: List[Dict[str, Union[str, int]]],
        ticker_to_cik_mapping: Dict[str, str],
    ) -> pl.DataFrame:
        """
        Fetch specific 10-K filings based on a list of (ticker, filing_year) combinations.
        """
        specific_filings_dfs: List[pl.DataFrame] = []
        logger.info(
            f"Starting fetch for {len(filing_combinations)} specific filing combinations."
        )

        self._cik_raw_data_store.clear()  # Clear raw CIK data store at the start of a new batch
        logger.debug("Cleared CIK raw data store.")

        processed_combinations = 0
        for combo in filing_combinations:
            ticker_val = combo.get("ticker")  # Renamed
            year_val = combo.get("filing_year")  # Renamed

            if not isinstance(ticker_val, str) or not isinstance(year_val, int):
                logger.warning(f"Skipping invalid combination: {combo}")
                continue

            if ticker_val in ticker_to_cik_mapping:
                cik_val = ticker_to_cik_mapping[ticker_val]  # Renamed

                logger.debug(
                    f"Processing: Ticker {ticker_val}, CIK {cik_val}, Year {year_val}"
                )

                # Call the new method
                filing_df = self.get_filings_for_cik_ticker_year(
                    cik_val, ticker_val, year_val
                )

                if not filing_df.is_empty():
                    # get_filings_for_cik_ticker_year should ideally return one 10-K or none.
                    # If multiple (e.g. amendments), this logic might take the first if not handled inside.
                    # For now, assume it's handled or take what's given.
                    if len(filing_df) > 1:
                        logger.warning(
                            f"Found {len(filing_df)} filings for {ticker_val} in year {year_val}. Expected one 10-K. Using all."
                        )
                    specific_filings_dfs.append(filing_df)
                    logger.debug(
                        f"Successfully processed data for {ticker_val}, year {year_val}. Rows: {len(filing_df)}"
                    )
                else:
                    logger.info(
                        f"No 10-K filing data found for {ticker_val}, year {year_val}."
                    )  # Changed to info

                processed_combinations += 1
                # Modest delay between CIKs, more frequent for actual fetches inside helpers
                if processed_combinations % 10 == 0:
                    time.sleep(self.request_interval * 0.5)  # Shorter general delay
            else:
                logger.warning(
                    f"Skipping {ticker_val} year {year_val}: CIK not found in provided mapping."
                )

        self._cik_raw_data_store.clear()  # Clear store at the end
        logger.debug("Cleared CIK raw data store after operation.")

        if specific_filings_dfs:
            # Ensure all dataframes have compatible schemas before concat, using TARGET_METADATA_SCHEMA
            # This is crucial if some processing steps fail and DataFrames have different columns
            final_dfs_to_concat = []
            from alphaledger.universe import (
                TARGET_METADATA_SCHEMA,
            )  # Import for schema reference

            expected_cols_ordered = list(TARGET_METADATA_SCHEMA.keys())
            polars_schema = {
                name: dtype for name, dtype in TARGET_METADATA_SCHEMA.items()
            }

            for i, df in enumerate(specific_filings_dfs):
                if df.is_empty():
                    # Create an empty DataFrame with the target schema to ensure concat works
                    # This handles cases where a ticker/year combo yielded no results
                    final_dfs_to_concat.append(pl.DataFrame(schema=polars_schema))
                    continue

                current_cols = df.columns
                select_exprs = []
                for col_name in expected_cols_ordered:
                    if col_name in current_cols:
                        select_exprs.append(pl.col(col_name))
                    else:
                        # Add missing column as null literal with correct type
                        select_exprs.append(
                            pl.lit(None, dtype=TARGET_METADATA_SCHEMA[col_name]).alias(
                                col_name
                            )
                        )
                try:
                    aligned_df = df.select(select_exprs)
                    final_dfs_to_concat.append(aligned_df)
                except Exception as e:
                    logger.error(
                        f"Schema alignment error for DataFrame {i} (Ticker: {df.select('ticker').head(1).item() if 'ticker' in df.columns else 'N/A'}): {e}. Columns: {df.columns}",
                        exc_info=True,
                    )
                    # Append an empty DataFrame with target schema on error
                    final_dfs_to_concat.append(pl.DataFrame(schema=polars_schema))

            if not final_dfs_to_concat:  # If all processing failed or no data
                logger.warning("No valid DataFrames to concatenate after alignment.")
                return pl.DataFrame()

            try:
                combined_df = pl.concat(
                    final_dfs_to_concat, how="diagonal_relaxed"
                )  # Diagonal might be safer if alignment isn't perfect
                logger.info(
                    f"Finished fetching. Combined {len(combined_df)} specific filings successfully."
                )
                return combined_df
            except Exception as e:
                logger.error(
                    f"Error concatenating final DataFrames: {e}", exc_info=True
                )
                # Log details of the DFs that failed to concat
                for i, df_to_log in enumerate(final_dfs_to_concat):
                    logger.debug(
                        f"DataFrame {i} for concat: Schema: {df_to_log.schema}, Shape: {df_to_log.shape}"
                    )
                return pl.DataFrame()  # Return empty on concat error
        else:
            logger.info(
                "Finished fetching specific filings. No new filings were found or processed."
            )
            return pl.DataFrame()

    def save_filings_to_disk(
        self,
        filings_df: pl.DataFrame,  # Updated input type
        output_path: Optional[Path] = None,  # Use Path type hint
        file_format="parquet",
        universe_name="universe",
    ) -> Optional[str]:  # Return type remains str path or None
        """
        Save the filings Polars DataFrame to disk.

        Args:
            filings_df: Polars DataFrame containing the filings data
            output_path: Path where the file should be saved (directory or full file path)
            file_format: Format to save the file (parquet or csv)
            universe_name: Name of the universe (used in filename if output_path is dir)

        Returns:
            Path to the saved file as string, or None if failed/empty
        """
        if filings_df.is_empty():  # Use Polars is_empty()
            logger.warning("No filings data to save.")
            return None

        # Determine output directory and filename
        if output_path is None:
            # Default directory based on universe name if possible
            base_dir = settings.output_dir / "filings" / universe_name
        elif output_path.is_dir():
            base_dir = output_path
        else:  # Assume output_path includes filename
            base_dir = output_path.parent
            base_filename = output_path.stem  # Get filename without extension
            # Override format based on provided path extension if different
            provided_ext = output_path.suffix.lower().lstrip(".")
            if (
                provided_ext in ["parquet", "csv", "delta"]
                and provided_ext != file_format.lower()
            ):
                logger.warning(
                    f"Output path extension '.{provided_ext}' overrides requested format '{file_format}'. Using '.{provided_ext}'."
                )
                file_format = provided_ext

        # Ensure output directory exists
        base_dir.mkdir(parents=True, exist_ok=True)

        # Construct filename if not provided in output_path
        if output_path is None or output_path.is_dir():
            # Generate filename, check if it exists as a directory (less likely now)
            base_filename = f"sec_filings_{universe_name}"
            # This check might be less necessary if using dedicated dirs, but keep for safety
            if (base_dir / base_filename).is_dir():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = f"{base_filename}_{timestamp}"

        # Create full file path with extension
        file_path = base_dir / f"{base_filename}.{file_format.lower()}"

        # Save based on format using Polars writers
        try:
            if file_format.lower() == "parquet":
                filings_df.write_parquet(file_path)
            elif file_format.lower() == "csv":
                filings_df.write_csv(file_path)
            elif file_format.lower() == "delta":
                # Delta write expects a directory path, ensure it's treated as such
                if file_path.suffix == ".delta":  # if user provided path/to/table.delta
                    delta_table_path = str(file_path)
                else:  # if user provided path/to/table (no ext) or path/to/dir
                    delta_table_path = str(file_path)

                # For overwrite, Delta Lake typically requires removing the old table first if schema changes etc.
                # Or use mode="overwrite" with appropriate options.
                # Simple overwrite for now:
                if Path(delta_table_path).exists():
                    import shutil

                    if Path(delta_table_path).is_dir():
                        shutil.rmtree(delta_table_path)
                    else:  # Parquet file was perhaps written here before
                        Path(delta_table_path).unlink()

                filings_df.write_delta(
                    delta_table_path, mode="overwrite"
                )  # Ensure path is string
            else:
                logger.error(f"Unsupported file format for saving: {file_format}")
                return None

            logger.info(f"Saved {len(filings_df)} filings to {file_path}")
            return str(file_path)  # Return path as string

        except Exception as e:
            logger.error(
                f"Error saving filings DataFrame to {file_path}: {e}", exc_info=True
            )
            return None

    def load_filings_from_disk(self, file_path: Union[str, Path]) -> pl.DataFrame:
        file_path_obj = Path(file_path)
        # Ensure TARGET_METADATA_SCHEMA is available for returning empty DFs with schema
        from alphaledger.universe import TARGET_METADATA_SCHEMA

        if not file_path_obj.exists():
            logger.error(f"File not found: {file_path_obj}")
            return pl.DataFrame(schema=TARGET_METADATA_SCHEMA)

        try:
            if file_path_obj.suffix.lower() == ".parquet":
                # For parquet, we assume schema matches if written by this system
                # Consider adding alignment if parquet files can come from other sources
                return pl.read_parquet(file_path_obj)
            elif file_path_obj.suffix.lower() == ".csv":
                # CSVs are less strict, try to parse dates but schema alignment might be needed
                df = pl.read_csv(
                    file_path_obj, infer_schema_length=10000, try_parse_dates=True
                )
                # Basic alignment for CSVs - this could be more robust
                aligned_cols = []
                for col_name, expected_type in TARGET_METADATA_SCHEMA.items():
                    if col_name in df.columns:
                        try:
                            aligned_cols.append(
                                pl.col(col_name).cast(expected_type, strict=False)
                            )
                        except pl.PolarsError:
                            logger.warning(
                                f"CSV load: Could not cast '{col_name}' to {expected_type}. Keeping original or null."
                            )
                            aligned_cols.append(
                                pl.lit(None, dtype=expected_type).alias(col_name)
                            )
                    else:
                        aligned_cols.append(
                            pl.lit(None, dtype=expected_type).alias(col_name)
                        )
                return df.select(aligned_cols)  # Select to ensure order and presence
            elif (
                file_path_obj.is_dir() or file_path_obj.suffix.lower() == ".delta"
            ):  # Delta tables are dirs
                try:
                    df = pl.read_delta(str(file_path_obj))
                    # Ensure schema matches TARGET_METADATA_SCHEMA after load
                    aligned_df_list = []
                    for col_name, expected_dtype in TARGET_METADATA_SCHEMA.items():
                        if col_name in df.columns:
                            if df[col_name].dtype != expected_dtype:
                                try:
                                    aligned_df_list.append(
                                        pl.col(col_name).cast(
                                            expected_dtype, strict=False
                                        )
                                    )
                                except pl.PolarsError:
                                    logger.warning(
                                        f"Delta load: Could not cast column '{col_name}' to {expected_dtype}. Keeping original."
                                    )
                                    aligned_df_list.append(
                                        pl.col(col_name)
                                    )  # keep original if cast fails
                            else:
                                aligned_df_list.append(pl.col(col_name))
                        else:
                            aligned_df_list.append(
                                pl.lit(None, dtype=expected_dtype).alias(col_name)
                            )
                    return df.select(aligned_df_list)
                except Exception as inner_e:
                    logger.error(
                        f"Error reading or aligning Delta table {file_path_obj}: {inner_e}",
                        exc_info=True,
                    )
                    return pl.DataFrame(schema=TARGET_METADATA_SCHEMA)
            else:
                logger.error(f"Unsupported file format or path type: {file_path_obj}")
                return pl.DataFrame(schema=TARGET_METADATA_SCHEMA)
        except Exception as e:
            logger.error(f"Error loading file {file_path_obj}: {e}", exc_info=True)
            return pl.DataFrame(schema=TARGET_METADATA_SCHEMA)

    def get_filing_by_accession_number(
        self, cik: str, accession_number: str, file_type="10-K"
    ):  # Unused 'file_type'
        """
        Get a specific filing by accession number.

        Args:
            cik (str): The CIK number of the company
            accession_number (str): The accession number of the filing
            file_type (str): The type of filing (default: "10-K")

        Returns:
            The text content of the filing
        """
        cik_formatted = str(cik).zfill(10)
        acc_no_clean = accession_number.replace("-", "")
        url = f"{self.base_url}edgar/data/{cik_formatted}/{acc_no_clean}/{accession_number}.txt"
        logger.debug(f"Fetching full submission text from: {url}")
        time.sleep(self.request_interval)
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Error fetching filing text for CIK {cik}, Acc# {accession_number}: {e}"
            )
            return None

    def extract_text_from_filing(self, filing_text: Optional[str]):
        """
        Extract the relevant text content from an SEC filing.
        This is a basic implementation - you may need more sophisticated parsing
        based on your specific needs.

        Args:
            filing_text (str): The raw text of the SEC filing

        Returns:
            The extracted text content (simplified)
        """
        if not filing_text:
            return "No text content provided."
        from bs4 import BeautifulSoup

        documents = filing_text.split("<DOCUMENT>")
        for doc in documents:
            if "<TYPE>10-K" in doc:  # Simple check for 10-K
                start_idx = doc.find("<TEXT>")
                end_idx = doc.find("</TEXT>")
                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    text_content = doc[start_idx + len("<TEXT>") : end_idx]
                    soup = BeautifulSoup(text_content, "html.parser")
                    return soup.get_text(separator="\n", strip=True)
        return "No 10-K text content found in the filing."

    def parse_filing_xbrl(
        self, row: pl.Series
    ):  # row is a Polars Series if df.iter_rows(named=True)
        """
        Parse an XBRL filing using the schema and document URLs.

        Args:
            row: DataFrame row containing 'documents_url' and 'xbrl_root_url'

        Returns:
            XbrlInstance object for the filing
        """
        from xbrl.instance import XbrlParser  # , XbrlInstance (unused)

        parser = XbrlParser(self.http_cache)  # Use renamed http_cache
        try:
            schema_url_key = "xbrl_instance_url"
            # row can be a dict if iterated from df.to_dicts() or Series from iter_rows
            # Safely get value from polars Series or dict
            schema_url = (
                row[schema_url_key]
                if isinstance(row, pl.Series)
                else row.get(schema_url_key)
            )

            if not schema_url:
                ticker_val = (
                    row["ticker"]
                    if isinstance(row, pl.Series) and "ticker" in row.index
                    else row.get("ticker", "N/A")
                )
                logger.warning(f"Missing or empty XBRL URL for ticker {ticker_val}")
                return None

            logger.debug(f"Parsing XBRL instance from: {schema_url}")
            time.sleep(self.request_interval)  # Ensure rate limiting
            inst = parser.parse_instance(schema_url)
            return inst
        except Exception as e:
            ticker_val = (
                row["ticker"]
                if isinstance(row, pl.Series) and "ticker" in row.index
                else row.get("ticker", "N/A")
            )
            filing_date_val = (
                row["filingDate"]
                if isinstance(row, pl.Series) and "filingDate" in row.index
                else row.get("filingDate", "N/A")
            )
            logger.error(
                f"Error parsing XBRL for {ticker_val} (Filing Date: {filing_date_val}): {e}",
                exc_info=True,
            )
            return None


# Example usage (commented out, would need update if used)
# if __name__ == "__main__":
#     # Replace with your information
#     user_agent = "Your Name (your.email@example.com)"

#     # Apple Inc. CIK
#     apple_cik = "0000320193"
#     apple_ticker = "AAPL"

#     fetcher = EDGARFetcher(user_agent)

#     # Example 1: Search for Apple's 10-K filings in a specific year
#     apple_10ks_2020 = fetcher.search_10k_filings(apple_ticker, apple_cik, year=2020) # Now returns Polars
#     print(f"Found {len(apple_10ks_2020)} 10-K filings for Apple Inc. in 2020")
#     if not apple_10ks_2020.is_empty():
#          print(apple_10ks_2020)

#     # Example 2: Search for Apple's 10-K filings in a year range
#     apple_10ks_range = fetcher.search_10k_filings(
#         apple_ticker, apple_cik, start_year=2021, end_year=2023 # Updated range
#     ) # Now returns Polars
#     print(
#         f"Found {len(apple_10ks_range)} 10-K filings for Apple Inc. between 2021-2023"
#     )
#     if not apple_10ks_range.is_empty():
#         print(apple_10ks_range)
#         # Example: Save the range data
#         # fetcher.save_filings_to_disk(apple_10ks_range, universe_name="AAPL_Test", file_format="parquet")

#     # Example 3: Using with a Universe requires Universe class update first
#     # (already done in previous step)
