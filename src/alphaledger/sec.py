import requests
import time
import os
import json
import polars as pl
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Union, TYPE_CHECKING
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
            ticker, cik = parts
            mapping[ticker.upper()] = cik.zfill(10)  # Ensure CIK is 10 digits

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
    for ticker in tickers:
        upper_ticker = ticker.upper()
        if upper_ticker in full_mapping:
            result[ticker] = full_mapping[upper_ticker]
        else:
            missing_tickers.append(ticker)

    if missing_tickers:
        logger.warning(
            f"Could not find CIKs for the following tickers: {', '.join(missing_tickers)}"
        )

    return result


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
        self.cache = HttpCache(cache_dir_str)
        self.cache.set_headers({"User-Agent": self.headers["User-Agent"]})

    def get_company_filings(self, cik, custom_url: Optional[str] = None):
        """
        Get the list of filings for a company using the new SEC submissions API.

        Args:
            cik (str): The CIK number of the company

        Returns:
            A dictionary containing the company's filing data
        """
        # Format CIK with leading zeros to 10 digits
        cik_formatted = str(cik).zfill(10)
        if custom_url:
            base_url = "https://data.sec.gov/submissions"
            url = f"{base_url}/{custom_url}"
        else:
            url = f"https://data.sec.gov/submissions/CIK{cik_formatted}.json"

        logger.info(f"Fetching filings index from: {url}")

        # console.print(f"[bold magenta]Fetching filings from: {url}[/bold magenta]") # Replaced with logger
        time.sleep(self.request_interval)  # Respect rate limits
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            logger.info(f"Successfully fetched filings index for CIK {cik_formatted}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(
                # f"[bold red]Error fetching filings: {response.status_code} - {response.reason}[/bold red]" # Replaced with logger format
                f"Error fetching filings index for CIK {cik_formatted} from {url}: {e}"
            )
            return None
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding JSON response for CIK {cik_formatted} from {url}: {e}"
            )
            return None

    def search_10k_filings(
        self,
        ticker: str,
        cik: str,
        year: Optional[int] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> pl.DataFrame:  # Updated return type
        """
        Search for 10-K filings for a specific company using the SEC submissions API.

        Args:
            ticker (str): The ticker symbol of the company
            cik (str): The CIK number of the company
            year (int, optional): Filter by specific filing year
            start_year (int, optional): Start year for filing search range
            end_year (int, optional): End year for filing search range

        Returns:
            A Polars DataFrame of 10-K filing data
        """
        # Get all filings for the company
        filings_data = self.get_company_filings(cik)
        if not filings_data:
            return pl.DataFrame()

        all_filings_list = []

        def extract_filings_from_data(filings_data):
            filings_list = []
            if isinstance(filings_data, dict):
                if isinstance(filings_data.get("accessionNumber"), list):
                    try:
                        num_filings = len(filings_data.get("accessionNumber", []))
                        keys = filings_data.keys()
                        for i in range(num_filings):
                            filing_dict = {
                                key: filings_data[key][i]
                                for key in keys
                                if i < len(filings_data[key])
                            }
                            filings_list.append(filing_dict)
                    except Exception as e:
                        logger.error(
                            f"Error processing potentially columnar 'recent' data for CIK {cik}: {e}"
                        )
                else:  # Assume it's already a list of dicts (the common case)
                    filings_list.extend(filings_data)

            return filings_list

        # 1. Process recent filings (if they exist)
        if "filings" in filings_data and "recent" in filings_data["filings"]:
            recent_filings = filings_data["filings"]["recent"]
            all_filings_list.extend(extract_filings_from_data(recent_filings))

        if "filings" in filings_data and "files" in filings_data["filings"]:
            historical_files_data = filings_data["filings"]["files"]

            for file in historical_files_data:
                logger.info(
                    f"Fetching historical file submission file {file['name']} for CIK {cik}"
                )
                all_filings_list.extend(
                    extract_filings_from_data(
                        self.get_company_filings(cik, file["name"])
                    )
                )

        if not all_filings_list:
            logger.warning(
                f"CIK {cik}: No filings found in 'recent' or 'files' sections."
            )
            return pl.DataFrame()

        # Define expected columns and types (can remain the same)
        expected_schema = {
            "accessionNumber": pl.Utf8,
            "filingDate": pl.Utf8,
            "reportDate": pl.Utf8,
            "form": pl.Utf8,
            "primaryDocument": pl.Utf8,
            "primaryDocDescription": pl.Utf8,
            # Allow other columns to exist
        }

        try:
            # Create Polars DataFrame from the combined list
            filings_df = pl.DataFrame(
                all_filings_list, schema_overrides=expected_schema
            )
        except Exception as e:
            logger.error(
                f"Error creating Polars DataFrame from combined SEC filings data for CIK {cik}: {e}",
                exc_info=True,  # Add traceback
            )
            # Log sample data for debugging
            if all_filings_list:
                logger.error(
                    f"Sample problematic data for CIK {cik}: {all_filings_list[:2]}"
                )
            return pl.DataFrame()

        # --- Debugging Step: Log raw data ---
        if not filings_df.is_empty():
            logger.debug(
                f"Combined filings data columns for CIK {cik}: {filings_df.columns}"
            )
            logger.debug(
                f"Combined filings data head for CIK {cik}:\n{filings_df.head(5)}"
            )
        else:
            logger.debug(f"No combined filings data created for CIK {cik}.")
        # --- End Debugging Step ---

        # Check if essential columns exist (can remain the same)
        required_cols = [
            "form",
            "filingDate",
            # "reportDate", # Report date might be missing in older filings sometimes
            "accessionNumber",
            "primaryDocument",
        ]
        # Check only for absolutely essential columns for filtering and URL generation
        if not all(
            col in filings_df.columns
            for col in ["form", "accessionNumber", "primaryDocument", "filingDate"]
        ):
            logger.warning(
                f"Combined filing data structure is missing essential columns for CIK {cik}. Found: {filings_df.columns}. Needed: ['form', 'accessionNumber', 'primaryDocument', 'filingDate']"
            )
            # Allow proceeding but log heavily
            # return pl.DataFrame() # Maybe too strict?

        # Filter only 10-K filings using the 'form' column (this is now applied to the combined data)
        # --- Corrected Filter ---
        form_10k_df = filings_df.filter(pl.col("form") == "10-K")  # Use 'form' column
        # --- End Correction ---

        if form_10k_df.is_empty():
            # Log check for other forms too
            logger.info(
                f"No primary 10-K filings found (form='10-K') in the combined filings for CIK {cik}. Available forms: {filings_df['form'].unique().to_list()}"
            )
            return pl.DataFrame()

        # Process dates and add columns using Polars `with_columns`
        # Use Polars expressions for transformations
        try:
            form_10k_df = form_10k_df.with_columns(
                [
                    # Extract filing year
                    pl.col("filingDate")
                    .str.split("-")
                    .list.get(0)
                    .cast(pl.Int64)
                    .alias("filing_year"),
                    # Convert filingDate to Date type
                    pl.col("filingDate")
                    .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
                    .alias("filing_date_dt"),
                    # Convert reportDate to Date type, handling potential errors
                    pl.col("reportDate")
                    .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
                    .alias("report_date_dt"),
                ]
            )
        except Exception as e:
            logger.error(
                f"Error processing dates for CIK {cik}: {e}. Some date conversions might fail."
            )
            # Continue processing, but be aware dates might be null

        # Apply year filters if specified using Polars filter
        if year is not None:
            form_10k_df = form_10k_df.filter(pl.col("filing_year") == year)
        elif start_year is not None and end_year is not None:
            form_10k_df = form_10k_df.filter(
                (pl.col("filing_year") >= start_year)
                & (pl.col("filing_year") <= end_year)
            )

        if form_10k_df.is_empty():
            logger.info(
                f"No 10-K filings found for CIK {cik} matching the year filter ({year or f'{start_year}-{end_year}'})."
            )
            return pl.DataFrame()  # Return empty Polars DF

        # Format CIK with leading zeros
        cik_formatted = str(cik).zfill(10)

        # Add remaining columns using Polars `with_columns`
        # Generate URLs using Polars string expressions
        try:
            form_10k_df = form_10k_df.with_columns(
                [
                    pl.lit(cik_formatted).alias("cik"),  # Add cik column
                    pl.lit(ticker).alias("ticker"),  # Add ticker column
                    # Generate document URLs (index page)
                    (
                        pl.lit("https://www.sec.gov/Archives/edgar/data/")
                        + pl.lit(cik_formatted)
                        + "/"
                        + pl.col("accessionNumber").str.replace_all("-", "")
                        + "/"
                        + pl.col("accessionNumber")
                        + "-index.html"
                    ).alias("documents_url"),
                    # Generate direct URL to the primary document (often the iXBRL/HTML file)
                    pl.when(
                        pl.col("primaryDocument").is_not_null()
                        & (pl.col("primaryDocument") != "")
                    )
                    .then(
                        pl.lit("https://www.sec.gov/Archives/edgar/data/")
                        + pl.lit(cik_formatted)
                        + "/"
                        + pl.col("accessionNumber").str.replace_all("-", "")
                        + "/"
                        + pl.col(
                            "primaryDocument"
                        )  # Use the primaryDocument filename directly
                    )
                    .otherwise(None)  # Set to null if primaryDocument is missing/empty
                    .alias(
                        "xbrl_instance_url"
                    ),  # Keep alias name for consistency downstream
                ]
            )
        except Exception as e:
            logger.error(f"Error adding CIK, Ticker, or URL columns for CIK {cik}: {e}")
            # Return the DataFrame as is, but URLs might be missing/incorrect

        # Select and reorder columns for final output
        final_columns = [
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
        ]
        # Filter to only columns that actually exist in the df, in the desired order
        final_columns = [col for col in final_columns if col in form_10k_df.columns]
        form_10k_df = form_10k_df.select(final_columns)

        return form_10k_df

    def fetch_filings_for_universe(
        # Use the TYPE_CHECKING import here for the type hint
        self,
        universe: "Universe",
        ticker_to_cik_mapping: Dict[str, str],
    ) -> pl.DataFrame:  # Updated return type
        """
        Fetch 10-K filings for all securities in a Universe based on its year range.

        Args:
            universe: The Universe object containing securities and year range.
            ticker_to_cik_mapping: Dictionary mapping tickers to CIK numbers.

        Returns:
            Polars DataFrame containing all filings found for the universe within its range.
        """
        all_filings_dfs: List[pl.DataFrame] = []  # List to hold Polars DataFrames
        filing_years = universe.get_filing_years()

        # Use default range if no filing years specified
        if not filing_years:
            current_year = datetime.now().year
            start_year = current_year - 3  # Default to last 3 years + current
            end_year = current_year
            logger.info(
                f"No filing years specified in universe '{universe.name}'. Using default range: {start_year}-{end_year}"
            )
        else:
            start_year = min(filing_years)
            end_year = max(filing_years)

        for ticker in universe.get_tickers():
            if ticker in ticker_to_cik_mapping:
                cik = ticker_to_cik_mapping[ticker]
                # Ensure CIK is properly formatted
                cik_formatted = str(cik).zfill(10)

                logger.info(
                    f"Searching for {ticker} (CIK: {cik_formatted}) filings between {start_year}-{end_year}"
                )

                # This now returns a Polars DataFrame
                filings_df = self.search_10k_filings(
                    ticker, cik_formatted, start_year=start_year, end_year=end_year
                )

                if not filings_df.is_empty():  # Use Polars is_empty()
                    all_filings_dfs.append(filings_df)
                    logger.info(
                        f"Found {len(filings_df)} 10-K filings for {ticker} between {start_year}-{end_year}"
                    )
                else:
                    logger.info(
                        f"No 10-K filings found for {ticker} in period {start_year}-{end_year}."
                    )

                time.sleep(
                    self.request_interval
                    * 1.5  # Slightly increased delay between companies
                )
            else:
                # logger already warns about missing CIKs in load_ticker_to_cik_mapping
                pass

        # Combine all Polars DataFrames into a single one
        if all_filings_dfs:
            # Use pl.concat for combining Polars DataFrames
            combined_df = pl.concat(
                all_filings_dfs, how="vertical_relaxed"
            )  # Use relaxed to handle potential schema variations slightly
            # Add processing date/time as metadata using Polars
            combined_df = combined_df.with_columns(
                pl.lit(datetime.now()).alias("processed_datetime")
            )
            logger.info(
                f"Combined {len(combined_df)} filings for universe '{universe.name}'."
            )
            return combined_df
        else:
            logger.warning(
                f"No filings found for any ticker in universe '{universe.name}'."
            )
            return pl.DataFrame()  # Return empty Polars DataFrame if no filings found

    def fetch_specific_filings(
        self,
        filing_combinations: List[Dict[str, Union[str, int]]],
        ticker_to_cik_mapping: Dict[str, str],
    ) -> pl.DataFrame:
        """
        Fetch specific 10-K filings based on a list of (ticker, filing_year) combinations.

        Args:
            filing_combinations: List of dictionaries, each like {'ticker': 'T', 'filing_year': Y}.
            ticker_to_cik_mapping: Dictionary mapping required tickers to CIK numbers.

        Returns:
            A Polars DataFrame containing the successfully fetched filings.
        """
        specific_filings_dfs: List[pl.DataFrame] = []
        logger.info(
            f"Starting fetch for {len(filing_combinations)} specific filing combinations."
        )

        processed_combinations = 0
        for combo in filing_combinations:
            ticker = combo.get("ticker")
            year = combo.get("filing_year")

            if not isinstance(ticker, str) or not isinstance(year, int):
                logger.warning(f"Skipping invalid combination: {combo}")
                continue

            if ticker in ticker_to_cik_mapping:
                cik = ticker_to_cik_mapping[ticker]
                cik_formatted = str(cik).zfill(10)

                logger.debug(
                    f"Fetching specific filing for {ticker} (CIK: {cik_formatted}), year: {year}"
                )

                # Reuse existing search method, filtering by specific year
                filing_df = self.search_10k_filings(ticker, cik_formatted, year=year)

                if not filing_df.is_empty():
                    # We expect only one 10-K per year, but search_10k_filings might return amendments.
                    # Filter further if needed, or just take the first one.
                    # For simplicity, let's assume search_10k_filings is precise enough or take the latest if multiple.
                    if len(filing_df) > 1:
                        logger.warning(
                            f"Found {len(filing_df)} filings for {ticker} in year {year}. Using the latest based on filingDate."
                        )
                        # Ensure 'filing_date_dt' exists and is sortable
                        if (
                            "filing_date_dt" in filing_df.columns
                            and filing_df["filing_date_dt"].dtype == pl.Date
                        ):
                            filing_df = filing_df.sort(
                                "filing_date_dt", descending=True
                            ).head(1)
                        else:
                            # Fallback if date conversion failed or column missing
                            filing_df = filing_df.head(1)

                    specific_filings_dfs.append(filing_df)
                    logger.debug(
                        f"Successfully fetched filing for {ticker}, year {year}."
                    )
                else:
                    logger.warning(
                        f"Did not find 10-K filing for {ticker}, year {year}."
                    )

                processed_combinations += 1
                if processed_combinations % 5 == 0:  # Add delay every few requests
                    time.sleep(self.request_interval * 1.5)
                else:
                    time.sleep(self.request_interval)  # Standard delay

            else:
                logger.warning(
                    f"Skipping {ticker} year {year}: CIK not found in provided mapping."
                )

        # Combine results
        if specific_filings_dfs:
            combined_df = pl.concat(specific_filings_dfs, how="vertical_relaxed")
            # Add processing date/time
            combined_df = combined_df.with_columns(
                pl.lit(datetime.now()).alias("processed_datetime")
            )
            logger.info(
                f"Finished fetching. Combined {len(combined_df)} specific filings successfully."
            )
            return combined_df
        else:
            logger.info(
                "Finished fetching specific filings. No new filings were found."
            )
            return pl.DataFrame()  # Return empty DataFrame if nothing was fetched

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
                filings_df.write_delta(file_path)
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

    def load_filings_from_disk(
        self, file_path: Union[str, Path]
    ) -> pl.DataFrame:  # Updated return type and input hint
        """
        Load filings from a saved file into a Polars DataFrame.

        Args:
            file_path: Path (string or Path object) to the saved filings file

        Returns:
            Polars DataFrame containing the filings data, or empty DataFrame if error/not found.
        """
        file_path_obj = Path(file_path)  # Ensure Path object

        if not file_path_obj.exists():
            logger.error(f"File not found: {file_path_obj}")
            return pl.DataFrame()  # Return empty Polars DF

        try:
            if file_path_obj.suffix.lower() == ".parquet":
                logger.info(f"Loading Parquet file: {file_path_obj}")
                return pl.read_parquet(file_path_obj)
            elif file_path_obj.suffix.lower() == ".csv":
                logger.info(f"Loading CSV file: {file_path_obj}")
                # Use infer_schema_length=10000 for better type inference on larger CSVs
                return pl.read_csv(
                    file_path_obj, infer_schema_length=10000, try_parse_dates=True
                )
            elif file_path_obj.suffix.lower() == ".delta":
                logger.info(f"Loading Delta file: {file_path}")
                return pl.read_delta(file_path)
            else:
                logger.error(f"Unsupported file format: {file_path_obj.suffix}")
                return pl.DataFrame()  # Return empty Polars DF
        except Exception as e:
            logger.error(f"Error loading file {file_path_obj}: {e}", exc_info=True)
            return pl.DataFrame()  # Return empty Polars DF

    def get_filing_by_accession_number(self, cik, accession_number, file_type="10-K"):
        """
        Get a specific filing by accession number.

        Args:
            cik (str): The CIK number of the company
            accession_number (str): The accession number of the filing
            file_type (str): The type of filing (default: "10-K")

        Returns:
            The text content of the filing
        """
        # Format CIK with leading zeros to 10 digits
        cik_formatted = str(cik).zfill(10)
        # Remove dashes from accession number
        acc_no_clean = accession_number.replace("-", "")

        # Format the URL to get the full text submission
        # Check if it's the full submission txt or a specific document
        # Assuming full submission based on '.txt'
        # Example: https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm (primary doc)
        # Example: https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/0000320193-23-000106.txt (full submission)

        # Let's stick to the full submission text file for now
        url = f"{self.base_url}Archives/edgar/data/{cik_formatted}/{acc_no_clean}/{accession_number}.txt"
        logger.debug(f"Fetching full submission text from: {url}")

        time.sleep(self.request_interval)  # Respect rate limits
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Error fetching filing text for CIK {cik}, Acc# {accession_number}: {e}"
            )
            return None

    def extract_text_from_filing(self, filing_text):
        """
        Extract the relevant text content from an SEC filing.
        This is a basic implementation - you may need more sophisticated parsing
        based on your specific needs.

        Args:
            filing_text (str): The raw text of the SEC filing

        Returns:
            The extracted text content (simplified)
        """
        from bs4 import BeautifulSoup  # Keep import local if only used here

        # Split by <DOCUMENT> tags to separate documents
        documents = filing_text.split("<DOCUMENT>")

        for doc in documents:
            # Look for 10-K document
            if "<TYPE>10-K" in doc:
                # Find the text content - this is simplified
                # Real implementation would need better parsing
                start_idx = doc.find("<TEXT>")
                end_idx = doc.find("</TEXT>")

                if start_idx > 0 and end_idx > 0:
                    text_content = doc[start_idx + 6 : end_idx]
                    # Remove HTML tags if present (simplified)
                    soup = BeautifulSoup(text_content, "html.parser")
                    return soup.get_text()

        return "No text content found in the filing."

    def parse_filing_xbrl(self, row):
        """
        Parse an XBRL filing using the schema and document URLs.

        Args:
            row: DataFrame row containing 'documents_url' and 'xbrl_root_url'

        Returns:
            XbrlInstance object for the filing
        """
        from xbrl.instance import XbrlParser, XbrlInstance  # Keep import local

        parser = XbrlParser(self.cache)
        try:
            # Assuming row has 'xbrl_instance_url' or similar key
            schema_url_key = (
                "xbrl_instance_url"  # Match name used in search_10k_filings
            )
            if schema_url_key not in row or not row[schema_url_key]:
                logger.warning(
                    f"Missing or empty XBRL URL key ('{schema_url_key}') in row for ticker {row.get('ticker', 'N/A')}"
                )
                return None

            schema_url = row[schema_url_key]
            logger.debug(f"Parsing XBRL instance from: {schema_url}")
            inst = parser.parse_instance(schema_url)
            return inst
        except Exception as e:
            ticker = row.get("ticker", "N/A")
            filing_date = row.get("filingDate", "N/A")
            logger.error(
                f"Error parsing XBRL for {ticker} (Filing Date: {filing_date}): {str(e)}",
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
