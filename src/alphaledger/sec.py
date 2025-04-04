import requests
import time
import os
import json
from bs4 import BeautifulSoup
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from alphaledger.config import settings, console
from alphaledger.universe import Universe
from alphaledger import get_logger

logger = get_logger(__name__)


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
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
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
        with open(cache_file, "r") as f:
            full_mapping = json.load(f)
    else:
        full_mapping = download_ticker_to_cik_mapping(cache_file)

    # Filter to just the tickers we need
    result = {}
    for ticker in tickers:
        upper_ticker = ticker.upper()
        if upper_ticker in full_mapping:
            result[ticker] = full_mapping[upper_ticker]

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
        self.cache_dir = cache_dir or settings.output_dir / "cache"

        # Initialize HTTP cache for XBRL parsing
        from xbrl.cache import HttpCache

        # Convert Path object to string to avoid the endswith() error
        cache_dir_str = str(self.cache_dir)
        self.cache = HttpCache(cache_dir_str)
        self.cache.set_headers({"User-Agent": self.headers["User-Agent"]})

    def get_company_filings(self, cik):
        """
        Get the list of filings for a company using the new SEC submissions API.

        Args:
            cik (str): The CIK number of the company

        Returns:
            A dictionary containing the company's filing data
        """
        # Format CIK with leading zeros to 10 digits
        cik_formatted = str(cik).zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik_formatted}.json"

        logger.info(f"Fetching filings from: {url}")

        # console.print(f"[bold magenta]Fetching filings from: {url}[/bold magenta]")
        time.sleep(self.request_interval)  # Respect rate limits
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            logger.info(f"Successfully fetched filings: {response.json}")
            return response.json()
        else:
            logger.error(
                f"[bold red]Error fetching filings: {response.status_code} - {response.reason}[/bold red]"
            )
            return None

    def search_10k_filings(
        self, ticker, cik, year=None, start_year=None, end_year=None
    ) -> pd.DataFrame:
        """
        Search for 10-K filings for a specific company using the SEC submissions API.

        Args:
            ticker (str): The ticker symbol of the company
            cik (str): The CIK number of the company
            year (int, optional): Filter by specific filing year
            start_year (int, optional): Start year for filing search range
            end_year (int, optional): End year for filing search range

        Returns:
            A DataFrame of 10-K filing data
        """
        # Get all filings for the company
        filings_data = self.get_company_filings(cik)
        if not filings_data:
            return pd.DataFrame()

        # The recent filings are in the "filings" section
        if "filings" not in filings_data or "recent" not in filings_data["filings"]:
            return pd.DataFrame()

        recent_filings = filings_data["filings"]["recent"]

        # Check if we have the necessary data
        if not all(
            key in recent_filings for key in ["form", "filingDate", "accessionNumber"]
        ):
            console.print(
                "[bold yellow]Warning: Filing data structure is not as expected[/bold yellow]"
            )
            return pd.DataFrame()

        # Create DataFrame from recent filings
        filings_df = pd.DataFrame(recent_filings)

        # Filter only 10-K filings
        form_10k_df = filings_df[filings_df["form"] == "10-K"].copy()

        # Extract filing year from filing date
        form_10k_df["filing_year"] = (
            form_10k_df["filingDate"].str.split("-").str[0].astype(int)
        )
        form_10k_df["filing_date"] = pd.to_datetime(form_10k_df["filingDate"])
        form_10k_df["report_date"] = pd.to_datetime(form_10k_df["reportDate"])

        # Apply year filters if specified
        if year is not None:
            form_10k_df = form_10k_df[form_10k_df["filing_year"] == year]
        elif start_year is not None and end_year is not None:
            form_10k_df = form_10k_df[
                (form_10k_df["filing_year"] >= start_year)
                & (form_10k_df["filing_year"] <= end_year)
            ]

        # Format CIK with leading zeros
        cik_formatted = str(cik).zfill(10)

        # Create a copy to avoid DataFrame warnings
        form_10k_df = form_10k_df.copy()

        # Add each column individually using vectorized operations where possible
        form_10k_df["cik"] = cik_formatted
        form_10k_df["ticker"] = ticker

        # Generate document URLs - using vectorized string operations
        form_10k_df["documents_url"] = (
            "https://www.sec.gov/Archives/edgar/data/"
            + cik_formatted
            + "/"
            + form_10k_df["accessionNumber"].str.replace("-", "")
            + "/"
            + form_10k_df["accessionNumber"]
            + "-index.html"
        )

        # Generate XBRL URLs - using a list comprehension instead of apply
        xbrl_urls = []
        for i, row in form_10k_df.iterrows():
            acc = row["accessionNumber"]
            if pd.isna(row["report_date"]):
                # Handle missing report date
                xbrl_urls.append(None)
            else:
                date_str = row["report_date"].strftime("%Y%m%d")
                xbrl_url = f"https://www.sec.gov/Archives/edgar/data/{cik_formatted}/{acc.replace('-', '')}/{ticker.lower()}-{date_str}.htm"
                xbrl_urls.append(xbrl_url)

        form_10k_df["xbrl_root_url"] = xbrl_urls

        return form_10k_df

    def fetch_filings_for_universe(self, universe: Universe, ticker_to_cik_mapping):
        """
        Fetch 10-K filings for all securities in a Universe.

        Args:
            universe: The Universe object containing securities
            ticker_to_cik_mapping: Dictionary mapping tickers to CIK numbers

        Returns:
            DataFrame containing all filings for the universe
        """
        all_filings_dfs = []
        filing_years = universe.get_filing_years()

        # Use default range of last 3 years if no filing years specified
        if not filing_years:
            current_year = datetime.now().year
            start_year = current_year - 3
            end_year = current_year
            console.print(
                f"[bold magenta]No filing years specified in universe. Using default range: {start_year}-{end_year}[/bold magenta]"
            )
        else:
            start_year = min(filing_years)
            end_year = max(filing_years)

        for ticker in universe.get_tickers():
            if ticker in ticker_to_cik_mapping:
                cik = ticker_to_cik_mapping[ticker]
                # Ensure CIK is properly formatted with leading zeros
                cik_formatted = str(cik).zfill(10)

                console.print(
                    f"Searching for {ticker} (CIK: {cik_formatted}) filings between {start_year}-{end_year}"
                )

                filings_df = self.search_10k_filings(
                    ticker, cik_formatted, start_year=start_year, end_year=end_year
                )

                if not filings_df.empty:
                    all_filings_dfs.append(filings_df)
                    print(
                        f"Found {len(filings_df)} 10-K filings for {ticker} between {start_year}-{end_year}"
                    )

                time.sleep(
                    self.request_interval * 2
                )  # Additional delay between companies
            else:
                print(f"Warning: No CIK found for ticker {ticker}")

        # Combine all DataFrames into a single one
        if all_filings_dfs:
            combined_df = pd.concat(all_filings_dfs, ignore_index=True)
            # Add processing date/time as metadata
            combined_df["processed_date"] = datetime.now()
            return combined_df
        else:
            return pd.DataFrame()  # Return empty DataFrame if no filings found

    def save_filings_to_disk(
        self,
        filings_df,
        output_path=None,
        file_format="parquet",
        universe_name="universe",
    ):
        """
        Save the filings DataFrame to disk.

        Args:
            filings_df: DataFrame containing the filings data
            output_path: Path where the file should be saved
            file_format: Format to save the file (parquet or csv)
            universe_name: Name of the universe (used in filename)

        Returns:
            Path to the saved file
        """
        if filings_df.empty:
            print("No filings to save.")
            return None

        # Set default output directory
        if output_path is None:
            output_path = settings.output_dir / "filings"

        # Ensure output_path is a Path object for consistent handling
        from pathlib import Path

        output_path = Path(output_path)

        # Create the directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename, ensuring we don't use a name that's already a directory
        base_filename = f"sec_filings_{universe_name}"

        # Check if the intended file path is already a directory
        if (output_path / base_filename).is_dir():
            # If it's a directory, add a timestamp to make the filename unique
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{base_filename}_{timestamp}"

        # Create full file path with filename and extension
        if file_format.lower() == "parquet":
            file_path = output_path / f"{base_filename}.parquet"
        else:
            file_path = output_path / f"{base_filename}.csv"

        # Convert to string for pandas
        file_path_str = str(file_path)

        # Ensure the parent directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Save based on format
        if file_format.lower() == "parquet":
            filings_df.to_parquet(file_path_str)
            print(f"Saved {len(filings_df)} filings to {file_path_str}")
        else:
            # CSV format
            filings_df.to_csv(file_path_str, index=False)
            print(f"Saved {len(filings_df)} filings to {file_path_str}")

        return file_path_str

    def load_filings_from_disk(self, file_path=settings.output_dir / "filings"):
        """
        Load filings from a saved file.

        Args:
            file_path: Path to the saved filings file

        Returns:
            DataFrame containing the filings data
        """
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return pd.DataFrame()

        if file_path.endswith(".parquet"):
            return pd.read_parquet(file_path)
        elif file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        else:
            print(f"Unsupported file format: {file_path}")
            return pd.DataFrame()

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
        # Remove dashes from accession number
        acc_no_clean = accession_number.replace("-", "")

        # Format the URL to get the full text submission
        url = f"{self.base_url}edgar/data/{cik}/{acc_no_clean}/{accession_number}.txt"

        time.sleep(self.request_interval)  # Respect rate limits
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            return response.text
        else:
            print(f"Error: {response.status_code} - {response.reason}")
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
        from xbrl.instance import XbrlParser, XbrlInstance

        parser = XbrlParser(self.cache)
        try:
            schema_url = row["xbrl_root_url"]
            inst = parser.parse_instance(schema_url)
            return inst
        except Exception as e:
            print(
                f"Error parsing XBRL for {row['ticker']} ({row['filing_date']}): {str(e)}"
            )
            return None


# Example usage
if __name__ == "__main__":
    # Replace with your information
    user_agent = "Your Name (your.email@example.com)"

    # Apple Inc. CIK
    apple_cik = "0000320193"

    fetcher = EDGARFetcher(user_agent)

    # Example 1: Search for Apple's 10-K filings in a specific year
    apple_10ks_2020 = fetcher.search_10k_filings(apple_cik, year=2020)
    print(f"Found {len(apple_10ks_2020)} 10-K filings for Apple Inc. in 2020")

    # Example 2: Search for Apple's 10-K filings in a year range
    apple_10ks_range = fetcher.search_10k_filings(
        apple_cik, start_year=2018, end_year=2022
    )
    print(
        f"Found {len(apple_10ks_range)} 10-K filings for Apple Inc. between 2018-2022"
    )

    # Example 3: Using with a Universe (commented out as it requires additional setup)
    """
    from alphaledger.universe import load_universe
    
    # Sample CIK mapping (in production this would come from a database or service)
    ticker_to_cik = {
        "AAPL": "0000320193",
        "MSFT": "0000789019",
        "GOOGL": "0001652044"
    }
    
    # Load universe with time range
    universe = load_universe("sample_universe", start_year=2020, end_year=2022)
    
    # Fetch filings for all tickers in the universe
    all_filings = fetcher.fetch_filings_for_universe(universe, ticker_to_cik)
    
    # Show results
    for ticker, filings in all_filings.items():
        print(f"{ticker}: {len(filings)} filings found")
    """
