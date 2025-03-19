import requests
import time
import os
import json
from bs4 import BeautifulSoup
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from alphaledger.config import settings


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
    def __init__(self, user_agent=None):
        """
        Initialize the EDGAR fetcher with your contact information.

        Args:
            user_agent (str, optional): Your name, email, and organization as required by SEC
                                       If None, uses the value from settings
        """
        # Use provided user_agent or fall back to settings
        self.headers = {"User-Agent": user_agent or settings.sec_user_agent}
        self.base_url = settings.sec_base_url
        # Respect SEC rate limits (10 requests per second)
        self.request_interval = settings.sec_request_interval

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

        print(f"Fetching filings from: {url}")
        time.sleep(self.request_interval)  # Respect rate limits
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching filings: {response.status_code} - {response.reason}")
            return None

    def search_10k_filings(self, cik, year=None, start_year=None, end_year=None):
        """
        Search for 10-K filings for a specific company using the SEC submissions API.

        Args:
            cik (str): The CIK number of the company
            year (int, optional): Filter by specific filing year
            start_year (int, optional): Start year for filing search range
            end_year (int, optional): End year for filing search range

        Returns:
            A list of 10-K filing data
        """
        # Get all filings for the company
        filings_data = self.get_company_filings(cik)
        if not filings_data:
            return []

        results = []

        # The recent filings are in the "filings" section
        if "filings" not in filings_data or "recent" not in filings_data["filings"]:
            return []

        recent_filings = filings_data["filings"]["recent"]

        # Check if we have the necessary data
        if not all(
            key in recent_filings for key in ["form", "filingDate", "accessionNumber"]
        ):
            print("Warning: Filing data structure is not as expected")
            return []

        # Extract 10-K filings
        for i, form_type in enumerate(recent_filings["form"]):
            if form_type == "10-K":
                filing_date = recent_filings["filingDate"][i]
                filing_year = int(filing_date.split("-")[0])

                # Filter by year if specified
                if year and filing_year != year:
                    continue
                if (
                    start_year
                    and end_year
                    and not (start_year <= filing_year <= end_year)
                ):
                    continue

                acc_no = recent_filings["accessionNumber"][i]

                # Create a filing entry
                filing = {
                    "filing_type": form_type,
                    "filing_date": filing_date,
                    "accession_number": acc_no,
                    "cik": str(cik).zfill(10),
                    # Generate the documents URL
                    "documents_url": f"https://www.sec.gov/Archives/edgar/data/{str(cik).zfill(10)}/{acc_no.replace('-', '')}/{acc_no}-index.html",
                }

                results.append(filing)

        return results

    def fetch_filings_for_universe(self, universe, ticker_to_cik_mapping):
        """
        Fetch 10-K filings for all securities in a Universe.

        Args:
            universe: The Universe object containing securities
            ticker_to_cik_mapping: Dictionary mapping tickers to CIK numbers

        Returns:
            Dictionary mapping tickers to their filing data
        """
        all_filings = {}
        filing_years = universe.get_filing_years()

        # Use default range of last 3 years if no filing years specified
        if not filing_years:
            current_year = datetime.now().year
            start_year = current_year - 3
            end_year = current_year
            print(
                f"No filing years specified in universe. Using default range: {start_year}-{end_year}"
            )
        else:
            start_year = min(filing_years)
            end_year = max(filing_years)

        for ticker in universe.get_tickers():
            if ticker in ticker_to_cik_mapping:
                cik = ticker_to_cik_mapping[ticker]
                # Ensure CIK is properly formatted with leading zeros
                cik_formatted = str(cik).zfill(10)

                print(
                    f"Searching for {ticker} (CIK: {cik_formatted}) filings between {start_year}-{end_year}"
                )

                filings = self.search_10k_filings(
                    cik_formatted, start_year=start_year, end_year=end_year
                )

                all_filings[ticker] = filings
                print(
                    f"Found {len(filings)} 10-K filings for {ticker} between {start_year}-{end_year}"
                )
                time.sleep(
                    self.request_interval * 2
                )  # Additional delay between companies
            else:
                print(f"Warning: No CIK found for ticker {ticker}")

        return all_filings

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
