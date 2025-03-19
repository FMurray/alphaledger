import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import logging
from pathlib import Path

from alphaledger.universe import load_universe, Universe
from alphaledger.sec import EDGARFetcher, load_ticker_to_cik_mapping
from alphaledger.config import settings


def process_filing_contents(
    sec_fetcher: EDGARFetcher,
    filings_metadata: Dict,
    output_dir: Path,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> int:
    """
    Download and process the actual content of SEC filings.

    Args:
        sec_fetcher: Initialized EDGARFetcher
        filings_metadata: Dictionary of filing metadata by ticker
        output_dir: Directory to save processed filings
        verbose: Whether to log detailed information
        logger: Logger instance to use

    Returns:
        Number of filings processed
    """
    filings_dir = output_dir / "filings"
    os.makedirs(filings_dir, exist_ok=True)

    processed_count = 0

    for ticker, ticker_filings in filings_metadata.items():
        ticker_dir = filings_dir / ticker
        os.makedirs(ticker_dir, exist_ok=True)

        for filing in ticker_filings:
            # Extract filing date and create a filename
            filing_date = filing.get("filing_date", "unknown_date")
            try:
                date_obj = datetime.strptime(filing_date, "%Y-%m-%d")
                year = date_obj.year
            except ValueError:
                # Try another common format
                try:
                    date_obj = datetime.strptime(filing_date, "%m/%d/%Y")
                    year = date_obj.year
                except ValueError:
                    year = "unknown_year"

            filename = f"{ticker}_10K_{year}.txt"
            filepath = ticker_dir / filename

            # Skip if already downloaded
            if os.path.exists(filepath):
                if verbose and logger:
                    logger.info(f"Skipping already downloaded filing: {filepath}")
                continue

            # Extract accession number from URL
            documents_url = filing.get("documents_url", "")
            if "/accession_number=" in documents_url:
                accession_number = documents_url.split("accession_number=")[1].split(
                    "&"
                )[0]

                # Get the full filing text using CIK stored in the filing metadata
                if verbose and logger:
                    logger.info(
                        f"Fetching filing: {ticker} - {year} (Accession: {accession_number})"
                    )

                cik = filing.get("cik")
                if cik:
                    filing_text = sec_fetcher.get_filing_by_accession_number(
                        cik, accession_number
                    )

                    if filing_text:
                        # Save raw filing
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(filing_text)

                        # Extract and save just the text content
                        text_content = sec_fetcher.extract_text_from_filing(filing_text)
                        text_filepath = filepath.replace(".txt", "_text.txt")
                        with open(text_filepath, "w", encoding="utf-8") as f:
                            f.write(text_content)

                        processed_count += 1
                else:
                    if verbose and logger:
                        logger.warning(
                            f"No CIK information found for filing {ticker} - {year}"
                        )

    return processed_count


def build_kb(logger: Optional[logging.Logger] = None) -> None:
    """
    Build the knowledge base from a specified universe.

    Args:
        logger: Logger instance
    """
    # Load the universe with specified time period
    universe = load_universe(
        settings.universe_name, settings.start_year, settings.end_year
    )

    if settings.verbose and logger:
        logger.info(f"Universe loaded: {universe}")
        logger.info(f"Knowledge base depth: {settings.kb_depth}")
        logger.info(f"Analysis period: {settings.start_year} - {settings.end_year}")
        logger.info(f"Using SEC EDGAR with user agent: {settings.sec_user_agent}")

    # Create output directory if it doesn't exist
    os.makedirs(settings.output_dir, exist_ok=True)

    # Load ticker to CIK mapping using the function from sec.py
    cache_file = settings.output_dir / "ticker_to_cik.json"
    ticker_to_cik = load_ticker_to_cik_mapping(universe.get_tickers(), str(cache_file))

    if settings.verbose and logger:
        logger.info(
            f"Found CIK mappings for {len(ticker_to_cik)} out of {len(universe.get_tickers())} tickers"
        )

    # Initialize SEC fetcher
    sec_fetcher = EDGARFetcher(settings.sec_user_agent)

    # Fetch SEC filings
    if settings.verbose and logger:
        logger.info("Fetching SEC filings metadata...")

    filings = sec_fetcher.fetch_filings_for_universe(universe, ticker_to_cik)

    # Save filing metadata
    metadata_path = settings.output_dir / "filings_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(filings, f, indent=2)

    if settings.verbose and logger:
        total_filings = sum(len(f) for f in filings.values())
        logger.info(
            f"Fetched metadata for {total_filings} filings from {len(filings)} companies"
        )
        logger.info(f"Saved filing metadata to {metadata_path}")

    # Download and process actual filing content if depth > 1
    if settings.kb_depth > 1:
        if settings.verbose and logger:
            logger.info("Downloading and processing filing contents...")
        processed = process_filing_contents(
            sec_fetcher, filings, settings.output_dir, settings.verbose, logger
        )
        if settings.verbose and logger:
            logger.info(f"Downloaded and processed {processed} filing documents")
