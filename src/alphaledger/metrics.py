import time
import polars as pl
from datetime import datetime
from typing import Dict, List, Optional

from alphaledger.sec import EDGARFetcher, load_ticker_to_cik_mapping
from alphaledger.universe import Universe
from alphaledger.process_xbrl import extract_financial_metrics, ProcessingOptions
from alphaledger.config import settings
from alphaledger import get_logger

logger = get_logger(__name__)

# Define the schema for the output DataFrame for consistency
REVENUE_SCHEMA = {
    "ticker": pl.Utf8,
    "year": pl.Int64,
    "revenue": pl.Float64,  # Use Float64 to handle potential large numbers or NaNs
    "unit": pl.Utf8,
    "report_date": pl.Date,  # Use Polars Date type
}


def get_revenue_time_series(universe: Universe) -> pl.DataFrame:
    """
    Fetches revenue time series data for all tickers in a specified universe.

    This function assumes the filings data has already been fetched or loaded
    into the Universe object (e.g., via universe.get_filings()).

    Args:
        universe: An instance of the Universe class containing the securities,
                  date range, and associated SEC filings DataFrame.

    Returns:
        A Polars DataFrame with columns ['ticker', 'year', 'revenue', 'unit', 'report_date']
        containing the revenue data for each company and year found.
    """
    logger.info(
        f"Starting revenue time series collection for universe: {universe.name}"
    )

    # 1. Get Filings DataFrame from Universe object
    # This ensures filings are loaded/fetched if not already present
    filings_df = universe.get_filings()

    if filings_df is None or filings_df.is_empty():
        logger.warning(
            f"No filings data loaded or found for universe '{universe.name}'. Cannot extract revenue."
        )
        return pl.DataFrame(schema=REVENUE_SCHEMA)

    # Ensure necessary columns exist in the filings DataFrame
    required_cols = ["ticker", "accessionNumber", "reportDate", "xbrl_instance_url"]
    if not all(col in filings_df.columns for col in required_cols):
        logger.error(
            f"Filings DataFrame for universe '{universe.name}' is missing required columns: {required_cols}. Found: {filings_df.columns}"
        )
        return pl.DataFrame(schema=REVENUE_SCHEMA)

    # Filter out filings without a valid XBRL instance URL as we need it for metric extraction
    # Also filter out rows where reportDate might be unexpectedly null
    filings_df = filings_df.filter(
        pl.col("xbrl_instance_url").is_not_null() & pl.col("reportDate").is_not_null()
    )

    if filings_df.is_empty():
        logger.warning(
            f"No filings with valid XBRL URLs found for universe '{universe.name}' after filtering."
        )
        return pl.DataFrame(schema=REVENUE_SCHEMA)

    logger.info(f"Processing {len(filings_df)} filings from the loaded universe data.")

    # 2. Initialize EDGAR Fetcher (still needed for rate limiting info, potentially parsing later)
    fetcher = EDGARFetcher()

    results: List[Dict] = []
    processed_filings_count = 0

    # 3. Iterate through Filings DataFrame directly
    # Grouping by ticker might be slightly more efficient for logging, but iterating rows is fine
    for filing_row in filings_df.iter_rows(named=True):
        ticker = filing_row["ticker"]
        accession_number = filing_row["accessionNumber"]
        report_date_str = filing_row["reportDate"]  # Original string date
        xbrl_instance_url = filing_row["xbrl_instance_url"]  # Use the pre-generated URL

        # We don't need CIK here anymore as we have the filings_df

        logger.debug(
            f"Processing filing: {ticker} {accession_number} (Report Date: {report_date_str})"
        )
        logger.debug(f"  Using pre-loaded XBRL URL: {xbrl_instance_url}")
        processed_filings_count += 1

        # 4. Check if report_date is valid and get fiscal year
        try:
            # Use datetime.strptime for robust parsing before creating Polars Date
            report_date_dt = datetime.strptime(report_date_str, "%Y-%m-%d")
            fiscal_year = report_date_dt.year
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid or missing report date ('{report_date_str}') in filings data for {ticker} {accession_number}. Skipping."
            )
            continue  # Skip this filing

        # 5. Extract Financial Metrics (specifically revenue)
        try:
            options = ProcessingOptions.PRIMARY_FINANCIALS
            # Pass the URL from the filings DataFrame
            metrics = extract_financial_metrics(xbrl_instance_url, options)

            revenue_tags_to_check = [
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues",
                "Revenue",
            ]
            revenue_data = None
            for tag in revenue_tags_to_check:
                if (
                    tag in metrics
                    and metrics[tag] is not None
                    and "value" in metrics[tag]
                ):
                    revenue_data = metrics[tag]
                    logger.debug(
                        f"Found revenue using tag '{tag}' for {ticker} {fiscal_year}"
                    )
                    break

            if revenue_data:  # Check if revenue_data was found and has value
                # Convert revenue value carefully
                try:
                    revenue_value = float(revenue_data["value"])
                except (ValueError, TypeError):
                    logger.warning(
                        f"Could not convert revenue value '{revenue_data['value']}' to float for {ticker} {accession_number}. Skipping."
                    )
                    continue

                results.append(
                    {
                        "ticker": ticker,
                        "year": fiscal_year,
                        "revenue": revenue_value,
                        "unit": revenue_data.get("unit", "unknown"),
                        # Store the datetime object, Polars will handle conversion
                        "report_date": report_date_dt,
                    }
                )
                logger.info(
                    f"    Found Revenue for {ticker} {fiscal_year}: {revenue_value} {revenue_data.get('unit', '')}"
                )
            else:
                # Log if none of the revenue tags were found or had valid data
                logger.warning(
                    f"Could not find valid revenue metric in {ticker} filing {accession_number} using URL {xbrl_instance_url}."
                )

        except Exception as e:
            # Log the specific URL that failed
            logger.error(
                f"Error processing XBRL for {ticker} filing {accession_number} URL {xbrl_instance_url}: {e}",
                exc_info=True,
            )  # Add stack trace

        # Respect SEC rate limits - still important when calling extract_financial_metrics
        # as it likely makes HTTP requests via py-xbrl's cache/parser.
        time.sleep(fetcher.request_interval)

    logger.info(f"Finished processing {processed_filings_count} filings for revenue.")

    # 6. Create DataFrame
    if not results:
        logger.warning("No revenue data collected.")
        return pl.DataFrame(schema=REVENUE_SCHEMA)  # Return empty DataFrame with schema

    # Create Polars DataFrame from list of dictionaries
    try:
        revenue_df = pl.DataFrame(results, schema=REVENUE_SCHEMA)

        # Ensure types after creation (schema helps, but good practice)
        revenue_df = revenue_df.with_columns(
            [
                pl.col("year").cast(pl.Int64, strict=False),
                pl.col("revenue").cast(pl.Float64, strict=False),
            ]
        )

        # Sort using Polars
        revenue_df = revenue_df.sort(["ticker", "year"])

        logger.info(f"Successfully collected {len(revenue_df)} revenue data points.")

    except Exception as e:
        logger.error(
            f"Error creating or processing final DataFrame: {e}", exc_info=True
        )
        return pl.DataFrame(
            schema=REVENUE_SCHEMA
        )  # Return empty on DataFrame processing error

    return revenue_df


# Example Usage (adjust universe name as needed)
if __name__ == "__main__":
    # Ensure you have a universe file (e.g., 'data/universes/tech_stocks.yaml')
    UNIVERSE_NAME = (
        "tech_stocks"  # Replace with your actual universe file name (without extension)
    )

    try:
        # Optional: Pre-download/update CIK mapping if needed
        # ... (CIK download code) ...

        # --- Updated Example Usage ---
        # 1. Load the universe object (this will also trigger filings load/fetch)
        logger.info(f"Loading universe: {UNIVERSE_NAME}")
        # Make sure load_universe is imported if needed here
        from alphaledger.universe import load_universe

        # Set load_filings=True (which is the default)
        universe_obj = load_universe(UNIVERSE_NAME, load_filings=True)

        # 2. Pass the universe object to the function
        logger.info(
            "Calling get_revenue_time_series with the loaded universe object..."
        )
        revenue_data = get_revenue_time_series(universe_obj)  # Pass the object
        # --- End Updated Example Usage ---

        if not revenue_data.is_empty():  # Use Polars is_empty()
            logger.info("\nCollected Revenue Data:")
            print(revenue_data)  # Polars DataFrames print nicely

            # Example: Save to CSV using Polars
            output_path = settings.output_dir / "metrics"
            output_path.mkdir(parents=True, exist_ok=True)
            csv_path = output_path / f"{UNIVERSE_NAME}_revenue_timeseries.csv"
            revenue_data.write_csv(csv_path)  # Use Polars write_csv
            logger.info(f"\nData saved to: {csv_path}")
        else:
            logger.info("\nNo revenue data collected.")

    except FileNotFoundError:
        logger.error(f"Error: Universe file '{UNIVERSE_NAME}' not found.")
        logger.error(
            "Please ensure the universe file exists in the configured universe directory."
        )
    except Exception as e:
        logger.exception(
            f"An error occurred during the example run: {e}"
        )  # Use logger.exception for stack trace
