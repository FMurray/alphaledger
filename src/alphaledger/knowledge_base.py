import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Literal
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import lancedb
from lancedb.table import Table
import pyarrow as pa
import math
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    SpinnerColumn,
    TimeElapsedColumn,
)

from alphaledger.universe import load_universe, Universe
from alphaledger.sec import EDGARFetcher, load_ticker_to_cik_mapping
from alphaledger.config import settings

# Default embedding dimension for OpenAI text-embedding models
DEFAULT_EMBEDDING_DIM = 3072


class KnowledgeBase:
    """
    Knowledge base for storing and retrieving information about securities.

    Supports both local and remote storage using LanceDB.
    """

    def __init__(
        self,
        uri: str = None,
        table_name: str = "securities",
        embedding_model: str = "text-embedding-3-small",
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        create_if_not_exists: bool = True,
        index_metric: Literal["l2", "cosine", "dot"] = "l2",
        index_type: str = "IVF_PQ",
        index_num_partitions: int = 256,
        index_num_sub_vectors: Optional[int] = None,
        index_num_bits: int = 8,
        accelerator: Optional[str] = None,
    ):
        """
        Initialize a knowledge base.

        Args:
            uri: LanceDB connection URI (local://path or s3://bucket/path)
            table_name: Name of the table to store documents
            embedding_model: Model to use for generating embeddings
            embedding_dim: Dimension of embeddings
            create_if_not_exists: Create the table if it doesn't exist
            index_type: Type of index to use ("IVF_PQ", "FLAT", etc.)
            index_num_partitions: Number of partitions for IVF_PQ index
            index_num_sub_vectors: Number of sub-vectors for PQ (if None, defaults to dimension/16)
            index_num_bits: Number of bits for encoding each sub-vector (4 or 8)
            accelerator: GPU acceleration ('cuda', 'mps', or None)
        """
        self.uri = uri
        self.table_name = table_name
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.index_metric = index_metric
        self.index_type = index_type
        self.index_num_partitions = index_num_partitions
        self.index_num_sub_vectors = index_num_sub_vectors or self.embedding_dim // 16
        self.index_num_bits = index_num_bits
        self.accelerator = accelerator

        # Connect to LanceDB
        self.db = lancedb.connect(uri)

        # Create or load the table
        if create_if_not_exists:
            self._create_or_load_table()
        else:
            self.table = self.db.open_table(table_name)

    def _create_or_load_table(self):
        """Create the table if it doesn't exist, or load it if it does."""
        try:
            # Try to open the existing table
            self.table = self.db.open_table(self.table_name)
        except Exception:
            # Create a PyArrow schema with the correct data types and vector dimension
            schema = pa.schema(
                [
                    pa.field("ticker", pa.string()),
                    pa.field("text", pa.string()),
                    pa.field("date", pa.string()),
                    pa.field("source", pa.string()),
                    pa.field("section", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), self.embedding_dim)),
                ]
            )

            # Create the table with the PyArrow schema
            self.table = self.db.create_table(
                self.table_name, schema=schema, mode="overwrite"
            )
            # Index will be created after data is added

    def _ensure_index_exists(self):
        """Create an index if it doesn't already exist and we have data."""
        # Check if table has data before creating index
        table_size = len(self.table)
        if table_size == 0:
            return

        # For IVF_PQ, we need at least num_partitions data points
        # If we don't have enough, adjust the number of partitions down
        index_params = {
            "replace": True,
        }

        if self.index_type == "IVF_PQ":
            # The number of partitions should be less than the number of vectors
            # A good rule of thumb is sqrt(n) for n vectors
            adjusted_partitions = min(
                self.index_num_partitions, max(2, int(math.sqrt(table_size)))
            )

            if adjusted_partitions < self.index_num_partitions:
                print(
                    f"WARNING: Reducing number of partitions from {self.index_num_partitions} to {adjusted_partitions} based on table size ({table_size} vectors)"
                )

            # LanceDB requires num_bits=8 for PQ
            num_bits = 8
            if self.index_num_bits != num_bits:
                print(
                    f"WARNING: Setting num_bits to {num_bits} for PQ index (was {self.index_num_bits})"
                )

            # Calculate appropriate number of sub-vectors
            # Each sub-vector must be a multiple of 4 for efficiency
            # and we need to ensure we don't have too many sub-vectors
            # LanceDB recommends embedding_dim / 16 as a default
            sub_vector_size = 16  # Recommended value for efficiency
            num_sub_vectors = max(1, min(self.embedding_dim // sub_vector_size, 96))

            # For extremely large embedding dimensions, further reduce sub-vectors
            # to avoid memory issues or compatibility problems
            if num_sub_vectors > 32:
                num_sub_vectors = 32

            if num_sub_vectors != self.index_num_sub_vectors:
                print(
                    f"WARNING: Adjusting num_sub_vectors from {self.index_num_sub_vectors} to {num_sub_vectors} for compatibility"
                )

            index_params.update(
                {
                    "num_partitions": adjusted_partitions,
                    "num_sub_vectors": num_sub_vectors,
                    "num_bits": num_bits,
                }
            )

        # Add GPU acceleration if specified
        if self.accelerator:
            index_params["accelerator"] = self.accelerator

        # Create the index with configured parameters
        try:
            self.table.create_index(
                self.index_metric, index_type=self.index_type, **index_params
            )
        except Exception as e:
            print(f"ERROR creating index: {e}")
            print("Falling back to IVF index without PQ for reliability")
            # Fall back to a simpler index type (IVF without PQ)
            try:
                self.table.create_index(
                    self.index_metric,
                    index_type="IVF",
                    num_partitions=max(2, int(math.sqrt(table_size))),
                    replace=True,
                )
            except Exception as e2:
                print(f"ERROR creating IVF index: {e2}")
                print("Falling back to no index")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a text.

        Args:
            text: The text to embed

        Returns:
            A list of floats representing the embedding
        """
        # This is a placeholder. In a real implementation, you would call
        # an embedding API (OpenAI, local model, etc.)
        try:
            from alphaledger.models import AzureOpenAIEmbedding

            client = AzureOpenAIEmbedding()
            response = client.embed_documents(texts=[text])
            embedding = response.data[0].embedding

            # Ensure the embedding has the expected dimension
            if len(embedding) != self.embedding_dim:
                print(
                    f"WARNING: Expected embedding dimension {self.embedding_dim}, got {len(embedding)}"
                )
                # Either pad or truncate to match expected dimension
                if len(embedding) < self.embedding_dim:
                    # Pad with zeros
                    embedding = embedding + [0.0] * (
                        self.embedding_dim - len(embedding)
                    )
                else:
                    # Truncate
                    embedding = embedding[: self.embedding_dim]

            return embedding
        except ImportError:
            # If OpenAI is not available, return a random embedding
            # (for testing purposes only)
            print("WARNING: OpenAI not available, using random embeddings")
            return list(np.random.rand(self.embedding_dim))
        except Exception as e:
            print(f"ERROR generating embedding: {e}")
            # In case of error, return a zero vector of the correct dimension
            return [0.0] * self.embedding_dim

    def add_document(self, document: Dict[str, Any]):
        """
        Add a single document to the knowledge base.

        Args:
            document: A dictionary with at least 'ticker' and 'text' keys.
                     Optional: 'date', 'source'
        """
        self.add_documents([document])

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add multiple documents to the knowledge base.

        Args:
            documents: A list of dictionaries, each with at least 'ticker' and 'text' keys.
                      Optional: 'date', 'source'
        """
        if not documents:
            return

        # Prepare the documents for insertion
        df_docs = []
        for doc in documents:
            # Generate embedding if not provided
            if "vector" not in doc:
                doc["vector"] = self.generate_embedding(doc["text"])

            # Set default values for optional fields
            if "date" not in doc:
                doc["date"] = datetime.now().strftime("%Y-%m-%d")
            if "source" not in doc:
                doc["source"] = "unknown"

            df_docs.append(doc)

        # Convert to dataframe and add to table
        df = pd.DataFrame(df_docs)
        self.table.add(df)

        # Create index after adding data if it doesn't exist yet
        self._ensure_index_exists()

    def search(
        self,
        query: str,
        ticker: Optional[str] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for documents matching the query.

        Args:
            query: The search query
            ticker: Optional ticker to filter results
            limit: Maximum number of results to return
            min_score: Minimum similarity score (0-1)

        Returns:
            A list of matching documents with similarity scores
        """
        # Generate embedding for the query
        query_embedding = self.generate_embedding(query)

        # Build the search query
        search_query = self.table.search(query_embedding)

        # Filter by ticker if provided
        if ticker:
            search_query = search_query.where(f"ticker = '{ticker}'")

        # Execute the search
        results = search_query.limit(limit).to_pandas()

        # Convert to list of dictionaries and add similarity score
        docs = []
        for _, row in results.iterrows():
            doc = row.to_dict()
            # Calculate cosine similarity score (0-1)
            score = float(
                np.dot(query_embedding, doc["vector"])
                / (np.linalg.norm(query_embedding) * np.linalg.norm(doc["vector"]))
            )

            # Skip if below minimum score
            if score < min_score:
                continue

            # Clean up the result
            doc["score"] = score
            doc.pop("vector", None)  # Remove the embedding from the result
            docs.append(doc)

        return docs

    def get_documents_for_ticker(
        self, ticker: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all documents for a specific ticker.

        Args:
            ticker: The ticker symbol
            limit: Maximum number of documents to return

        Returns:
            A list of documents
        """
        results = self.table.where(f"ticker = '{ticker}'").limit(limit).to_pandas()

        # Convert to list of dictionaries and remove embeddings
        docs = []
        for _, row in results.iterrows():
            doc = row.to_dict()
            doc.pop("vector", None)  # Remove the embedding
            docs.append(doc)

        return docs

    def clear(self):
        """Delete all documents from the knowledge base."""
        self.db.drop_table(self.table_name)
        self._create_or_load_table()

    def close(self):
        """Close the connection to the database."""
        # LanceDB handles this automatically, but including for completeness
        pass


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

            # Get the accession number directly from the metadata
            accession_number = filing.get("accession_number", "")

            # If accession number isn't directly available, try to extract it from URL
            if not accession_number:
                documents_url = filing.get("documents_url", "")
                # The URL format is typically: .../data/0001018724/000101872424000008/0001018724-24-000008-index.html
                # Extract the accession number from the path segment
                if "/Archives/edgar/data/" in documents_url:
                    try:
                        # Extract from path - take the second-to-last segment before the filename
                        path_parts = documents_url.split("/")
                        accession_number = path_parts[-2]
                    except Exception as e:
                        if verbose and logger:
                            logger.warning(
                                f"Failed to extract accession number from URL: {documents_url}, error: {e}"
                            )

            # Get the full filing text using CIK stored in the filing metadata
            if accession_number:
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
                        # Use the Path object's proper method for manipulating paths
                        text_filepath = filepath.with_name(filepath.stem + "_text.txt")
                        with open(text_filepath, "w", encoding="utf-8") as f:
                            f.write(text_content)

                        processed_count += 1
                    else:
                        if verbose and logger:
                            logger.warning(
                                f"Failed to fetch filing content for {ticker} - {year} (Accession: {accession_number})"
                            )
                else:
                    if verbose and logger:
                        logger.warning(
                            f"No CIK information found for filing {ticker} - {year}"
                        )
            else:
                if verbose and logger:
                    logger.warning(
                        f"No accession number found for filing {ticker} - {year}"
                    )

    return processed_count


def fetch_and_process_filing(
    ticker: str,
    filing_type: str,
    filing_date: str,
    accession_number: str,
    documents_url: str,
    output_dir: Path,
    depth: int = 1,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch and process an SEC filing into chunks suitable for the knowledge base.

    Args:
        ticker: The ticker symbol
        filing_type: The type of filing (10-K, 10-Q, etc.)
        filing_date: The date of the filing
        accession_number: The SEC accession number
        documents_url: URL to the filing documents
        output_dir: Base output directory
        depth: Depth of processing (1-3)
        verbose: Whether to print progress
        logger: Logger instance to use

    Returns:
        List of text chunks with metadata
    """
    from .process_xbrl import (
        ProcessingOptions,
        process_filing_content,
        generate_placeholder_data,
    )

    # Convert depth to processing options
    options = ProcessingOptions.NONE
    if depth == 1:
        options = ProcessingOptions.LEVEL_1
    elif depth == 2:
        options = ProcessingOptions.LEVEL_2
    else:  # depth 3 or higher
        options = ProcessingOptions.LEVEL_3

    # Log the options if verbose
    if verbose and logger:
        logger.info(f"Processing {ticker} {filing_type} with options: {options}")

    # This function processes filings that were already downloaded
    filings_dir = output_dir / "filings" / ticker

    # Try to find the filing based on the year from filing_date
    year = filing_date.split("-")[0]
    text_file_path = filings_dir / f"{ticker}_10K_{year}_text.txt"
    raw_file_path = filings_dir / f"{ticker}_10K_{year}.txt"

    chunks = []

    # If we have the processed text file, use it
    if os.path.exists(text_file_path) and os.path.exists(raw_file_path):
        if verbose and logger:
            logger.info(f"Using existing processed filing: {text_file_path}")

        # Read the file for text-based processing
        try:
            with open(text_file_path, "r", encoding="utf-8") as f:
                full_text = f.read()

            # Also load the raw filing for XBRL extraction if needed
            with open(raw_file_path, "r", encoding="utf-8") as f:
                raw_filing = f.read()

            # Check if we need to extract XBRL data and if the raw filing contains XBRL
            has_xbrl = (
                "<xbrl" in raw_filing.lower() or "xmlns:xbrl" in raw_filing.lower()
            )

            if (
                ProcessingOptions.PRIMARY_FINANCIALS in options
                or ProcessingOptions.DETAILED_FINANCIALS in options
            ) and not has_xbrl:
                # If XBRL processing is requested but the filing doesn't seem to have XBRL data,
                # try to find the XBRL URL from the documents_url
                if documents_url and "/Archives/edgar/data/" in documents_url:
                    # The original URL typically points to an index page
                    # Try to construct a URL to the XBRL instance document
                    base_url = documents_url.split("/index.html")[0]
                    possible_xbrl_urls = [
                        f"{base_url}/{ticker.lower()}-{year}0101.xml",  # Common format
                        f"{base_url}/{ticker.lower()}-{year}1231.xml",  # Year-end format
                        f"{base_url}/{ticker.lower()}.xml",  # Simple format
                        f"{base_url}/Financial_Report.xml",  # Generic format
                    ]

                    if verbose and logger:
                        logger.info(
                            f"Filing doesn't contain embedded XBRL. Will try direct XBRL URLs."
                        )

                    # Use the URL directly in our extraction function instead of the raw filing
                    for xbrl_url in possible_xbrl_urls:
                        if verbose and logger:
                            logger.info(f"Trying XBRL URL: {xbrl_url}")
                        raw_filing = xbrl_url  # Pass the URL instead of content
                        break  # Just try the first URL for now

            # Process the filing content with the specified options
            chunks = process_filing_content(
                ticker, filing_type, filing_date, full_text, raw_filing, options, logger
            )

        except Exception as e:
            if verbose and logger:
                logger.error(f"Error processing {text_file_path}: {e}")

            # Fall back to placeholder content
            chunks.append(
                {
                    "text": f"Error processing {ticker} {filing_type} from {filing_date}.",
                    "section": "error",
                    "section_name": "Processing Error",
                }
            )
    else:
        # If we don't have the file, use placeholder data
        if verbose and logger:
            logger.warning(
                f"No processed file found for {ticker} {filing_type} {year}, using placeholder data"
            )

        # Generate placeholder data based on options
        chunks = generate_placeholder_data(ticker, filing_type, filing_date, options)

    return chunks


def build_kb(
    logger: Optional[logging.Logger] = None, filings_file: Optional[str] = None
) -> KnowledgeBase:
    """
    Build the knowledge base from a specified universe.

    Args:
        logger: Logger instance
        filings_file: Path to the parquet file containing filings data

    Returns:
        The populated knowledge base
    """
    # Create output directory if it doesn't exist
    os.makedirs(settings.output_dir, exist_ok=True)

    # Set up logging to capture LanceDB output
    if settings.verbose:
        logging_level = logging.INFO
    else:
        # Set LanceDB logs to WARNING to reduce output noise
        logging_level = logging.WARNING

    lancedb_logger = logging.getLogger("lancedb")
    lancedb_logger.setLevel(logging_level)

    # Continue with the same Rich progress implementation as before
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        # Main task for overall progress
        main_task = progress.add_task("[bold green]Building knowledge base...", total=5)

        # Load the universe with specified time period
        universe_task = progress.add_task("[cyan]Loading universe...", total=1)
        universe = load_universe(
            settings.universe_name, settings.start_year, settings.end_year
        )
        progress.update(universe_task, advance=1)
        progress.update(main_task, advance=1)

        # Initialize the knowledge base
        kb_task = progress.add_task("[cyan]Initializing knowledge base...", total=1)
        kb = KnowledgeBase(
            uri=settings.kb_uri,
            embedding_model=settings.embedding_model,
            index_metric=settings.kb_index_metric,
            index_type=settings.kb_index_type,
            index_num_partitions=settings.kb_index_num_partitions,
            index_num_sub_vectors=settings.kb_index_num_sub_vectors,
            index_num_bits=settings.kb_index_num_bits,
            accelerator=settings.kb_accelerator,
        )
        progress.update(kb_task, advance=1)

        # Process filings - load from parquet file if provided
        filings_task = progress.add_task("[cyan]Processing filings data...", total=1)

        # Initialize SEC fetcher
        sec_fetcher = EDGARFetcher(settings.sec_user_agent)

        if filings_file and os.path.exists(filings_file):
            progress.console.print(f"[yellow]Loading filings data from {filings_file}")
            filings_df = pd.read_parquet(filings_file)
            progress.console.print(
                f"[green]Loaded {len(filings_df)} filings for {filings_df['ticker'].nunique()} companies"
            )
        else:
            progress.console.print(
                "[yellow]No filings file found. Fetching filings for universe..."
            )

            # Load ticker to CIK mapping
            cache_file = settings.output_dir / "ticker_to_cik.json"
            ticker_to_cik = load_ticker_to_cik_mapping(
                universe.get_tickers(), str(cache_file)
            )

            # Fetch filings for universe
            filings_df = sec_fetcher.fetch_filings_for_universe(universe, ticker_to_cik)

            # Save filings to disk
            filings_dir = settings.output_dir / "filings"
            os.makedirs(filings_dir, exist_ok=True)
            filings_file = sec_fetcher.save_filings_to_disk(
                filings_df,
                output_path=filings_dir,
                file_format="parquet",
                universe_name=settings.universe_name,
            )

            progress.console.print(
                f"[green]Fetched and saved {len(filings_df)} filings for {filings_df['ticker'].nunique()} companies"
            )

        progress.update(filings_task, advance=1)
        progress.update(main_task, advance=1)

        # # Download and process actual filing content if depth > 1
        # if settings.kb_depth > 1 and not filings_df.empty:
        #     processing_task = progress.add_task(
        #         "[cyan]Processing filing contents...", total=1
        #     )
        #     processed = process_filing_contents_from_df(
        #         sec_fetcher, filings_df, settings.output_dir, settings.verbose, logger
        #     )
        #     progress.console.print(
        #         f"[green]Downloaded and processed {processed} filing documents"
        #     )
        #     progress.update(processing_task, advance=1)

        # progress.update(main_task, advance=1)

        # # Temporarily replace the _ensure_index_exists method to prevent index creation
        # original_ensure_index = kb._ensure_index_exists
        # kb._ensure_index_exists = lambda: None  # Do nothing

        # # Build the knowledge base from the filings
        # kb_build_task = progress.add_task(
        #     "[cyan]Adding documents to knowledge base...",
        #     total=len(universe.get_tickers()),
        # )

        # document_count = 0

        # # Collect information for each security - modified to track progress
        # for ticker in universe.get_tickers():
        #     security = universe.get_security(ticker)

        #     progress.console.print(f"[blue]Processing {ticker} ({security.name})")

        #     # Add basic company information
        #     kb.add_document(
        #         {
        #             "ticker": ticker,
        #             "text": f"Company overview: {security.name} operates in the {security.sector} sector.",
        #             "source": "company_profile",
        #             "date": f"{settings.end_year}-01-01",
        #             "section": "overview",
        #         }
        #     )
        #     document_count += 1

        #     # Process SEC filings if available - updated for DataFrame
        #     ticker_filings = (
        #         filings_df[filings_df["ticker"] == ticker]
        #         if not filings_df.empty
        #         else pd.DataFrame()
        #     )

        #     filing_count = len(ticker_filings)
        #     if filing_count > 0:
        #         progress.console.print(
        #             f"[dim]Found {filing_count} filings for {ticker}"
        #         )

        #         for _, filing in ticker_filings.iterrows():
        #             filing_date = filing["filing_date"]
        #             if isinstance(filing_date, pd.Timestamp):
        #                 filing_date = filing_date.strftime("%Y-%m-%d")
        #                 filing_year = filing_date.split("-")[0]
        #             else:
        #                 filing_year = str(filing_date).split("-")[0]

        #             filing_year = int(filing_year)

        #             # Skip filings outside the specified year range
        #             if (
        #                 filing_year < settings.start_year
        #                 or filing_year > settings.end_year
        #             ):
        #                 continue

        #             # Get filing details from DataFrame columns
        #             filing_type = "10-K"  # Our fetcher currently only gets 10-Ks
        #             accession_number = filing.get("accessionNumber", "")
        #             documents_url = filing.get("documents_url", "")

        #             # Fetch and process the filing content
        #             filing_chunks = fetch_and_process_filing(
        #                 ticker=ticker,
        #                 filing_type=filing_type,
        #                 filing_date=filing_date,
        #                 accession_number=accession_number,
        #                 documents_url=documents_url,
        #                 output_dir=settings.output_dir,
        #                 depth=settings.kb_depth,
        #                 verbose=settings.verbose,
        #                 logger=logger,
        #             )

        #             # Add the processed chunks to the knowledge base
        #             for chunk in filing_chunks:
        #                 kb.add_document(
        #                     {
        #                         "ticker": ticker,
        #                         "text": chunk["text"],
        #                         "source": f"{filing_type}_{accession_number}",
        #                         "date": filing_date,
        #                         "section": chunk.get("section", "unknown"),
        #                     }
        #                 )
        #                 document_count += 1
        #     else:
        #         progress.console.print(f"[yellow]No filings found for {ticker}")

        #     # Update progress for this ticker
        #     progress.update(kb_build_task, advance=1)

        # progress.update(main_task, advance=1)

        # # Create index once after all documents are added (if we're not using KNN)
        # indexing_task = progress.add_task("[cyan]Building vector index...", total=1)
        # if kb.index_type in ["KNN", "FLAT"]:
        #     progress.console.print(
        #         f"[green]Using exhaustive kNN search for {document_count} documents - no index needed"
        #     )
        # else:
        #     progress.console.print(
        #         f"[yellow]Creating vector index for {document_count} documents - this may take some time..."
        #     )
        #     # Restore original method and create index
        #     kb._ensure_index_exists = original_ensure_index
        #     kb._ensure_index_exists()

        # progress.update(indexing_task, advance=1)
        # progress.update(main_task, advance=1)
        # progress.console.print("[bold green]Knowledge base built successfully!")

    return kb


def query_knowledge_base(query: str, ticker: str = None, limit: int = 5):
    """
    Query an existing knowledge base.

    Args:
        query: The search query text
        ticker: Optional ticker symbol to filter results
        limit: Maximum number of results to return

    Returns:
        List of matching documents with their scores
    """
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("alphaledger.kb_query")

    # Connect to the existing knowledge base using the same URI
    kb = KnowledgeBase(
        uri=settings.kb_uri,
        embedding_model=settings.embedding_model,
        index_metric=settings.kb_index_metric,
        index_type=settings.kb_index_type,
        index_num_partitions=settings.kb_index_num_partitions,
        index_num_sub_vectors=settings.kb_index_num_sub_vectors,
        index_num_bits=settings.kb_index_num_bits,
        accelerator=settings.kb_accelerator,
    )

    logger.info(f"Connected to knowledge base at {settings.kb_uri}")

    # Perform the search
    results = kb.search(query, ticker=ticker, limit=limit)

    return results


if __name__ == "__main__":
    # Example usage
    query = "What are the revenue trends for cloud services?"
    results = query_knowledge_base(query, limit=3)

    print(f"\nResults for query: '{query}'")
    print("-" * 80)

    for i, doc in enumerate(results, 1):
        score = 1.0 - doc["_distance"]  # Convert distance to similarity score
        print(f"Result {i} (Score: {score:.2f}):")
        print(f"Ticker: {doc['ticker']}")
        print(f"Source: {doc['source']}")
        print(f"Date: {doc['date']}")
        print(f"Section: {doc['section']}")
        print(f"Text: {doc['text'][:150]}...")  # Show first 150 chars
        print("-" * 80)
