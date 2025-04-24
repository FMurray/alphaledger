# AlphaLedger


## Installation

Install uv [reference](https://docs.astral.sh/uv/getting-started/installation/)


## Usage

_N.B._
`uv run alphaledger -h|--help` for help

### Universes

Universes define collections of securities for analysis. A universe is comprised of some number of securities and an analysis period. The lists of securities can be defined in YAML or JSON format, typically within the `universes/` directory (or subdirectories). Currently they won't be assigned an analysis period until loaded.

# Example structure:
# - `universes/indices/sp500.yaml`
# - `universes/sectors/cloud_computing.yaml`
# - `universes/my_custom_stocks.json`

#### Using Universes in Code

```python
from alphaledger.universe import load_universe
# Load a universe by name (finds .yaml or .json in universe_dir)
# It handles subdirectories automatically.
cloud_stocks = load_universe("sectors/cloud_computing")
sp500 = load_universe("indices/sp500")

# Access securities
print(f"Loaded universe: {cloud_stocks.name} with {len(cloud_stocks)} securities.")
print(f"Tickers: {cloud_stocks.get_tickers()[:5]}...") # Show first 5 tickers

msft_info = cloud_stocks.get_security("MSFT")
if msft_info:
    print(f"Info for MSFT: {msft_info.name} (Sector: {msft_info.sector})")

## Filter by sector (example)
#tech_securities = [s for s in universe.get_all_securities()
#if s.sector == "Information Technology"]
```


### Processing Financial Filings (iXBRL)

AlphaLedger includes tools to parse and process iXBRL (Inline XBRL) documents, often found in SEC filings. This allows extracting both textual content and tagged financial facts.

```python
import logging
from alphaledger.process_xbrl import IXBRLDocumentParser, ProcessingOptions, extract_us_gaap_facts
from alphaledger.formatter import MarkdownFormatter, PlainTextFormatter
# Assuming you have fetched an iXBRL file (e.g., filing.htm) and potentially parsed it with py-xbrl
# from xbrl.instance import XbrlParser
# parser = XbrlParser()
# xbrl_doc = parser.parse_instance("path/to/filing.htm") # Or raw content
# all_facts = xbrl_doc.get_facts()

# --- Placeholder for getting facts ---
# In a real scenario, you'd get 'all_facts' from parsing the document with py-xbrl
all_facts = []
instance_path = "path/to/your/downloaded/filing.htm"
encoding = "utf-8" # Or detect encoding
# --- End Placeholder ---

# 1. Initialize the Parser
# Provide the path to the iXBRL file and the list of facts extracted by py-xbrl
try:
    ixbrl_parser = IXBRLDocumentParser(instance_path=instance_path, facts=all_facts, encoding=encoding)

    # 2. Parse the document
    # This reads the HTML, finds sections, and interleaves text and fact objects
    parsed_doc = ixbrl_parser.parse()

    # 3. Choose a Formatter and Options
    # Define the level of detail needed using ProcessingOptions
    options = ProcessingOptions.LEVEL_2 # Basic info, primary + detailed financials, sections

    # Instantiate a formatter, passing the options
    md_formatter = MarkdownFormatter(options=options)
    txt_formatter = PlainTextFormatter(options=options)

    # 4. Format the Document
    # Get a string representation based on the formatter and options
    markdown_output = parsed_doc.to_string(md_formatter)
    plaintext_output = parsed_doc.to_string(txt_formatter)

    # print("--- Markdown Output ---")
    # print(markdown_output[:2000]) # Print first 2000 chars
    # print("\n--- Plain Text Output ---")
    # print(plaintext_output[:2000]) # Print first 2000 chars

except FileNotFoundError:
    logging.error(f"Filing file not found at {instance_path}")
except Exception as e:
    logging.error(f"Error processing iXBRL document: {e}")

```

**Explanation:**

1.  **`IXBRLDocumentParser`:** Takes the path to the iXBRL file (`.htm`, `.html`) and a list of `AbstractFact` objects (obtained beforehand using a library like `py-xbrl`) as input.
2.  **`parser.parse()`:** Returns an `IXBRLDocument` object, which contains a list of `DocumentSection` objects. Each section holds a list of `DocumentPart` objects (either text or XBRL facts).
3.  **`ProcessingOptions`:** Use flags like `LEVEL_1`, `LEVEL_2`, `LEVEL_3` to control the detail level. These options are passed to the formatter.
4.  **Formatters (`MarkdownFormatter`, `PlainTextFormatter`, etc.):** These classes take the `ProcessingOptions` and define how to render the `IXBRLDocument` structure into a string. They control section formatting, text formatting, and fact formatting (including how `TextFact` instances are displayed - truncated for lower detail levels, full for higher levels).
5.  **`parsed_doc.to_string(formatter)`:** The core method to generate the final string output according to the chosen formatter and options.

*(Future Work: A `FactExtractor` class will be added to specifically extract structured fact data (e.g., for databases) based on `ProcessingOptions`.)*

#### Extracting Structured Fact Data

Beyond formatted text output, the `IXBRLDocument` object provides methods to extract structured data directly into Polars DataFrames. This is useful for quantitative analysis and data storage.

```python
# Assuming 'parsed_doc' is an IXBRLDocument object obtained from IXBRLDocumentParser.parse()

# 1. Extract Numeric Facts
# Returns a Polars DataFrame containing only NumericFact instances
# Schema defined by TARGET_SCHEMA_NUMERIC_POLARS in process_xbrl.py
numeric_facts_df = parsed_doc.to_numeric_dataframe()
print("--- Numeric Facts DataFrame ---")
print(numeric_facts_df.head())

# Example: Calculate total revenue from the extracted facts
# Note: This requires the 'Revenue' concept to be present and numeric
# revenue_total = numeric_facts_df.filter(
#     pl.col("concept_name") == "Revenue"
# )["fact_value"].sum()
# print(f"\nTotal Revenue: {revenue_total}")

# 2. Extract Text Facts
# Returns a Polars DataFrame containing only TextFact instances (and other non-numeric types)
# Schema defined by TARGET_SCHEMA_TEXT_POLARS in process_xbrl.py
text_facts_df = parsed_doc.to_text_dataframe()
print("\n--- Text Facts DataFrame ---")
print(text_facts_df.head())

# Example: Find specific text information
# accounting_policies = text_facts_df.filter(
#     pl.col("concept_name").str.contains("AccountingPolicy")
# )
# print("\nAccounting Policies:")
# print(accounting_policies)

# 3. Extract All Facts (Combined) - Less common, usually prefer separated DFs
# Returns a Polars DataFrame containing all fact types, conforming to TARGET_SCHEMA_POLARS
# Numeric values are cast to float, text values remain strings.
# all_facts_df = parsed_doc.to_dataframe(format="polars")
# print("\n--- All Facts DataFrame ---")
# print(all_facts_df.head())

# Note: The underlying `process_filing_urls` function leverages these methods
# to process multiple filings and aggregate facts efficiently.
```

**Explanation:**

*   **`parsed_doc.to_numeric_dataframe()`:** Extracts all `NumericFact` objects found during parsing into a Polars DataFrame. The schema includes fields like `concept_name`, `fact_value` (as float), `unit`, `period_start`, `period_end`, etc.
*   **`parsed_doc.to_text_dataframe()`:** Extracts all non-numeric facts (like `TextFact`) into a Polars DataFrame. The schema includes `concept_name`, `fact_value` (as string), `period_start`, `period_end`, etc.
*   **Schemas:** The exact columns and data types are defined by `TARGET_SCHEMA_NUMERIC_POLARS` and `TARGET_SCHEMA_TEXT_POLARS` within `src/alphaledger/process_xbrl.py`.


### Scripts

AlphaLedger provides various scripts that can be run using uv. All scripts support a common set of arguments:

#### Common Script Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--universe` | `-u` | sectors/cloud_computing | The universe of securities to analyze |
| `--start-year` | | 2022 | Start year for analysis period |
| `--end-year` | | 2023 | End year for analysis period |
| `--output` | `-o` | output | Output directory for results |
| `--verbose` | `-v` | | Enable verbose output |

#### Running Scripts

##### Build Knowledge Base

The `build-kb` script collects and organizes information about securities in a specified universe.

Basic usage
```bash
uv run build-kb
```

With options
```bash
uv run build-kb -v --universe sectors/cloud_computing --start-year 2020 --end-year 2024
```

Build Knowledge Base Options:

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--depth` | `-d` | 1 | Depth of information to collect for knowledge base |
| `--embedding-model` | | text-embedding-3-small | OpenAI/other model to use for embeddings |



#### Build Knowledge Base

```python
from alphaledger.kb import KnowledgeBase
# Create a local knowledge base (for testing)
local_kb = KnowledgeBase(uri="local://kb_data")
# Create a remote knowledge base (for production)
remote_kb = KnowledgeBase(uri="s3://your-bucket/kb_data")
# Add documents to the knowledge base
local_kb.add_documents([
{"ticker": "MSFT", "text": "Microsoft's cloud revenue grew by 22% in Q2 2023", "date": "2023-07-15"},
{"ticker": "AMZN", "text": "Amazon Web Services reports slower growth amid cloud spending cuts", "date": "2023-08-03"}
])
# Search the knowledge base
results = local_kb.search("cloud revenue growth", limit=5)
for doc in results:
    print(f"{doc.ticker}: {doc.text} ({doc.date}) - Relevance: {doc.score}")
```


#### Logging

```python
# At the top of each module, import the logger:
from alphaledger import get_logger

# Create a module-specific logger:
logger = get_logger(__name__)

# Use the logger with rich formatting:
logger.debug("Processing [bold]data[/bold]...")
logger.info("Operation [green]successful[/green]")
logger.warning("[yellow]Warning:[/yellow] Incomplete data found")
logger.error("[bold red]Error:[/bold red] Failed to connect to database")
logger.critical("[white on red]CRITICAL:[/white on red] System shutdown required")

# For dynamic values, use f-strings or format:
user_input = "example"
logger.info(f"Processing user input: [blue]{user_input}[/blue]")

# For conditional logging (better performance):
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"Complex calculation result: {expensive_calculation()}")
```



