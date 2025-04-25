# AlphaLedger - Generative AI toolkit for open finance data

## Installation

Install uv [reference](https://docs.astral.sh/uv/getting-started/installation/)


## Usage

alphaledger uses pydantic settings for configuration management. There are three ways to customize the configuration:
1. set values in `.env` at the project root (copy `.env.template`) or in your shell
2. cli args - these are the 'kebab-cased' version of the env vars
3. at runtime by `from alphaledger.config import settings; settings.my_var = 'foo'`

use `from alphaledger import logger; log.info, etc.` for Rich logging to stdout 

_N.B.'s_
- `uv run alphaledger -h|--help` for help
- when running from a notebook or executing alphaledger via uv use `sys.argv = [sys.argv[0]]` to override required cli args

### Universes

Universes define collections of securities for analysis. A universe is comprised of some number of securities and an analysis period. The lists of securities can be defined in YAML or JSON format, typically within the `universes/` directory (or subdirectories). Currently they won't be assigned an analysis period until loaded (the values set in config.settings will be used).

Example structure:
- `universes/indices/sp500.yaml`
- `universes/sectors/cloud_computing.yaml`
- `universes/my_custom_stocks.json`

#### Using Universes in Code

```python
from alphaledger.universe import Universe
# Load a universe by name (finds .yaml or .json in universe_dir)
# It handles subdirectories automatically.
cloud_stocks = Universe("sectors/cloud_computing")
sp500 = Universe("indices/sp500")
```

Universes are lazy-loaded and the public APIs generally return polars lazy frames [lazy-loaded](https://docs.pola.rs/user-guide/lazy/). Creating a universe instance will just load the metadata. To collect filings `universe.collect_filings()` will:
- Check existing metadata to find missing (ticker, year) pairs. This will be the whole universe when initalizing from scratch. 
- Fetch the needed filings and associated metadata from SEC EDGAR. This step uses an HTTP cache on disk. _n.b.:_ populate the user agent value in env or set it manually with:
    `from alphaledger.config import settings; settings.sec_user_agent(<your email>)`
- XBRL facts are also lazy loaded, you can collect all numeric facts or text facts with:
    `universe.get_numeric_facts()`
- To get only facts for a single security:
    `universe.get_security_numeric_facts("AMZ")`


### Processing Financial Filings (XBRL + iXBRL)

Currenly alphaledger just handles 10k filings from SEC EDGAR. There are two modalities, structured and unstructured. Structured APIs (e.g. `process_filings_structured`) extract only the XBRL entities from an instance. Unstructured APIs are used for aligning facts within the unstructured filing document (iXBRL).


#### Working with unstructured APIs using IXBRLDocumentParser

```python
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

### Knowledge Base

The current knowledge base implementation uses [LanceDB](https://lancedb.github.io/) for local index creation/storage/retrieval. 

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


