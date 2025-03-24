# AlphaLedger


## Installation

Install uv [reference](https://docs.astral.sh/uv/getting-started/installation/)


## Usage

## Usage

### Universes

```python
from alphaledger.universe import load_universe
# Load a universe
universe = load_universe("cloud_computing")
# Access securities
for ticker in universe.get_tickers():
    security = universe.get_security(ticker)
    print(f"{ticker}: {security.name} ({security.sector})")
# Filter by sector
tech_securities = [s for s in universe.get_all_securities()
if s.sector == "Information Technology"]
```

Universes define collections of securities for analysis. They are stored in YAML or JSON format in the `universes/` directory:

- `universes/indices/` - Major market indices (S&P 500, NASDAQ, etc.)
- `universes/sectors/` - Sector-based collections (Technology, Healthcare, etc.)
- `universes/custom/` - Your custom universes

#### Using Universes in Code


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




