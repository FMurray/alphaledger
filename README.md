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