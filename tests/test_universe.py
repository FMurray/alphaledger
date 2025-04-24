import pytest
from pathlib import Path
import yaml
import json
import polars as pl
from polars.testing import assert_frame_equal
from unittest.mock import patch, MagicMock
import datetime
from pydantic import ValidationError
import dotenv  # Import dotenv for patching

from alphaledger.universe import (
    Security,
    Universe,
    load_from_yaml,
    load_from_json,
    load_universe,
    get_universe_path,
    save_to_yaml,
    save_to_json,
    YearRange,
    EARLIEST_YEAR_PLACEHOLDER,
    DEFAULT_YEAR_RANGE,
)
from alphaledger.config import settings

# --- Constants for Testing ---
CURRENT_YEAR = datetime.datetime.now().year

# --- Fixtures ---


@pytest.fixture(scope="function")
def mock_settings(tmp_path, monkeypatch):
    """Fixture to mock settings using a temporary directory and prevent .env loading."""
    # Prevent dotenv from loading any .env file during tests
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)

    # Define test-specific paths
    test_universe_dir = tmp_path / "universes"
    test_output_dir = tmp_path / "output"

    # Create the directories
    test_universe_dir.mkdir(parents=True, exist_ok=True)
    test_output_dir.mkdir(parents=True, exist_ok=True)

    # Monkeypatch the settings attributes *before* they are potentially used by test setup
    # Note: We patch the *module's* settings object
    monkeypatch.setattr("alphaledger.config.settings.universe_dir", test_universe_dir)
    monkeypatch.setattr("alphaledger.config.settings.output_dir", test_output_dir)

    # Yield the tmp_path or the settings object if needed by tests, though patching is usually sufficient
    # Yielding settings might still give the originally loaded object, patching is safer.
    yield tmp_path  # Or yield settings if tests explicitly need the object

    # Monkeypatch automatically handles teardown/restoration


@pytest.fixture
def sample_securities_data():
    return [
        {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "exchange": "NASDAQ",
            "sector": "Information Technology",
        },
        {
            "ticker": "MSFT",
            "name": "Microsoft Corp.",
            "exchange": "NASDAQ",
            "sector": "Information Technology",
            "industry": "Software",
        },
        {
            "ticker": "GOOG",
            "name": "Alphabet Inc.",
            "exchange": "NASDAQ",
        },  # Missing sector/industry
    ]


@pytest.fixture
def sample_universe_data(sample_securities_data):
    return {"name": "tech_giants", "securities": sample_securities_data}


@pytest.fixture
def sample_universe_yaml_file(mock_settings, sample_universe_data):
    """Creates a sample YAML universe file in the mocked universe_dir."""
    path = mock_settings / "universes" / "tech_giants.yaml"
    with open(path, "w") as f:
        yaml.dump(sample_universe_data, f, default_flow_style=False)
    return path


@pytest.fixture
def sample_universe_json_file(mock_settings, sample_universe_data):
    """Creates a sample JSON universe file in the mocked universe_dir."""
    path = mock_settings / "universes" / "tech_giants.json"
    with open(path, "w") as f:
        json.dump(sample_universe_data, f, indent=2)
    return path


@pytest.fixture
def sample_subfolder_universe_yaml_file(mock_settings, sample_universe_data):
    """Creates a sample YAML universe file in a subfolder of the mocked universe_dir."""
    subfolder = mock_settings / "universes" / "sectors"
    subfolder.mkdir(exist_ok=True)
    path = subfolder / "cloud_computing.yaml"
    # Modify data slightly to avoid name collision if needed, but keep structure
    cloud_data = {
        "name": "sectors/cloud_computing",  # Use raw name for loading identification
        "securities": [
            {"ticker": "MSFT", "name": "Microsoft Corp.", "exchange": "NASDAQ"},
            {"ticker": "AMZN", "name": "Amazon.com, Inc.", "exchange": "NASDAQ"},
        ],
    }
    with open(path, "w") as f:
        yaml.dump(cloud_data, f, default_flow_style=False)
    return path


# --- Test Classes ---


class TestYearRange:
    def test_defaults(self):
        yr = YearRange()
        assert yr.start == CURRENT_YEAR - DEFAULT_YEAR_RANGE
        assert yr.end == CURRENT_YEAR

    def test_specific_int_years(self):
        yr = YearRange(start=2019, end=2022)
        assert yr.start == 2019
        assert yr.end == 2022
        assert yr.get_filing_years() == [2019, 2020, 2021, 2022]

    def test_earliest_latest(self):
        yr = YearRange(start="earliest", end="latest")
        assert yr.start == "earliest"
        assert yr.end == "latest"
        assert yr.get_filing_years() == list(
            range(EARLIEST_YEAR_PLACEHOLDER, CURRENT_YEAR + 1)
        )

    def test_earliest_to_specific(self):
        yr = YearRange(start="earliest", end=2020)
        assert yr.start == "earliest"
        assert yr.end == 2020
        assert yr.get_filing_years() == list(range(EARLIEST_YEAR_PLACEHOLDER, 2021))

    def test_specific_to_latest(self):
        yr = YearRange(start=2021, end="latest")
        assert yr.start == 2021
        assert yr.end == "latest"
        assert yr.get_filing_years() == list(range(2021, CURRENT_YEAR + 1))

    def test_start_year_only_int(self):
        yr = YearRange(start=2022)
        assert yr.start == 2022
        assert yr.end == CURRENT_YEAR  # Defaults
        assert yr.get_filing_years() == list(range(2022, CURRENT_YEAR + 1))

    def test_end_year_only_int(self):
        yr = YearRange(end=2021)
        assert yr.start == CURRENT_YEAR - DEFAULT_YEAR_RANGE  # Defaults
        assert yr.end == 2021
        assert yr.get_filing_years() == list(
            range(CURRENT_YEAR - DEFAULT_YEAR_RANGE, 2022)
        )

    def test_start_year_only_earliest(self):
        yr = YearRange(start="earliest")
        assert yr.start == "earliest"
        assert yr.end == CURRENT_YEAR  # Defaults
        assert yr.get_filing_years() == list(
            range(EARLIEST_YEAR_PLACEHOLDER, CURRENT_YEAR + 1)
        )

    def test_end_year_only_latest(self):
        yr = YearRange(end="latest")
        assert yr.start == CURRENT_YEAR - DEFAULT_YEAR_RANGE  # Defaults
        assert yr.end == "latest"
        assert yr.get_filing_years() == list(
            range(CURRENT_YEAR - DEFAULT_YEAR_RANGE, CURRENT_YEAR + 1)
        )

    def test_invalid_range_start_gt_end_int(self):
        # Validator should adjust end = start
        yr = YearRange(start=2023, end=2020)
        assert yr.start == 2023
        assert yr.end == 2023  # Adjusted
        assert yr.get_filing_years() == [2023]

    def test_invalid_range_resolved_start_gt_resolved_end(self):
        # e.g., start=2025, end='latest' (assuming current year < 2025)
        if CURRENT_YEAR < 2025:
            yr = YearRange(start=2025, end="latest")
            assert yr.start == 2025
            assert yr.end == "latest"
            assert yr.get_filing_years() == []  # Resolved start > resolved end
        else:
            pytest.skip("Current year is not less than 2025, skipping test")

    def test_invalid_type(self):
        with pytest.raises(ValidationError):
            YearRange(start="not_a_year")
        with pytest.raises(ValidationError):
            YearRange(end=2020.5)

    def test_str_representation(self):
        assert (
            str(YearRange()) == f"[{CURRENT_YEAR - DEFAULT_YEAR_RANGE}-{CURRENT_YEAR}]"
        )
        assert str(YearRange(start=2019, end=2022)) == "[2019-2022]"
        assert str(YearRange(start="earliest", end="latest")) == "[earliest-latest]"
        assert str(YearRange(start=2021)) == f"[2021-{CURRENT_YEAR}]"
        assert str(YearRange(end=2020)) == f"[{CURRENT_YEAR - DEFAULT_YEAR_RANGE}-2020]"


class TestSecurity:
    def test_security_initialization(self):
        sec = Security(
            ticker="TEST",
            name="Test Inc.",
            exchange="NYSE",
            sector="Test Sector",
            industry="Test Industry",
            currency="USD",
            country="US",
            custom_sector="Custom Test Sector",
            custom_industry="Custom Test Industry",
            subsector="Test Subsector",
            theme=["Test Theme 1", "Test Theme 2"],
        )
        assert sec.ticker == "TEST"
        assert sec.name == "Test Inc."
        assert sec.exchange == "NYSE"
        assert sec.sector == "Test Sector"
        assert sec.industry == "Test Industry"
        assert sec.currency == "USD"
        assert sec.country == "US"
        assert sec.custom_sector == "Custom Test Sector"
        assert sec.custom_industry == "Custom Test Industry"
        assert sec.subsector == "Test Subsector"
        assert sec.theme == ["Test Theme 1", "Test Theme 2"]

    def test_security_minimal_initialization(self):
        sec = Security(ticker="MIN", name="Minimal Corp.", exchange="LSE")
        assert sec.ticker == "MIN"
        assert sec.name == "Minimal Corp."
        assert sec.exchange == "LSE"
        assert sec.sector is None
        assert sec.industry is None
        assert sec.currency == "USD"  # Default
        assert sec.country == "US"  # Default
        assert sec.custom_sector is None
        assert sec.custom_industry is None
        assert sec.subsector is None
        assert sec.theme is None

    def test_security_str(self):
        sec = Security(ticker="STR", name="String Test", exchange="NASDAQ")
        assert str(sec) == "STR - String Test (NASDAQ)"


class TestUniverseLoading:
    def test_load_from_yaml(self, sample_universe_yaml_file, sample_universe_data):
        universe = load_from_yaml(str(sample_universe_yaml_file), load_filings=False)
        assert universe.name == sample_universe_data["name"]
        assert len(universe) == len(sample_universe_data["securities"])
        assert set(universe.get_tickers()) == {"AAPL", "MSFT", "GOOG"}
        msft = universe.get_security("MSFT")
        assert msft is not None
        assert msft.name == "Microsoft Corp."
        assert msft.sector == "Information Technology"
        assert msft.industry == "Software"
        goog = universe.get_security("GOOG")
        assert goog is not None
        assert goog.sector is None  # Check handling of missing optional fields
        assert goog.industry is None

    def test_load_from_json(self, sample_universe_json_file, sample_universe_data):
        universe = load_from_json(str(sample_universe_json_file), load_filings=False)
        # Adjust expected name as filename is different
        assert (
            universe.name == sample_universe_data["name"]
        )  # Use name from data in file
        # The JSON file contains the same securities data
        assert len(universe) == len(sample_universe_data["securities"])
        assert set(universe.get_tickers()) == {"AAPL", "MSFT", "GOOG"}
        aapl = universe.get_security("AAPL")
        assert aapl is not None
        assert aapl.name == "Apple Inc."
        assert aapl.sector == "Information Technology"

    def test_load_from_yaml_file_not_found(self, mock_settings):
        universe_dir = mock_settings / "universes"
        with pytest.raises(FileNotFoundError):
            load_from_yaml(str(universe_dir / "non_existent.yaml"), load_filings=False)

    def test_load_from_json_file_not_found(self, mock_settings):
        universe_dir = mock_settings / "universes"
        with pytest.raises(FileNotFoundError):
            load_from_json(str(universe_dir / "non_existent.json"), load_filings=False)

    def test_get_universe_path_direct_yaml(
        self, mock_settings, sample_universe_yaml_file
    ):
        path = get_universe_path("tech_giants")
        assert path == sample_universe_yaml_file
        path_ext = get_universe_path("tech_giants.yaml")
        assert path_ext == sample_universe_yaml_file

    def test_get_universe_path_direct_json(
        self, mock_settings, sample_universe_json_file
    ):
        path = get_universe_path("tech_giants")
        assert path == sample_universe_json_file
        path_ext = get_universe_path("tech_giants.json")
        assert path_ext == sample_universe_json_file

    def test_get_universe_path_subfolder(
        self, mock_settings, sample_subfolder_universe_yaml_file
    ):
        path = get_universe_path("sectors/cloud_computing")
        assert path == sample_subfolder_universe_yaml_file
        path_ext = get_universe_path("sectors/cloud_computing.yaml")
        assert path_ext == sample_subfolder_universe_yaml_file

    def test_get_universe_path_not_found(self, mock_settings):
        with pytest.raises(FileNotFoundError):
            get_universe_path("non_existent_universe")

    def test_load_universe_yaml(
        self, mock_settings, sample_universe_yaml_file, sample_universe_data
    ):
        universe = load_universe("tech_giants", load_filings=False)
        assert universe.name == sample_universe_data["name"]
        assert len(universe) == len(sample_universe_data["securities"])

    def test_load_universe_json(
        self, mock_settings, sample_universe_json_file, sample_universe_data
    ):
        # Assumes tech_giants.yaml doesn't exist for this test case or json is preferred somehow
        yaml_path = mock_settings / "universes" / "tech_giants.yaml"
        if yaml_path.exists():
            yaml_path.unlink()

        # Load json by name, assuming tech_giants.json exists
        json_path = mock_settings / "universes" / "tech_giants.json"
        assert json_path.exists(), "Test assumes tech_giants.json exists for this case"

        universe = load_universe("tech_giants", load_filings=False)
        assert universe.name == "tech_giants"  # Name derived from file stem
        assert len(universe) == len(sample_universe_data["securities"])

    def test_load_universe_subfolder(
        self, mock_settings, sample_subfolder_universe_yaml_file
    ):
        universe = load_universe("sectors/cloud_computing", load_filings=False)
        assert (
            universe.raw_name == "sectors/cloud_computing"
        )  # Check raw name is preserved
        assert universe.name == "sectors_cloud_computing"  # Check normalized name
        assert set(universe.get_tickers()) == {"MSFT", "AMZN"}
        assert universe.filings_df is None
        # Check default years are set in year_range
        assert isinstance(universe.year_range, YearRange)
        assert universe.year_range.start == CURRENT_YEAR - DEFAULT_YEAR_RANGE
        assert universe.year_range.end == CURRENT_YEAR

    def test_load_universe_not_found(self, mock_settings):
        with pytest.raises(FileNotFoundError):
            load_universe("completely_made_up", load_filings=False)

    # Test load_filings=True requires mocking EDGARFetcher, add later


class TestUniverseClass:
    @pytest.fixture
    def basic_universe(self):
        """Creates a basic Universe object without loading from file."""
        u = Universe(name="basic")
        u.add_security(Security(ticker="SEC1", name="Security One", exchange="NYSE"))
        u.add_security(
            Security(
                ticker="SEC2", name="Security Two", exchange="LSE", sector="Finance"
            )
        )
        return u

    def test_universe_initialization(self):
        u = Universe(name="My Universe")
        assert u.name == "My_Universe"  # Normalized
        assert u.raw_name == "My Universe"
        assert len(u) == 0
        assert u.securities == {}
        assert u.filings_df is None
        # Check default years from settings (assuming defaults)
        # assert u.start_year == settings.start_year # Can be None
        # assert u.end_year == settings.end_year # Can be None

    def test_universe_initialization_with_slashes(self):
        u = Universe(name="category/sub_category")
        assert u.name == "category_sub_category"  # Normalized
        assert u.raw_name == "category/sub_category"

    def test_empty_universe_methods(self):
        """Test methods on a universe with no securities added."""
        u_empty = Universe(name="empty_test")
        assert u_empty.get_all_securities() == []
        assert u_empty.get_tickers() == []
        assert len(u_empty) == 0
        assert (
            str(u_empty)
            == f"empty_test [{CURRENT_YEAR - DEFAULT_YEAR_RANGE}-{CURRENT_YEAR}] (0 securities) (Filings not loaded)"
        )

    def test_add_remove_get_security(self, basic_universe):
        assert len(basic_universe) == 2
        assert set(basic_universe.get_tickers()) == {"SEC1", "SEC2"}

        sec1 = basic_universe.get_security("SEC1")
        assert sec1 is not None
        assert sec1.name == "Security One"

        sec3 = Security(ticker="SEC3", name="Security Three", exchange="TSE")
        basic_universe.add_security(sec3)
        assert len(basic_universe) == 3
        assert basic_universe.get_security("SEC3") == sec3

        basic_universe.remove_security("SEC1")
        assert len(basic_universe) == 2
        assert basic_universe.get_security("SEC1") is None
        assert set(basic_universe.get_tickers()) == {"SEC2", "SEC3"}

        # Test removing non-existent ticker (should not raise error)
        basic_universe.remove_security("NONEXISTENT")
        assert len(basic_universe) == 2

    def test_get_all_securities(self, basic_universe):
        all_secs = basic_universe.get_all_securities()
        assert len(all_secs) == 2
        assert isinstance(all_secs[0], Security)
        assert {s.ticker for s in all_secs} == {"SEC1", "SEC2"}

    def test_get_filing_years(self):
        # Test various scenarios via the YearRange model integration
        # Defaults
        u_default = Universe(name="default_years")
        assert u_default.get_filing_years() == list(
            range(CURRENT_YEAR - DEFAULT_YEAR_RANGE, CURRENT_YEAR + 1)
        )

        # Specific Ints
        u_specific = Universe(name="specific", start_year=2018, end_year=2020)
        assert u_specific.get_filing_years() == [2018, 2019, 2020]

        # Earliest to Latest
        u_full = Universe(name="full", start_year="earliest", end_year="latest")
        assert u_full.get_filing_years() == list(
            range(EARLIEST_YEAR_PLACEHOLDER, CURRENT_YEAR + 1)
        )

        # Specific to Latest
        u_to_latest = Universe(name="to_latest", start_year=2021, end_year="latest")
        assert u_to_latest.get_filing_years() == list(range(2021, CURRENT_YEAR + 1))

        # Earliest to Specific
        u_from_earliest = Universe(
            name="from_earliest", start_year="earliest", end_year=2019
        )
        assert u_from_earliest.get_filing_years() == list(
            range(EARLIEST_YEAR_PLACEHOLDER, 2020)
        )

        # Invalid range (Start > End Int) -> adjusted
        u_invalid_int = Universe(name="invalid_int", start_year=2023, end_year=2020)
        assert u_invalid_int.get_filing_years() == [2023]

        # Invalid range (Resolved Start > Resolved End) -> empty list
        if CURRENT_YEAR < 2025:
            u_invalid_resolved = Universe(
                name="invalid_resolved", start_year=2025, end_year="latest"
            )
            assert u_invalid_resolved.get_filing_years() == []
        else:
            print("Skipping resolved invalid range test as CURRENT_YEAR >= 2025")

    def test_universe_str(self, basic_universe):
        # Default range
        u_default = Universe(name="str_test_default")
        assert (
            str(u_default)
            == f"str_test_default [{CURRENT_YEAR - DEFAULT_YEAR_RANGE}-{CURRENT_YEAR}] (0 securities) (Filings not loaded)"
        )

        # Specific range
        basic_universe.year_range = YearRange(start=2021, end=2023)
        assert (
            str(basic_universe)
            == "basic [2021-2023] (2 securities) (Filings not loaded)"
        )

        # With filings loaded (mock)
        basic_universe.filings_df = pl.DataFrame({"ticker": ["SEC1"], "year": [2022]})
        assert (
            str(basic_universe) == "basic [2021-2023] (2 securities) (Filings loaded)"
        )

        # Earliest to latest range
        basic_universe.year_range = YearRange(start="earliest", end="latest")
        assert (
            str(basic_universe)
            == "basic [earliest-latest] (2 securities) (Filings loaded)"
        )

        # Range needing defaults
        u_part_default = Universe(name="part_def", start_year=2022)
        assert (
            str(u_part_default)
            == f"part_def [2022-{CURRENT_YEAR}] (0 securities) (Filings not loaded)"
        )

    def test_save_load_yaml_roundtrip(self, basic_universe, mock_settings):
        save_path = mock_settings / "universes" / "roundtrip.yaml"
        save_to_yaml(basic_universe, str(save_path))
        assert save_path.exists()

        loaded_universe = load_universe("roundtrip", load_filings=False)
        assert loaded_universe.name == basic_universe.name
        assert len(loaded_universe) == len(basic_universe)
        assert set(loaded_universe.get_tickers()) == set(basic_universe.get_tickers())
        sec2_original = basic_universe.get_security("SEC2")
        sec2_loaded = loaded_universe.get_security("SEC2")
        assert sec2_loaded is not None
        assert sec2_loaded.name == sec2_original.name
        assert sec2_loaded.sector == sec2_original.sector

    def test_save_load_json_roundtrip(self, basic_universe, mock_settings):
        save_path = mock_settings / "universes" / "roundtrip.json"
        save_to_json(basic_universe, str(save_path))
        assert save_path.exists()

        # Ensure yaml version doesn't exist to force json loading
        yaml_save_path = mock_settings / "universes" / "roundtrip.yaml"
        if yaml_save_path.exists():
            yaml_save_path.unlink()

        loaded_universe = load_universe("roundtrip", load_filings=False)
        assert loaded_universe.name == basic_universe.name
        assert len(loaded_universe) == len(basic_universe)
        assert set(loaded_universe.get_tickers()) == set(basic_universe.get_tickers())
        sec1_original = basic_universe.get_security("SEC1")
        sec1_loaded = loaded_universe.get_security("SEC1")
        assert sec1_loaded is not None
        assert sec1_loaded.name == sec1_original.name
        assert sec1_loaded.exchange == sec1_original.exchange


# TODO: Add TestUniverseFilings class with mocked EDGARFetcher etc.
# - Test _get_filings_path
# - Test fetch_or_load_filings logic (disk load, fetch, force fetch, errors)
# - Test get_missing_filings
# - Test sync_filings

# --- Test Classes ---


@pytest.mark.skip(reason="Filings tests require mocking EDGARFetcher")
class TestUniverseFilings:
    # Add tests for TestUniverseFilings class
    pass
