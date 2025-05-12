"""
Module for processing XBRL financial filings.
"""

import re
from typing import Dict, Any, List, Optional, Set, Union, cast, Tuple
from enum import Flag, auto, Enum
from xbrl.instance import XbrlInstance, AbstractFact, NumericFact, TextFact
from xbrl.cache import HttpCache
from alphaledger.config import settings
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup, Tag, NavigableString
import polars as pl
from pathlib import Path

# Use TYPE_CHECKING to avoid circular import error at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .formatter import DocumentFormatter  # Import for type hinting


from alphaledger import get_logger
from alphaledger.sec import EDGARFetcher

logger = get_logger(__name__)

# Define the target schema outside the method for clarity and reuse
TARGET_SCHEMA_POLARS = {
    "concept_name": pl.Utf8,
    "concept_namespace": pl.Utf8,
    "fact_value": pl.Float64,  # Target type after casting
    "fact_type": pl.Utf8,
    "section_name": pl.Utf8,
    "unit": pl.Utf8,  # Added based on _fact_to_row logic
    "period_instant": pl.Date,  # Added based on _fact_to_row logic
    "period_start": pl.Date,  # Added based on _fact_to_row logic
    "period_end": pl.Date,  # Added based on _fact_to_row logic
    "context_id": pl.Utf8,  # Added based on _fact_to_row logic
    "context_entity": pl.Utf8,  # Added based on _fact_to_row logic
    "context_scenario": pl.Utf8,  # Added based on _fact_to_row logic
    "metadata": pl.Struct(
        [
            # Use Int64 for numeric metadata to avoid potential overflow
            pl.Field("decimals", pl.Int64),
            pl.Field("precision", pl.Int64),
        ]
    ),  # Added based on _fact_to_row logic
}

# Define common columns first
COMMON_COLUMNS = {
    "concept_name": pl.Utf8,
    "concept_namespace": pl.Utf8,
    "fact_type": pl.Utf8,  # Keep track of original type
    "section_name": pl.Utf8,
    "period_instant": pl.Date,
    "period_start": pl.Date,
    "period_end": pl.Date,
    "context_id": pl.Utf8,
    "context_entity": pl.Utf8,
    "context_scenario": pl.Utf8,
    # Add filing dates
    "filing_date": pl.Date,
    "report_date": pl.Date,
}

TARGET_SCHEMA_NUMERIC_POLARS = {
    **COMMON_COLUMNS,
    "fact_value": pl.Float64,
    "unit": pl.Utf8,
    "metadata": pl.Struct(
        [
            pl.Field("decimals", pl.Int64),
            pl.Field("precision", pl.Int64),
        ]
    ),
}

TARGET_SCHEMA_TEXT_POLARS = {
    **COMMON_COLUMNS,
    "fact_value": pl.Utf8,
    # Text facts usually don't have units or numeric metadata
}

# --- Schemas for Direct Extraction (without section_name) ---
COMMON_COLUMNS_DIRECT = {k: v for k, v in COMMON_COLUMNS.items() if k != "section_name"}

TARGET_SCHEMA_NUMERIC_DIRECT_POLARS = {
    **COMMON_COLUMNS_DIRECT,
    "fact_value": pl.Float64,
    "unit": pl.Utf8,
    "metadata": pl.Struct(
        [
            pl.Field("decimals", pl.Int64),
            pl.Field("precision", pl.Int64),
        ]
    ),
}

TARGET_SCHEMA_TEXT_DIRECT_POLARS = {
    **COMMON_COLUMNS_DIRECT,
    "fact_value": pl.Utf8,
}


class ProcessingOptions(Flag):
    """Flags for controlling the level of detail in XBRL processing."""

    NONE = 0
    BASIC_INFO = auto()  # Basic filing information
    PRIMARY_FINANCIALS = auto()  # Key financial metrics (revenue, net income, etc.)
    DETAILED_FINANCIALS = auto()  # More detailed financial metrics
    SECTION_EXTRACTION = (
        auto()
    )  # Extract major filing sections (MD&A, Risk Factors, etc.)
    FOOTNOTES = auto()  # Extract and process footnotes
    SEGMENT_INFO = auto()  # Extract segment information

    # Common combinations (similar to the previous depth levels)
    LEVEL_1 = BASIC_INFO | PRIMARY_FINANCIALS
    LEVEL_2 = LEVEL_1 | DETAILED_FINANCIALS | SECTION_EXTRACTION
    LEVEL_3 = LEVEL_2 | FOOTNOTES | SEGMENT_INFO


class PartType(Enum):
    TEXT = "text"
    FACT = "fact"


@dataclass
class DocumentPart:
    type: PartType
    content: Union[str, AbstractFact]

    def is_text(self) -> bool:
        return self.type == PartType.TEXT

    def is_fact(self) -> bool:
        return self.type == PartType.FACT


@dataclass
class DocumentSection:
    title: str
    parts: List[DocumentPart]


@dataclass
class IXBRLDocument:
    sections: List[DocumentSection]

    def to_string(self, formatter: "DocumentFormatter") -> str:
        """Formats the document content into a string using the provided formatter."""
        return formatter.format_document(self)

    def to_dataframe(
        self, format: str = "polars", spark_session: Optional[any] = None
    ) -> Union[pl.DataFrame, any]:
        """
        Converts the document's facts into a structured DataFrame with a guaranteed schema.

        Args:
            format: Either "polars" or "spark" to specify the output format

        Returns:
            A DataFrame containing all facts from the document, conforming to TARGET_SCHEMA_POLARS.
        """
        rows = []
        target_keys = list(TARGET_SCHEMA_POLARS.keys())  # Get keys for _fact_to_row
        for section in self.sections:
            for part in section.parts:
                if part.type == PartType.FACT:
                    fact = cast(AbstractFact, part.content)
                    # Pass target keys to ensure consistent dict structure
                    row = self._fact_to_row(fact, section.title)
                    rows.append(row)

        if format == "polars":
            # Create DataFrame directly with the target schema.
            # This ensures all columns exist from the start, even if rows is empty.
            # Polars will handle casting from the dictionary values to schema types.
            # strict=False allows incompatible types (like string -> float error) to become null.
            try:
                df = pl.DataFrame(rows, schema=TARGET_SCHEMA_POLARS, strict=False)

                # Optional: Re-cast specific columns if initial cast might be ambiguous
                # The schema argument should handle this, but being explicit can help debugging.
                # Example: df = df.with_columns(pl.col("fact_value").cast(pl.Float64, strict=False))

            except Exception as e:
                # Log error if DataFrame creation fails even with schema
                logger.error(
                    f"Failed to create DataFrame with schema, rows may be incompatible: {e}"
                )
                # Return an empty DataFrame matching the schema as a fallback
                df = pl.DataFrame(schema=TARGET_SCHEMA_POLARS)

            return df

        elif format == "spark":
            if spark_session is None:
                raise ValueError("spark_session is required for Spark DataFrame output")
            # Define Spark schema based on TARGET_SCHEMA_POLARS if needed for consistency
            # ... Spark schema definition ...
            if not rows:
                # Return empty Spark DataFrame with schema
                # return spark_session.createDataFrame([], schema=spark_schema)
                return spark_session.createDataFrame(rows)  # Let Spark handle empty

            # Create Spark DataFrame (potentially with schema)
            spark_df = spark_session.createDataFrame(rows)  # , schema=spark_schema)
            from pyspark.sql import functions as F

            spark_df = spark_df.withColumn(
                "fact_value", F.col("fact_value").cast("double")
            )
            # Ensure other columns match target Spark schema if defined
            return spark_df
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _fact_to_row(self, fact: AbstractFact, section_title: str) -> Dict[str, Any]:
        """Converts any fact to a dictionary row with common fields."""
        # Add INFO log here
        logger.info(f"[_fact_to_row] Processing fact: {fact.concept.name}")
        row_base = {}
        # Basic Info
        row_base["concept_name"] = fact.concept.name
        if fact.concept.schema_url:
            row_base["concept_namespace"] = fact.concept.schema_url.split("/")[-2]
        else:
            row_base["concept_namespace"] = None
        row_base["fact_type"] = type(fact).__name__
        row_base["section_name"] = section_title

        # Period Info
        period = getattr(fact, "period", None)
        # -- DEBUGGING --
        if period:
            logger.debug(
                f"Fact {fact.concept.name}: Period found - Instant: {getattr(period, 'instant', None)} (Type: {type(getattr(period, 'instant', None))}), Start: {getattr(period, 'start_date', None)} (Type: {type(getattr(period, 'start_date', None))}), End: {getattr(period, 'end_date', None)} (Type: {type(getattr(period, 'end_date', None))})"
            )
        else:
            logger.debug(f"Fact {fact.concept.name}: No period object found on fact.")
        # ---------------
        if period:
            row_base["period_instant"] = getattr(period, "instant", None)
            row_base["period_start"] = getattr(period, "start_date", None)
            row_base["period_end"] = getattr(period, "end_date", None)

        # Context Info
        context = getattr(fact, "context", None)
        if context:
            row_base["context_id"] = getattr(context, "id", None)
            entity = getattr(context, "entity", None)
            row_base["context_entity"] = str(entity) if entity else None
            scenario = getattr(context, "scenario", None)
            row_base["context_scenario"] = str(scenario) if scenario else None

        return row_base

    def _numeric_fact_to_row(
        self,
        fact: NumericFact,
        section_title: str,
        filing_date: Optional[Any] = None,
        report_date: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Converts a NumericFact to a dictionary row conforming to numeric schema."""
        # 1. Initialize row with None for all target keys
        numeric_target_keys = list(TARGET_SCHEMA_NUMERIC_POLARS.keys())
        row = {key: None for key in numeric_target_keys}

        # 2. Get common fields from base method
        common_fields = self._fact_to_row(fact, section_title)

        # 3. Update row with common fields (only those present in numeric schema)
        for key, value in common_fields.items():
            if key in row:
                row[key] = value

        # Add filing dates if provided
        if "filing_date" in row:
            row["filing_date"] = filing_date
        if "report_date" in row:
            row["report_date"] = report_date

        # 4. Populate Numeric Specific Fields
        try:
            row["fact_value"] = float(fact.value) if fact.value is not None else None
        except (ValueError, TypeError):
            row["fact_value"] = None  # Set to null if conversion fails

        unit = getattr(fact, "unit", None)
        if "unit" in row:
            row["unit"] = str(unit) if unit else None

        if "metadata" in row:
            metadata_dict = {"decimals": None, "precision": None}
            decimals = getattr(fact, "decimals", None)
            if decimals is not None:
                try:
                    metadata_dict["decimals"] = int(decimals)
                except:
                    pass
            precision = getattr(fact, "precision", None)
            if precision is not None:
                try:
                    metadata_dict["precision"] = int(precision)
                except:
                    pass
            row["metadata"] = metadata_dict

        return row

    def _text_fact_to_row(
        self,
        fact: TextFact,
        section_title: str,
        filing_date: Optional[Any] = None,
        report_date: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Converts a NonNumericFact to a dictionary row conforming to text schema."""
        # 1. Initialize row with None for all target keys
        text_target_keys = list(TARGET_SCHEMA_TEXT_POLARS.keys())
        row = {key: None for key in text_target_keys}

        # 2. Get common fields from base method
        common_fields = self._fact_to_row(fact, section_title)

        # 3. Update row with common fields (only those present in text schema)
        for key, value in common_fields.items():
            if key in row:
                row[key] = value

        # Add filing dates if provided
        if "filing_date" in row:
            row["filing_date"] = filing_date
        if "report_date" in row:
            row["report_date"] = report_date

        # 4. Populate Text Specific Fields
        if "fact_value" in row:
            row["fact_value"] = str(fact.value) if fact.value is not None else None

        return row

    def to_numeric_dataframe(
        self, filing_date: Optional[Any] = None, report_date: Optional[Any] = None
    ) -> pl.DataFrame:
        """Creates a DataFrame of only the NumericFacts."""
        rows = []
        numeric_schema = {**TARGET_SCHEMA_NUMERIC_POLARS, "ticker": pl.Utf8}

        for section in self.sections:
            for part in section.parts:
                if part.type == PartType.FACT and isinstance(part.content, NumericFact):
                    fact = cast(NumericFact, part.content)
                    row = self._numeric_fact_to_row(
                        fact, section.title, filing_date, report_date
                    )
                    rows.append(row)

        # Create with the specific schema (including ticker)
        try:
            # We add ticker later in process_filing_urls, schema here should match _row output
            df = pl.DataFrame(rows, schema=TARGET_SCHEMA_NUMERIC_POLARS, strict=False)
        except Exception as e:
            logger.error(f"Failed to create numeric DataFrame: {e}")
            df = pl.DataFrame(schema=TARGET_SCHEMA_NUMERIC_POLARS)  # Empty fallback
        return df

    def to_text_dataframe(
        self, filing_date: Optional[Any] = None, report_date: Optional[Any] = None
    ) -> pl.DataFrame:
        """Creates a DataFrame of only the TextFacts."""
        rows = []
        text_schema = {**TARGET_SCHEMA_TEXT_POLARS, "ticker": pl.Utf8}

        for section in self.sections:
            for part in section.parts:
                if part.type == PartType.FACT and not isinstance(
                    part.content, NumericFact
                ):
                    fact = cast(TextFact, part.content)
                    row = self._text_fact_to_row(
                        fact, section.title, filing_date, report_date
                    )
                    rows.append(row)
        # Create with the specific schema (including ticker)
        try:
            # We add ticker later in process_filing_urls, schema here should match _row output
            df = pl.DataFrame(rows, schema=TARGET_SCHEMA_TEXT_POLARS, strict=False)
        except Exception as e:
            logger.error(f"Failed to create text DataFrame: {e}")
            df = pl.DataFrame(schema=TARGET_SCHEMA_TEXT_POLARS)  # Empty fallback
        return df


class IXBRLDocumentParser:
    def __init__(self, cache: "HttpCache"):
        self.cache = cache
        # self.facts_by_id: Dict[str, AbstractFact] = {
        #     fact.xml_id: fact for fact in facts if fact.xml_id
        # }

    def parse(self, xbrl_instance: XbrlInstance) -> IXBRLDocument:
        # Read the HTML/iXBRL file
        try:
            instance_path = self.cache.url_to_path(xbrl_instance.instance_url)
            self.cache
            if not Path(instance_path).exists():
                raise FileNotFoundError(
                    f"Cached file not found at {instance_path} for URL {xbrl_instance.instance_url}"
                )
        except Exception as e:
            logger.error(
                f"Failed to get cache path for {xbrl_instance.instance_url}: {e}"
            )
            return IXBRLDocument(sections=[])

        logger.info(
            f"[IXBRLDocumentParser.parse] Calling extract_us_gaap_facts for {xbrl_instance.instance_url}"
        )
        # facts = extract_us_gaap_facts(xbrl_instance)
        # Keep ALL facts from the instance, not just US-GAAP
        facts = xbrl_instance.facts
        logger.info(
            f"[IXBRLDocumentParser.parse] Using {len(facts)} total facts from instance."
        )
        self.facts_by_id: Dict[str, AbstractFact] = {
            fact.xml_id: fact for fact in facts if fact.xml_id
        }
        # --- DEBUG: Log created fact IDs ---
        created_fact_ids = list(self.facts_by_id.keys())
        logger.info(
            f"[IXBRLDocumentParser.parse] Created facts_by_id with {len(created_fact_ids)} IDs. First 5: {created_fact_ids[:5]}"
        )
        # ------------------------------------

        try:
            # Assuming default encoding or determine dynamically if needed
            with open(instance_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read cached file {instance_path}: {e}")
            return IXBRLDocument(sections=[])

        # Try different parsers to find one that works
        parsers_to_try = ["lxml", "html5lib", "html.parser"]  # Prioritize lxml/html5lib
        soup = None
        working_parser = None
        # Define the sets of tag names to try
        namespaced_fact_tags = ["ix:nonfraction", "ix:nonnumeric"]
        non_namespaced_fact_tags = ["nonfraction", "nonnumeric"]
        fact_tags_to_use = namespaced_fact_tags  # Default to namespaced

        for parser_name in parsers_to_try:
            try:
                logger.info(f"Attempting to parse with '{parser_name}'...")
                temp_soup = BeautifulSoup(content, parser_name)
                # Test if it finds namespaced tags and IDs correctly
                test_tags = temp_soup.find_all(namespaced_fact_tags)
                if test_tags:
                    logger.info(
                        f"Parser '{parser_name}' found {len(test_tags)} namespaced ix:non* tags."
                    )
                    ids_found = [tag.get("id") for tag in test_tags if tag.get("id")]
                    if ids_found:
                        logger.info(
                            f"Parser '{parser_name}' found {len(ids_found)} namespaced tags with IDs. Using this parser and namespaced tags."
                        )
                        soup = temp_soup
                        working_parser = parser_name
                        fact_tags_to_use = namespaced_fact_tags
                        break  # Found a working parser with namespaced tags
                    else:
                        logger.warning(
                            f"Parser '{parser_name}' found namespaced ix:non* tags but failed to extract IDs from them."
                        )
                else:  # Namespaced tags not found, try non-namespaced
                    logger.info(
                        f"Parser '{parser_name}' did not find namespaced ix:non* tags. Trying non-namespaced..."
                    )
                    test_tags_lower = temp_soup.find_all(non_namespaced_fact_tags)
                    if test_tags_lower:
                        logger.info(
                            f"Parser '{parser_name}' found {len(test_tags_lower)} non-namespaced non* tags."
                        )
                        ids_found_lower = [
                            tag.get("id") for tag in test_tags_lower if tag.get("id")
                        ]
                        if ids_found_lower:
                            logger.info(
                                f"Parser '{parser_name}' found {len(ids_found_lower)} non-namespaced tags with IDs. Using this parser and non-namespaced tags."
                            )
                            soup = temp_soup
                            working_parser = parser_name
                            fact_tags_to_use = non_namespaced_fact_tags
                            break  # Found a working parser with non-namespaced tags
                        else:
                            logger.warning(
                                f"Parser '{parser_name}' found non-namespaced non* tags but failed to extract IDs from them."
                            )
                    else:
                        logger.warning(
                            f"Parser '{parser_name}' did not find any namespaced 'ix:non*' or non-namespaced 'non*' fact tags."
                        )
            except Exception as e:
                logger.warning(f"Failed to parse or test with '{parser_name}': {e}")

        if not soup:
            logger.error(
                "Could not effectively parse HTML with any available parser ('lxml', 'html5lib', 'html.parser'). Ensure parsers are installed."
            )
            return IXBRLDocument(sections=[])
        logger.info(
            f"Using HTML parser: '{working_parser}' and fact tags: {fact_tags_to_use}"
        )

        # --- Detailed Debugging ---
        logger.info(f"Scanning parsed HTML for fact tags using: {fact_tags_to_use}...")
        facts_found_in_html = 0
        matched_facts_count = 0
        unmatched_ids = []
        for tag in soup.find_all(
            fact_tags_to_use
        ):  # Use the determined fact_tags_to_use
            facts_found_in_html += 1
            tag_id = tag.get("id")
            tag_name_attr = tag.get("name")  # Get name attribute from HTML tag
            logger.debug(
                f"Processing HTML tag: <{tag.name} id='{tag_id}' name='{tag_name_attr}'>"
            )

            if tag_id:
                if tag_id in self.facts_by_id:
                    matched_facts_count += 1
                    fact_obj = self.facts_by_id[tag_id]
                    # Verification step: Check if concept name matches
                    local_tag_name = None  # Initialize local_tag_name
                    if tag_name_attr and ":" in tag_name_attr:
                        # Split only on the first colon
                        parts = tag_name_attr.split(":", 1)
                        if len(parts) == 2:
                            local_tag_name = parts[1]  # Get the part after the prefix

                    # Check for mismatch only if local_tag_name was successfully extracted
                    if local_tag_name is not None:
                        if fact_obj.concept.name != local_tag_name:
                            logger.warning(
                                f"ID MATCH, NAME MISMATCH for id='{tag_id}': HTML name='{tag_name_attr}' (local='{local_tag_name}'), Fact concept.name='{fact_obj.concept.name}'"
                            )
                    elif tag_name_attr:
                        # Log if tag_name_attr exists but didn't have a colon or expected format
                        logger.debug(
                            f"ID MATCH, HTML name='{tag_name_attr}' has no prefix or unexpected format. Skipping name comparison."
                        )
                    elif not tag_name_attr:
                        logger.warning(
                            f"ID MATCH, but HTML tag id='{tag_id}' is missing the 'name' attribute."
                        )
                else:
                    unmatched_ids.append(tag_id)
                    logger.debug(
                        f"Found tag with id='{tag_id}', but this ID is not in the provided facts dictionary."
                    )
            else:
                logger.warning(
                    f"Found <{tag.name}> tag without an 'id' attribute: {tag}"
                )

        logger.info(f"HTML Scan Complete: Found {facts_found_in_html} <ix:non*> tags.")
        logger.info(
            f"Matched {matched_facts_count} tags to provided facts using the 'id' attribute."
        )
        if unmatched_ids:
            logger.warning(
                f"Found {len(unmatched_ids)} tags with IDs that were NOT in the provided facts dictionary. First 5 unmatched: {unmatched_ids[:5]}"
            )
        # --- End Debugging ---

        # Create sections based on document structure
        sections = []
        # Use a more reliable default title or detect from first heading
        body = soup.find("body")
        if not body:
            logger.error("No <body> tag found in the document.")
            return IXBRLDocument(sections=[])

        current_section_title = "Document Start"  # Initial default
        first_heading = body.find(["h1", "h2", "h3", "h4", "h5", "h6"])
        if first_heading:
            current_section_title = (
                first_heading.get_text(strip=True) or "Untitled Section"
            )

        current_parts = []

        def process_node(node):
            nonlocal current_section_title  # Allow modification

            if isinstance(node, NavigableString):
                # Append text only if it's not just whitespace or inside a fact tag
                parent_is_fact = isinstance(node.parent, Tag) and node.parent.name in [
                    "ix:nonfraction",
                    "ix:nonnumeric",
                ]
                text_content = node.strip()
                if text_content and not parent_is_fact:
                    logger.debug(f"Adding TEXT part: '{text_content[:50]}...'")
                    current_parts.append(
                        DocumentPart(type=PartType.TEXT, content=text_content)
                    )
            elif isinstance(node, Tag):
                # Skip script/style tags entirely
                if node.name in ["script", "style"]:
                    return

                # Check for headers to start new sections
                if node.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    # Save previous section if it has parts AND a title
                    if current_parts:
                        logger.info(
                            f"Completing section: '{current_section_title}' with {len(current_parts)} parts."
                        )
                        sections.append(
                            DocumentSection(
                                title=current_section_title,
                                parts=current_parts.copy(),  # Use copy
                            )
                        )
                    # Start new section
                    current_section_title = (
                        node.get_text(strip=True) or f"Untitled Section ({node.name})"
                    )
                    current_parts.clear()  # Clear parts for the new section
                    logger.debug(f"Starting new section: '{current_section_title}'")
                    # Don't process children of header, title is enough
                    return

                # Check for facts
                elif (
                    node.name in fact_tags_to_use
                ):  # Use the determined fact_tags_to_use
                    fact_id = node.get("id")
                    if fact_id and fact_id in self.facts_by_id:
                        fact_obj = self.facts_by_id[fact_id]
                        logger.debug(
                            f"Adding FACT part: id='{fact_id}', concept='{fact_obj.concept.name}'"
                        )
                        current_parts.append(
                            DocumentPart(type=PartType.FACT, content=fact_obj)
                        )
                        # Don't process children of fact tags, we captured the fact object
                        return
                    else:
                        # If it's a fact tag but not matched, log and potentially add its text content?
                        # For now, just log it was skipped (already logged above)
                        # We might need to extract text from unmatched facts if required.
                        pass

                # Recursively process children for other tags
                for child in node.children:
                    process_node(child)

        # Start processing from body
        logger.info("Starting recursive processing of document body...")
        process_node(body)

        # Add the last section if it has parts
        if current_parts:
            logger.info(
                f"Completing final section: '{current_section_title}' with {len(current_parts)} parts."
            )
            sections.append(
                DocumentSection(title=current_section_title, parts=current_parts)
            )

        # Filter out empty sections just in case
        sections = [s for s in sections if s.parts]
        logger.info(f"Processing complete. Created {len(sections)} sections.")

        return IXBRLDocument(sections=sections)


def extract_us_gaap_facts(inst: XbrlInstance) -> List[AbstractFact]:
    """
    Extract US GAAP facts from an XBRL instance.
    """
    us_gaap_facts = []
    for fact in inst.facts:
        concept = fact.concept
        if "us-gaap" in concept.schema_url:
            us_gaap_facts.append(fact)

    # Add INFO log
    logger.info(
        f"[extract_us_gaap_facts] Extracted {len(inst.facts)} total facts, {len(us_gaap_facts)} US-GAAP facts from instance."
    )
    # Log first few kept fact IDs
    kept_ids = [f.xml_id for f in us_gaap_facts if f.xml_id]
    logger.info(f"[extract_us_gaap_facts] First 5 kept fact IDs: {kept_ids[:5]}")

    return us_gaap_facts


def extract_financial_metrics(
    raw_filing: str, options: ProcessingOptions
) -> Dict[str, Any]:
    """
    Extract financial metrics from XBRL data using py-xbrl.

    Args:
        raw_filing: The raw filing content containing XBRL (or a URL to the filing)
        options: Processing options to control extraction level

    Returns:
        Dictionary of financial metrics with values and contexts
    """
    try:
        from xbrl.cache import HttpCache
        from xbrl.instance import XbrlParser, XbrlInstance
    except ImportError:
        logger.warning(
            "py-xbrl package not installed. Install with: pip install py-xbrl"
        )
        return {"error": "py-xbrl package not installed"}

    metrics = {}

    try:
        # Create a temporary file to save the raw filing for parsing
        import tempfile
        import os

        # Create a temporary directory for cache and files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up the cache
            cache_dir = os.path.join(temp_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache = HttpCache(cache_dir)

            # Set required headers for SEC EDGAR
            cache.set_headers(
                {
                    "User-Agent": "Example Company AdminContact@example.com",
                    "From": "admin@example.com",
                }
            )

            # Initialize the parser with the cache
            parser = XbrlParser(cache)

            # Determine if raw_filing is a URL or content
            if raw_filing.startswith(("http://", "https://")):
                # If it's a URL, parse directly
                xbrl_doc = parser.parse_instance(raw_filing)
            else:
                # If it's content, save to temporary file and parse
                filing_path = os.path.join(temp_dir, "filing.htm")
                with open(filing_path, "w", encoding="utf-8") as f:
                    f.write(raw_filing)

                xbrl_doc = parser.parse_instance(filing_path)

            # Basic financial metrics (for PRIMARY_FINANCIALS)
            basic_metrics = [
                "Revenue",
                "NetIncomeLoss",
                "Assets",
                "Liabilities",
                "StockholdersEquity",
            ]

            # Additional metrics for DETAILED_FINANCIALS
            detailed_metrics = [
                # Income Statement - additional
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "CostOfGoodsAndServicesSold",
                "GrossProfit",
                "OperatingExpenses",
                "OperatingIncomeLoss",
                "EarningsPerShareBasic",
                "EarningsPerShareDiluted",
                # Balance Sheet - additional
                "AssetsCurrent",
                "CashAndCashEquivalentsAtCarryingValue",
                "LiabilitiesCurrent",
                "CommonStockParOrStatedValuePerShare",
                "AccountsReceivableNetCurrent",
                "Inventory",
                "PropertyPlantAndEquipmentNet",
                # Cash Flow metrics
                "NetCashProvidedByUsedInOperatingActivities",
                "NetCashProvidedByUsedInInvestingActivities",
                "NetCashProvidedByUsedInFinancingActivities",
            ]

            # Define which metrics to extract based on options
            metrics_to_extract = []
            if ProcessingOptions.PRIMARY_FINANCIALS in options:
                metrics_to_extract.extend(basic_metrics)

            if ProcessingOptions.DETAILED_FINANCIALS in options:
                metrics_to_extract.extend(detailed_metrics)

            # Get all facts from the document using the py-xbrl API
            facts = xbrl_doc.facts

            # Extract the values for the required metrics
            for metric in metrics_to_extract:
                # Try to find the metric in different namespaces, starting with us-gaap
                namespaces = ["us-gaap", "ifrs", "dei"]

                for namespace in namespaces:
                    metric_facts = [
                        f
                        for f in facts
                        if f.concept.name == metric
                        and (
                            namespace in f.concept.prefix or namespace in f.concept.uri
                        )
                    ]

                    if metric_facts:
                        # Sort facts by period end date to get the most recent
                        sorted_facts = sorted(
                            metric_facts,
                            key=lambda x: getattr(x.period, "end_date", "1900-01-01"),
                            reverse=True,
                        )

                        if sorted_facts:
                            fact = sorted_facts[0]
                            # Extract the value and context information
                            metrics[metric] = {
                                "value": fact.value,
                                "unit": getattr(fact, "unit", "unknown"),
                                "period_end": str(
                                    getattr(fact.period, "end_date", "unknown")
                                ),
                                "period_start": str(
                                    getattr(fact.period, "start_date", "unknown")
                                ),
                            }
                            break  # Break once we've found and processed this metric

            # Create a summary DataFrame for easier analysis
            if metrics:
                df_data = []
                for metric, data in metrics.items():
                    df_data.append(
                        {
                            "Metric": metric,
                            "Value": data["value"],
                            "Unit": data["unit"],
                            "PeriodEnd": data["period_end"],
                            "PeriodStart": data["period_start"],
                        }
                    )

                if df_data:
                    metrics["summary_df"] = pl.DataFrame(df_data)

                # Calculate derived metrics if we have detailed financials
                if ProcessingOptions.DETAILED_FINANCIALS in options:
                    # Add some derived metrics
                    if "Assets" in metrics and "Liabilities" in metrics:
                        try:
                            assets = float(metrics["Assets"]["value"])
                            liabilities = float(metrics["Liabilities"]["value"])
                            metrics["DerivedEquity"] = {
                                "value": assets - liabilities,
                                "unit": metrics["Assets"]["unit"],
                                "period_end": metrics["Assets"]["period_end"],
                                "period_start": metrics["Assets"]["period_start"],
                            }
                        except (ValueError, TypeError):
                            pass

                    if "NetIncomeLoss" in metrics and "Revenue" in metrics:
                        try:
                            net_income = float(metrics["NetIncomeLoss"]["value"])
                            revenue = float(metrics["Revenue"]["value"])
                            metrics["DerivedNetMargin"] = {
                                "value": (net_income / revenue) * 100
                                if revenue != 0
                                else 0,
                                "unit": "%",
                                "period_end": metrics["Revenue"]["period_end"],
                                "period_start": metrics["Revenue"]["period_start"],
                            }
                        except (ValueError, TypeError):
                            pass

    except Exception as e:
        logger.error(f"Error processing XBRL: {str(e)}")
        metrics["error"] = str(e)

    return metrics


def extract_document_sections(
    full_text: str, options: ProcessingOptions
) -> Dict[str, str]:
    """
    Extract major sections from the filing document.

    Args:
        full_text: The full text of the filing
        options: Processing options to control extraction level

    Returns:
        Dictionary of section names and their content
    """
    sections = {}

    # Skip if not requested
    if ProcessingOptions.SECTION_EXTRACTION not in options:
        return sections

    # Define regex patterns for major sections of interest
    section_patterns = [
        (
            "business_description",
            r"(Item\s+1\.?\s*Business|ITEM\s+1\.?\s*BUSINESS).+?(Item\s+1A|ITEM\s+1A)",
            "Business Description",
        ),
        (
            "risk_factors",
            r"(Item\s+1A\.?\s*Risk\s+Factors|ITEM\s+1A\.?\s*RISK\s+FACTORS).+?(Item\s+1B|ITEM\s+1B)",
            "Risk Factors",
        ),
        (
            "mda",
            r"(Item\s+7\.?\s*Management's\s+Discussion|ITEM\s+7\.?\s*MANAGEMENT'S\s+DISCUSSION).+?(Item\s+7A|ITEM\s+7A)",
            "Management Discussion & Analysis",
        ),
        (
            "financial_statements",
            r"(Item\s+8\.?\s*Financial\s+Statements|ITEM\s+8\.?\s*FINANCIAL\s+STATEMENTS).+?(Item\s+9|ITEM\s+9)",
            "Financial Statements",
        ),
    ]

    # Extract each section
    for section_id, pattern, section_name in section_patterns:
        section_regex = re.compile(pattern, re.DOTALL | re.IGNORECASE)
        section_match = section_regex.search(full_text)

        if section_match:
            sections[section_id] = {
                "name": section_name,
                "content": section_match.group(0),
            }

    # Extract footnotes if requested
    if ProcessingOptions.FOOTNOTES in options:
        footnote_pattern = re.compile(
            r"(Notes\s+to\s+(?:Consolidated\s+)?Financial\s+Statements|NOTES\s+TO\s+(?:CONSOLIDATED\s+)?FINANCIAL\s+STATEMENTS).+?(Signatures|SIGNATURES|Item\s+9|ITEM\s+9)",
            re.DOTALL | re.IGNORECASE,
        )
        footnote_match = footnote_pattern.search(full_text)

        if footnote_match:
            sections["footnotes"] = {
                "name": "Notes to Financial Statements",
                "content": footnote_match.group(0),
            }

            # Extract individual notes if we can
            notes = {}
            note_pattern = re.compile(
                r"(Note\s+\d+[\.:]\s*[A-Z][^\.]+\.)(.+?)(?=Note\s+\d+[\.:]\s*[A-Z]|$)",
                re.DOTALL | re.IGNORECASE,
            )

            for note_match in note_pattern.finditer(sections["footnotes"]["content"]):
                note_title = note_match.group(1).strip()
                note_content = note_match.group(2).strip()

                # Create a clean section ID from the note title
                note_id = re.sub(r"[^a-z0-9_]", "_", note_title.lower())
                note_id = re.sub(r"_+", "_", note_id).strip("_")

                notes[note_id] = {"title": note_title, "content": note_content}

            if notes:
                sections["individual_notes"] = notes

    # Extract segment information if requested
    if ProcessingOptions.SEGMENT_INFO in options:
        segment_pattern = re.compile(
            r"(Segment\s+(?:Information|Data|Results)|SEGMENT\s+(?:INFORMATION|DATA|RESULTS)).+?(\d+\s+Table\s+of\s+Contents|\d+)",
            re.DOTALL | re.IGNORECASE,
        )
        segment_match = segment_pattern.search(full_text)

        if segment_match:
            sections["segment_info"] = {
                "name": "Segment Information",
                "content": segment_match.group(0),
            }

    return sections


def chunk_section(
    section_text: str, section_name: str, section_id: str, max_chunk_size: int = 1500
) -> List[Dict[str, str]]:
    """
    Split a section into manageable chunks.

    Args:
        section_text: The text content of the section
        section_name: Human-readable name of the section
        section_id: ID/key for the section
        max_chunk_size: Maximum size of each chunk in characters

    Returns:
        List of chunk dictionaries with text and metadata
    """
    chunks = []

    # Split the section text into paragraphs
    paragraphs = [p for p in section_text.split("\n\n") if p.strip()]

    current_chunk = ""
    chunk_number = 1

    for paragraph in paragraphs:
        # If adding this paragraph would exceed the max size and we already have content
        if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
            # Add the current chunk
            chunks.append(
                {
                    "text": current_chunk.strip(),
                    "section": f"{section_id}_part_{chunk_number}",
                    "section_name": f"{section_name} (Part {chunk_number})",
                }
            )

            # Start a new chunk
            current_chunk = paragraph + "\n\n"
            chunk_number += 1
        else:
            # Add to the current chunk
            current_chunk += paragraph + "\n\n"

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(
            {
                "text": current_chunk.strip(),
                "section": f"{section_id}_part_{chunk_number}"
                if chunk_number > 1
                else section_id,
                "section_name": f"{section_name} (Part {chunk_number})"
                if chunk_number > 1
                else section_name,
            }
        )

    return chunks


def process_filing_content(
    ticker: str,
    filing_type: str,
    filing_date: str,
    full_text: str,
    raw_filing: str,
    options: ProcessingOptions,
) -> List[Dict[str, Any]]:
    """
    Process the filing content according to the specified options.

    Args:
        ticker: The ticker symbol
        filing_type: The type of filing (10-K, 10-Q, etc.)
        filing_date: The date of the filing
        full_text: The full text of the filing
        raw_filing: The raw filing content (may contain XBRL)
        options: Processing options to control extraction level
        logger: Optional logger for tracking progress

    Returns:
        List of document chunks with extracted information
    """
    chunks = []

    # Always include basic info
    basic_info = {
        "text": f"Filing: {ticker} {filing_type} dated {filing_date}",
        "section": "basic_info",
        "section_name": "Filing Information",
    }
    chunks.append(basic_info)

    # Extract financial metrics if requested
    if (
        ProcessingOptions.PRIMARY_FINANCIALS in options
        or ProcessingOptions.DETAILED_FINANCIALS in options
    ) and "<xbrl" in raw_filing.lower():
        try:
            if logger:
                logger.info(f"Extracting financial metrics for {ticker} {filing_type}")

            financial_metrics = extract_financial_metrics(raw_filing, options)

            if financial_metrics and "error" not in financial_metrics:
                summary_text = (
                    f"Key Financial Metrics for {ticker} as of {filing_date}:\n\n"
                )

                # Add each metric to the summary
                for metric_name, metric_data in financial_metrics.items():
                    # Skip the summary DataFrame and derived metrics for this summary
                    if metric_name not in [
                        "summary_df",
                        "DerivedEquity",
                        "DerivedNetMargin",
                    ]:
                        summary_text += f"• {metric_name}: {metric_data['value']} {metric_data['unit']}"
                        summary_text += f" (Period: {metric_data['period_start']} to {metric_data['period_end']})\n"

                chunks.append(
                    {
                        "text": summary_text,
                        "section": "financial_metrics",
                        "section_name": "Financial Metrics",
                    }
                )

                # Add derived metrics if available
                derived_text = ""
                if "DerivedEquity" in financial_metrics:
                    derived_text += f"• Derived Equity (Assets - Liabilities): {financial_metrics['DerivedEquity']['value']} {financial_metrics['DerivedEquity']['unit']}\n"

                if "DerivedNetMargin" in financial_metrics:
                    derived_text += f"• Net Profit Margin: {financial_metrics['DerivedNetMargin']['value']:.2f}%\n"

                if derived_text:
                    chunks.append(
                        {
                            "text": f"Derived Financial Metrics for {ticker}:\n\n{derived_text}",
                            "section": "derived_metrics",
                            "section_name": "Derived Financial Metrics",
                        }
                    )
        except Exception as e:
            if logger:
                logger.warning(f"Failed to extract financial metrics: {e}")

    # Extract document sections if requested
    if ProcessingOptions.SECTION_EXTRACTION in options:
        try:
            if logger:
                logger.info(f"Extracting document sections for {ticker} {filing_type}")

            sections = extract_document_sections(full_text, options)

            for section_id, section_data in sections.items():
                # Skip the individual notes since we'll process them separately
                if section_id == "individual_notes":
                    continue

                section_name = section_data["name"]
                section_content = section_data["content"]

                # Chunk the section content
                section_chunks = chunk_section(
                    section_content, section_name, section_id
                )
                chunks.extend(section_chunks)

            # Process individual notes if available
            if "individual_notes" in sections:
                for note_id, note_data in sections["individual_notes"].items():
                    note_title = note_data["title"]
                    note_content = note_data["content"]

                    # Chunk the note content
                    note_chunks = chunk_section(
                        f"{note_title}\n\n{note_content}",
                        f"Note: {note_title}",
                        f"note_{note_id}",
                    )
                    chunks.extend(note_chunks)

        except Exception as e:
            if logger:
                logger.warning(f"Failed to extract document sections: {e}")

    return chunks


def generate_placeholder_data(
    ticker: str, filing_type: str, filing_date: str, options: ProcessingOptions
) -> List[Dict[str, Any]]:
    """
    Generate placeholder data when filing is not available.

    Args:
        ticker: The ticker symbol
        filing_type: The type of filing
        filing_date: The date of the filing
        options: The processing options requested

    Returns:
        List of placeholder document chunks
    """
    chunks = []

    # Basic information for all options
    chunks.append(
        {
            "text": f"{ticker} {filing_type} filed on {filing_date}. Filing content is not available for detailed processing.",
            "section": "overview",
            "section_name": "Filing Overview",
        }
    )

    # Add placeholders based on requested options
    requested_sections = []

    if ProcessingOptions.PRIMARY_FINANCIALS in options:
        requested_sections.append(
            (
                "financial_metrics",
                "Financial Metrics",
                f"Key financial metrics for {ticker} as reported in the {filing_type}.",
            )
        )

    if ProcessingOptions.SECTION_EXTRACTION in options:
        requested_sections.extend(
            [
                (
                    "business_description",
                    "Business Description",
                    f"Description of {ticker}'s business operations and markets.",
                ),
                (
                    "risk_factors",
                    "Risk Factors",
                    f"Key risks facing {ticker} as disclosed in the {filing_type}.",
                ),
                (
                    "mda",
                    "Management Discussion & Analysis",
                    f"Management's discussion and analysis of {ticker}'s financial condition and results of operations.",
                ),
            ]
        )

    if ProcessingOptions.FOOTNOTES in options:
        requested_sections.append(
            (
                "footnotes",
                "Notes to Financial Statements",
                f"Notes to financial statements for {ticker}, including accounting policies.",
            )
        )

    if ProcessingOptions.SEGMENT_INFO in options:
        requested_sections.append(
            (
                "segment_info",
                "Segment Information",
                f"Information about {ticker}'s operating segments and geographical regions.",
            )
        )

    # Add each placeholder section
    for section_id, section_name, section_text in requested_sections:
        chunks.append(
            {"text": section_text, "section": section_id, "section_name": section_name}
        )

    return chunks


def fetch_filing_with_xbrl(
    cik: str,
    accession_number: str,
    user_agent: str = settings.sec_user_agent,
) -> Dict[str, Any]:
    """
    Fetch a filing directly using py-xbrl.

    Args:
        cik: Central Index Key (CIK) for the company
        accession_number: The SEC accession number for the filing
        user_agent: User agent string for SEC EDGAR

    Returns:
        Dictionary containing the raw filing content, extracted text, and
        any XBRL instance document that was found
    """
    try:
        from xbrl.cache import HttpCache
        from xbrl.instance import XbrlParser
    except ImportError:
        logger.warning(
            "py-xbrl package not installed. Install with: pip install py-xbrl"
        )
        return {"error": "py-xbrl package not installed"}

    result = {
        "raw_filing": None,
        "text_content": None,
        "xbrl_instance": None,
        "error": None,
    }

    try:
        # Create a temporary directory for cache
        import tempfile
        import os
        from bs4 import BeautifulSoup

        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up the cache
            cache_dir = os.path.join(temp_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache = HttpCache(cache_dir)

            # Set required headers for SEC EDGAR
            cache.set_headers(
                {
                    "User-Agent": user_agent,
                    "From": user_agent.split()[0]
                    if "@" in user_agent
                    else "admin@example.com",
                }
            )

            # Format the CIK with leading zeros to 10 digits
            cik_formatted = cik.zfill(10) if cik.isdigit() else cik

            # Format the accession number for URL (remove dashes)
            acc_no_formatted = accession_number.replace("-", "")

            # Construct URL to the filing index page
            filing_url = f"https://www.sec.gov/ /edgar/data/{cik_formatted}/{acc_no_formatted}/{accession_number}-index.html"

            # Use the cache to get the index page
            index_content = cache.get(filing_url).decode("utf-8")
            result["raw_filing"] = index_content

            # Parse the index page to find the main filing document and potential XBRL
            soup = BeautifulSoup(index_content, "html.parser")

            # Extract text content
            if soup.text:
                result["text_content"] = soup.get_text(separator="\n", strip=True)

            # Look for XBRL instance documents in the index page
            xbrl_links = []

            # The filing table contains links to all documents
            table = soup.find("table", summary="Document Format Files")
            if table:
                for row in table.find_all("tr"):
                    cells = row.find_all("td")
                    if len(cells) >= 3:
                        # Check if this is an XBRL or XML document
                        doc_type = cells[0].text.strip().lower()
                        if ("xml" in doc_type or "xbrl" in doc_type) and cells[2].find(
                            "a"
                        ):
                            href = cells[2].find("a").get("href")
                            if href:
                                # Convert relative URL to absolute
                                if href.startswith("/"):
                                    href = f"https://www.sec.gov{href}"
                                xbrl_links.append(href)

            # Initialize the XBRL parser
            parser = XbrlParser(cache)

            # Try to parse the first XBRL document found
            for xbrl_url in xbrl_links:
                try:
                    xbrl_doc = parser.parse_instance(xbrl_url)
                    result["xbrl_instance"] = xbrl_doc
                    break  # Successfully parsed an XBRL document
                except Exception as e:
                    logger.warning(f"Failed to parse XBRL document {xbrl_url}: {e}")

    except Exception as e:
        logger.error(f"Error fetching filing: {str(e)}")
        result["error"] = str(e)

    return result


def extract_text_from_html(html_content: str) -> str:
    """
    Extract readable text from HTML content.

    Args:
        html_content: Raw HTML content

    Returns:
        Plain text extracted from HTML
    """
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()

        # Get text
        text = soup.get_text(separator="\n", strip=True)

        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())

        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

        # Remove blank lines
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text
    except Exception as e:
        logger.error(f"Error extracting text from HTML: {str(e)}")
        return html_content  # Return original content if extraction fails


def save_facts_to_delta(
    doc: IXBRLDocument,
    spark_session,
    table_name: str,
    filing_metadata: Dict[str, Any] = None,
) -> None:
    """
    Saves the facts from an IXBRLDocument to a Delta Lake table.

    Args:
        doc: The IXBRLDocument containing the facts
        spark_session: The Spark session to use
        table_name: Name of the Delta table to write to
        filing_metadata: Optional metadata about the filing (ticker, date, etc.)
    """
    from pyspark.sql import functions as F

    # Get facts as Spark DataFrame
    df = doc.to_dataframe(format="spark")

    # Add filing metadata if provided
    if filing_metadata:
        for key, value in filing_metadata.items():
            df = df.withColumn(key, F.lit(value))

    # Write to Delta table
    df.write.format("delta").mode("append").saveAsTable(table_name)


# def process_filings_structured_sections(
#     filings_df: pl.DataFrame, edgar_fetcher: EDGARFetcher
# ) -> Tuple[pl.DataFrame, pl.DataFrame]:
#     """
#     Processes a list of XBRL filing URLs, parses the HTML structure,
#     extracts numeric and text facts linked to document sections,
#     and aggregates them into separate Polars DataFrames.

#     Args:
#         filings_df: DataFrame containing 'ticker', 'xbrl_instance_url',
#                     'filingDate', and 'reportDate' columns.
#         edgar_fetcher: An instance of EDGARFetcher to handle SEC interactions.

#     Returns:
#         A tuple containing two Polars DataFrames:
#         1. Aggregated numeric facts DataFrame (schema includes ticker, dates, section_name).
#         2. Aggregated text facts DataFrame (schema includes ticker, dates, section_name).
#     """
#     logger.info("[process_filing_urls_structured] Starting structured fact extraction.")

#     # Use the original schemas that include section_name
#     numeric_schema_structured_with_ticker = {
#         **TARGET_SCHEMA_NUMERIC_POLARS,
#         "ticker": pl.Utf8,
#     }
#     text_schema_structured_with_ticker = {
#         **TARGET_SCHEMA_TEXT_POLARS,
#         "ticker": pl.Utf8,
#     }

#     all_numeric_facts_df = pl.DataFrame(schema=numeric_schema_structured_with_ticker)
#     all_text_facts_df = pl.DataFrame(schema=text_schema_structured_with_ticker)

#     try:
#         # Use the cache from edgar_fetcher for doc_parser
#         doc_parser = IXBRLDocumentParser(edgar_fetcher.http_cache)
#     except ImportError:
#         logger.error("py-xbrl package not installed. Run: pip install py-xbrl")
#         return all_numeric_facts_df, all_text_facts_df

#     # Check input columns
#     # Use the _dt columns for dates
#     expected_input_cols = {
#         "ticker",
#         "xbrl_instance_url",
#         "filing_date_dt",
#         "report_date_dt",
#     }
#     if not expected_input_cols.issubset(filings_df.columns):
#         logger.error(
#             f"[process_filing_urls_structured] Input DataFrame missing expected columns. Need: {expected_input_cols}, Got: {filings_df.columns}. Returning empty DataFrames."
#         )
#         return all_numeric_facts_df, all_text_facts_df

#     processed_count = 0
#     for record in filings_df.iter_rows(named=True):
#         url = record.get("xbrl_instance_url")
#         ticker = record.get("ticker")
#         # Extract dates using the _dt columns
#         filing_date = record.get("filing_date_dt")
#         report_date = record.get("report_date_dt")

#         if not url or not ticker:
#             logger.warning(
#                 f"[structured] Skipping record due to missing URL or Ticker: {record}"
#             )
#             continue

#         try:
#             # 1. Parse the XBRL instance using EDGARFetcher
#             inst = edgar_fetcher.parse_filing_xbrl(record)
#             if not inst:
#                 logger.warning(
#                     f"[structured] Skipping {url} (Ticker: {ticker}): Instance parsing failed via EDGARFetcher."
#                 )
#                 continue

#             # 2. Parse the HTML document structure using the instance
#             document = doc_parser.parse(xbrl_instance=inst)

#             # 3. Create DataFrames from the parsed document structure
#             numeric_df = document.to_numeric_dataframe(
#                 filing_date=filing_date, report_date=report_date
#             )
#             text_df = document.to_text_dataframe(
#                 filing_date=filing_date, report_date=report_date
#             )

#             # 4. Add ticker column before concatenating
#             if not numeric_df.is_empty():
#                 numeric_df = numeric_df.with_columns(pl.lit(ticker).alias("ticker"))
#             if not text_df.is_empty():
#                 text_df = text_df.with_columns(pl.lit(ticker).alias("ticker"))

#             # 5. Concatenate into respective accumulators
#             if not numeric_df.is_empty():
#                 all_numeric_facts_df = pl.concat(
#                     [all_numeric_facts_df, numeric_df], how="vertical_relaxed"
#                 )
#             if not text_df.is_empty():
#                 all_text_facts_df = pl.concat(
#                     [all_text_facts_df, text_df], how="vertical_relaxed"
#                 )

#             processed_count += 1

#         except Exception as e:
#             logger.error(
#                 f"[structured] ERROR processing {url} (Ticker: {ticker}): {e}",
#                 exc_info=True,
#             )
#             # Continue to next filing

#     logger.info(
#         f"[process_filing_urls_structured] Processing complete. Processed {processed_count} filings."
#     )
#     logger.info(f"[structured] Final Numeric DF shape: {all_numeric_facts_df.shape}")
#     logger.info(f"[structured] Final Text DF shape: {all_text_facts_df.shape}")

#     return all_numeric_facts_df, all_text_facts_df


def process_filings_structured_direct(
    filings_df: pl.DataFrame, edgar_fetcher: EDGARFetcher
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Processes a list of XBRL filing URLs, extracts numeric and text facts
    directly from the parsed XBRL instance (bypassing HTML structure),
    and aggregates them into separate Polars DataFrames.

    Args:
        filings_df: DataFrame containing 'ticker', 'xbrl_instance_url',
                    'filingDate', and 'reportDate' columns.
        edgar_fetcher: An instance of EDGARFetcher to handle SEC interactions.

    Returns:
        A tuple containing two Polars DataFrames:
        1. Aggregated numeric facts DataFrame (schema includes ticker, dates, NO section_name).
        2. Aggregated text facts DataFrame (schema includes ticker, dates, NO section_name).
    """
    logger.info("[process_filing_urls_direct] Starting direct fact extraction.")

    # Use the DIRECT schemas (no section_name)
    numeric_schema_direct_with_ticker = {
        **TARGET_SCHEMA_NUMERIC_DIRECT_POLARS,
        "ticker": pl.Utf8,
    }
    text_schema_direct_with_ticker = {
        **TARGET_SCHEMA_TEXT_DIRECT_POLARS,
        "ticker": pl.Utf8,
    }

    all_numeric_facts_df = pl.DataFrame(schema=numeric_schema_direct_with_ticker)
    all_text_facts_df = pl.DataFrame(schema=text_schema_direct_with_ticker)

    # Check input columns
    # Use the _dt columns for dates
    expected_input_cols = {
        "ticker",
        "xbrl_instance_url",
        "filing_date_dt",
        "report_date_dt",
    }
    if not expected_input_cols.issubset(filings_df.columns):
        logger.error(
            f"[process_filing_urls_direct] Input DataFrame missing expected columns. Need: {expected_input_cols}, Got: {filings_df.columns}. Returning empty DataFrames."
        )
        return all_numeric_facts_df, all_text_facts_df

    processed_count = 0
    for record in filings_df.iter_rows(named=True):
        url = record.get("xbrl_instance_url")
        ticker = record.get("ticker")
        # Extract dates using the _dt columns
        filing_date = record.get("filing_date_dt")
        report_date = record.get("report_date_dt")

        if not url or not ticker:
            logger.warning(
                f"[direct] Skipping record due to missing URL or Ticker: {record}"
            )
            continue

        try:
            # 1. Parse the XBRL instance using EDGARFetcher
            inst = edgar_fetcher.parse_filing_xbrl(record)

            if not inst:
                logger.warning(
                    f"[direct] Skipping {url} (Ticker: {ticker}): Instance parsing failed via EDGARFetcher."
                )
                continue

            # 2. Extract facts directly from the instance using the new helper
            numeric_df, text_df = extract_facts_from_instance(
                inst, filing_date=filing_date, report_date=report_date
            )

            # 3. Add ticker column before concatenating
            if not numeric_df.is_empty():
                numeric_df = numeric_df.with_columns(pl.lit(ticker).alias("ticker"))
            if not text_df.is_empty():
                text_df = text_df.with_columns(pl.lit(ticker).alias("ticker"))

            # 4. Concatenate into respective accumulators
            if not numeric_df.is_empty():
                all_numeric_facts_df = pl.concat(
                    [all_numeric_facts_df, numeric_df], how="vertical_relaxed"
                )
            if not text_df.is_empty():
                all_text_facts_df = pl.concat(
                    [all_text_facts_df, text_df], how="vertical_relaxed"
                )

            processed_count += 1

        except Exception as e:
            logger.error(
                f"[direct] ERROR processing {url} (Ticker: {ticker}): {e}",
                exc_info=True,
            )
            # Continue to next filing

    logger.info(
        f"[process_filing_urls_direct] Processing complete. Processed {processed_count} filings."
    )
    logger.info(f"[direct] Final Numeric DF shape: {all_numeric_facts_df.shape}")
    logger.info(f"[direct] Final Text DF shape: {all_text_facts_df.shape}")

    return all_numeric_facts_df, all_text_facts_df


# if __name__ == "__main__":
#     # Note: This example usage might need adjustment depending on how Universe evolves.
#     # It currently assumes a direct call to get_filings() and then processing.
#     # The new Universe pattern would likely involve get_numeric_facts().
#     from alphaledger.universe import Universe
#     from alphaledger.config import settings

#     # Use basicConfig for simplicity in example, or configure logger as needed
#     import logging

#     logging.basicConfig(level=logging.INFO)
#     # logger = get_logger(__name__) # Use project's logger if available

#     try:
#         # Example: Use a specific universe name defined in your settings or environment
#         universe_name = settings.universe_name  # Or replace with "your_universe_name"
#         logger.info(f"Loading universe: {universe_name}")
#         universe = Universe(universe_name)

#         # Ensure metadata is available
#         logger.info("Ensuring filings metadata is available...")
#         universe.collect_filings()

#         # Get the filings metadata DataFrame (required input for processing functions)
#         filings_lf = universe.get_filings_lazy()
#         if filings_lf is None:
#             logger.error("Failed to get filings metadata LazyFrame. Cannot proceed.")
#         else:
#             logger.info("Collecting filings metadata...")
#             # Select columns needed by both direct and structured methods
#             required_cols = ["ticker", "xbrl_instance_url", "filingDate", "reportDate"]
#             if not all(col in filings_lf.columns for col in required_cols):
#                 logger.error(
#                     f"Filings metadata missing required columns. Need: {required_cols}, Have: {filings_lf.columns}"
#                 )
#             else:
#                 filings_df = filings_lf.select(required_cols).collect()

#                 if filings_df.is_empty():
#                     logger.warning(
#                         "No filings metadata found for the universe. No facts to process."
#                     )
#                 else:
#                     # --- Choose which processing method to run ---
#                     PROCESS_METHOD = "direct"  # or "structured"

#                     logger.info(
#                         f"Processing {len(filings_df)} filings using '{PROCESS_METHOD}' method..."
#                     )

#                     if PROCESS_METHOD == "direct":
#                         edgar_fetcher = EDGARFetcher()
#                         numeric_df, text_df = process_filings_structured_direct(
#                             filings_df, edgar_fetcher
#                         )
#                         # Define output paths for direct facts
#                         numeric_path = (
#                             settings.output_dir
#                             / f"{universe.name}_numeric_facts_direct.delta"
#                         )
#                         text_path = (
#                             settings.output_dir
#                             / f"{universe.name}_text_facts_direct.delta"
#                         )
#                     else:  # structured
#                         edgar_fetcher = EDGARFetcher()
#                         numeric_df, text_df = process_filings_structured_sections(
#                             filings_df, edgar_fetcher
#                         )
#                         # Define output paths for structured facts
#                         numeric_path = (
#                             settings.output_dir
#                             / f"{universe.name}_numeric_facts_structured.delta"
#                         )
#                         text_path = (
#                             settings.output_dir
#                             / f"{universe.name}_text_facts_structured.delta"
#                         )

#                     logger.info(
#                         f"Saving numeric facts ({len(numeric_df)} rows) to {numeric_path}..."
#                     )
#                     numeric_df.write_delta(str(numeric_path), mode="overwrite")

#                     logger.info(
#                         f"Saving text facts ({len(text_df)} rows) to {text_path}..."
#                     )
#                     text_df.write_delta(str(text_path), mode="overwrite")

#                     logger.info("Processing and saving complete.")

#     except FileNotFoundError as e:
#         logger.error(f"Universe definition not found: {e}")
#     except Exception as e:
#         logger.error(
#             f"An error occurred in the main execution block: {e}", exc_info=True
#         )

# --- Direct Fact Row Conversion Helpers (No Section Title) ---


def _direct_fact_to_row(fact: AbstractFact) -> Dict[str, Any]:
    """Converts any fact to a dictionary row with common fields (excluding section)."""
    row_base = {}
    # Basic Info
    row_base["concept_name"] = fact.concept.name
    if fact.concept.schema_url:
        row_base["concept_namespace"] = fact.concept.schema_url.split("/")[-2]
    else:
        row_base["concept_namespace"] = None
    row_base["fact_type"] = type(fact).__name__

    context = getattr(fact, "context", None)
    period_instant = None
    period_start = None
    period_end = None

    if context:
        from xbrl.instance import (
            InstantContext,
            TimeFrameContext,
        )  # Lazy import for type checking

        if isinstance(context, InstantContext):
            period_instant = getattr(context, "instant_date", None)
        elif isinstance(context, TimeFrameContext):
            period_start = getattr(context, "start_date", None)
            period_end = getattr(context, "end_date", None)

        else:
            # Could be ForeverContext or other AbstractContext subclass
            pass
    else:
        # No context object found on fact
        pass

    # Assign extracted dates (or None) to the row
    row_base["period_instant"] = period_instant
    row_base["period_start"] = period_start
    row_base["period_end"] = period_end

    return row_base


def _direct_numeric_fact_to_row(
    fact: NumericFact,
    filing_date: Optional[Any] = None,
    report_date: Optional[Any] = None,
) -> Dict[str, Any]:
    """Converts a NumericFact directly to a dictionary row conforming to direct numeric schema."""
    numeric_target_keys = list(TARGET_SCHEMA_NUMERIC_DIRECT_POLARS.keys())
    row = {key: None for key in numeric_target_keys}
    common_fields = _direct_fact_to_row(fact)

    for key, value in common_fields.items():
        if key in row:
            row[key] = value

    # Add filing dates
    row["filing_date"] = filing_date
    row["report_date"] = report_date

    # Populate Numeric Specific Fields
    try:
        row["fact_value"] = float(fact.value) if fact.value is not None else None
    except (ValueError, TypeError):
        row["fact_value"] = None

    unit = getattr(fact, "unit", None)
    row["unit"] = str(unit) if unit else None

    metadata_dict = {"decimals": None, "precision": None}
    decimals = getattr(fact, "decimals", None)
    precision = getattr(fact, "precision", None)
    try:
        metadata_dict["decimals"] = int(decimals) if decimals is not None else None
    except:
        pass
    try:
        metadata_dict["precision"] = int(precision) if precision is not None else None
    except:
        pass
    row["metadata"] = metadata_dict

    return row


def _direct_text_fact_to_row(
    fact: TextFact,
    filing_date: Optional[Any] = None,
    report_date: Optional[Any] = None,
) -> Dict[str, Any]:
    """Converts a TextFact directly to a dictionary row conforming to direct text schema."""
    text_target_keys = list(TARGET_SCHEMA_TEXT_DIRECT_POLARS.keys())
    row = {key: None for key in text_target_keys}
    common_fields = _direct_fact_to_row(fact)

    for key, value in common_fields.items():
        if key in row:
            row[key] = value

    # Add filing dates
    row["filing_date"] = filing_date
    row["report_date"] = report_date

    # Populate Text Specific Fields
    row["fact_value"] = str(fact.value) if fact.value is not None else None

    return row


def extract_facts_from_instance(
    instance: XbrlInstance,
    filing_date: Optional[Any] = None,
    report_date: Optional[Any] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Extracts numeric and text facts directly from an XbrlInstance object,
    bypassing HTML document parsing.

    Args:
        instance: The parsed XbrlInstance object.
        filing_date: Optional filing date to associate with the facts.
        report_date: Optional report date to associate with the facts.

    Returns:
        A tuple containing two Polars DataFrames:
        1. Numeric facts DataFrame (schema: TARGET_SCHEMA_NUMERIC_DIRECT_POLARS).
        2. Text facts DataFrame (schema: TARGET_SCHEMA_TEXT_DIRECT_POLARS).
    """
    numeric_rows = []
    text_rows = []

    # Optional: Filter facts here if needed (e.g., for US-GAAP only)
    facts_to_process = instance.facts
    # To filter for US-GAAP:
    # facts_to_process = [f for f in instance.facts if "us-gaap" in f.concept.schema_url]

    for fact in facts_to_process:
        if isinstance(fact, NumericFact):
            row = _direct_numeric_fact_to_row(fact, filing_date, report_date)
            numeric_rows.append(row)
        elif isinstance(fact, TextFact):  # Or just else? Assuming AbstractFact is base
            row = _direct_text_fact_to_row(fact, filing_date, report_date)
            text_rows.append(row)
        # else: # Handle other potential AbstractFact types if necessary
        #     logger.debug(f"Skipping fact of unhandled type: {type(fact)}")

    # Create DataFrames using the direct schemas
    numeric_df = pl.DataFrame(
        numeric_rows, schema=TARGET_SCHEMA_NUMERIC_DIRECT_POLARS, strict=False
    )
    text_df = pl.DataFrame(
        text_rows, schema=TARGET_SCHEMA_TEXT_DIRECT_POLARS, strict=False
    )

    return numeric_df, text_df
