"""
Module for processing XBRL financial filings.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Set, Union
from enum import Flag, auto, Enum
from io import StringIO
import pandas as pd
from pathlib import Path
from xbrl.instance import XbrlInstance, AbstractFact
from alphaledger.config import settings
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup, Tag, NavigableString

# Use TYPE_CHECKING to avoid circular import error at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .formatter import DocumentFormatter  # Import for type hinting


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
        """
        Formats the document content into a string using the provided formatter.

        Args:
            formatter: An instance of a DocumentFormatter subclass.

        Returns:
            A string representation of the document based on the formatter's rules.
        """
        return formatter.format_document(self)


class IXBRLDocumentParser:
    def __init__(
        self, instance_path: str, facts: List[AbstractFact], encoding: str = None
    ):
        self.instance_path = instance_path
        self.encoding = encoding
        self.facts_by_id: Dict[str, AbstractFact] = {
            fact.xml_id: fact for fact in facts if fact.xml_id
        }
        logging.info(
            f"Parser Initialized: Received {len(facts)} facts, created lookup for {len(self.facts_by_id)} facts with IDs."
        )
        if len(facts) > len(self.facts_by_id):
            logging.warning(
                f"{len(facts) - len(self.facts_by_id)} facts were missing an xml_id in the input list."
            )
        logging.info(
            f"First 5 available fact IDs in lookup: {list(self.facts_by_id.keys())[:5]}"
        )

    def parse(self) -> IXBRLDocument:
        # Read the HTML/iXBRL file
        try:
            with open(self.instance_path, "r", encoding=self.encoding) as f:
                content = f.read()
        except Exception as e:
            logging.error(f"Failed to read file {self.instance_path}: {e}")
            return IXBRLDocument(sections=[])

        # Try different parsers to find one that works
        parsers_to_try = ["lxml", "html5lib", "html.parser"]  # Prioritize lxml/html5lib
        soup = None
        working_parser = None
        for parser_name in parsers_to_try:
            try:
                logging.info(f"Attempting to parse with '{parser_name}'...")
                temp_soup = BeautifulSoup(content, parser_name)
                # Test if it finds namespaced tags and IDs correctly
                test_tags = temp_soup.find_all(["ix:nonfraction", "ix:nonnumeric"])
                if test_tags:
                    logging.info(
                        f"Parser '{parser_name}' found {len(test_tags)} ix:non* tags."
                    )
                    ids_found = [tag.get("id") for tag in test_tags if tag.get("id")]
                    if ids_found:
                        logging.info(
                            f"Parser '{parser_name}' found {len(ids_found)} tags with IDs. Using this parser."
                        )
                        logging.debug(
                            f"Sample IDs found by '{parser_name}': {ids_found[:5]}"
                        )
                        soup = temp_soup
                        working_parser = parser_name
                        break  # Found a working parser
                    else:
                        logging.warning(
                            f"Parser '{parser_name}' found ix:non* tags but failed to extract IDs."
                        )
                else:
                    # Check for lowercase tags as a fallback
                    test_tags_lower = temp_soup.find_all(["nonfraction", "nonnumeric"])
                    if test_tags_lower:
                        logging.warning(
                            f"Parser '{parser_name}' did not find namespaced 'ix:non*' tags, but found {len(test_tags_lower)} lowercase tags."
                        )
                    else:
                        logging.warning(
                            f"Parser '{parser_name}' did not find any 'ix:non*' or lowercase fact tags."
                        )

            except Exception as e:
                logging.warning(f"Failed to parse or test with '{parser_name}': {e}")

        if not soup:
            logging.error(
                "Could not effectively parse HTML with any available parser ('lxml', 'html5lib', 'html.parser'). Ensure parsers are installed."
            )
            return IXBRLDocument(sections=[])
        logging.info(f"Using HTML parser: '{working_parser}'")

        # --- Detailed Debugging ---
        logging.info("Scanning parsed HTML for fact tags...")
        facts_found_in_html = 0
        matched_facts_count = 0
        unmatched_ids = []
        for tag in soup.find_all(["ix:nonfraction", "ix:nonnumeric"]):
            facts_found_in_html += 1
            tag_id = tag.get("id")
            tag_name_attr = tag.get("name")  # Get name attribute from HTML tag
            logging.debug(
                f"Processing HTML tag: <{tag.name} id='{tag_id}' name='{tag_name_attr}'>"
            )

            if tag_id:
                if tag_id in self.facts_by_id:
                    matched_facts_count += 1
                    fact_obj = self.facts_by_id[tag_id]
                    # Verification step: Check if concept name matches
                    if tag_name_attr and ":" in tag_name_attr:
                        # Split only on the first colon
                        parts = tag_name_attr.split(":", 1)
                        if len(parts) == 2:
                            local_tag_name = parts[1]  # Get the part after the prefix

                    if tag_name_attr and fact_obj.concept.name != local_tag_name:
                        logging.warning(
                            f"ID MATCH, NAME MISMATCH for id='{tag_id}': HTML name='{tag_name_attr}' (local='{local_tag_name}'), Fact concept.name='{fact_obj.concept.name}'"
                        )
                    elif not tag_name_attr:
                        logging.warning(
                            f"ID MATCH, but HTML tag id='{tag_id}' is missing the 'name' attribute."
                        )
                else:
                    unmatched_ids.append(tag_id)
                    logging.debug(
                        f"Found tag with id='{tag_id}', but this ID is not in the provided facts dictionary."
                    )
            else:
                logging.warning(
                    f"Found <{tag.name}> tag without an 'id' attribute: {tag}"
                )

        logging.info(f"HTML Scan Complete: Found {facts_found_in_html} <ix:non*> tags.")
        logging.info(
            f"Matched {matched_facts_count} tags to provided facts using the 'id' attribute."
        )
        if unmatched_ids:
            logging.warning(
                f"Found {len(unmatched_ids)} tags with IDs that were NOT in the provided facts dictionary. First 5 unmatched: {unmatched_ids[:5]}"
            )
        # --- End Debugging ---

        # Create sections based on document structure
        sections = []
        # Use a more reliable default title or detect from first heading
        body = soup.find("body")
        if not body:
            logging.error("No <body> tag found in the document.")
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
                    logging.debug(f"Adding TEXT part: '{text_content[:50]}...'")
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
                        logging.info(
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
                    logging.debug(f"Starting new section: '{current_section_title}'")
                    # Don't process children of header, title is enough
                    return

                # Check for facts
                elif node.name in ["ix:nonfraction", "ix:nonnumeric"]:
                    fact_id = node.get("id")
                    if fact_id and fact_id in self.facts_by_id:
                        fact_obj = self.facts_by_id[fact_id]
                        logging.debug(
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
        logging.info("Starting recursive processing of document body...")
        process_node(body)

        # Add the last section if it has parts
        if current_parts:
            logging.info(
                f"Completing final section: '{current_section_title}' with {len(current_parts)} parts."
            )
            sections.append(
                DocumentSection(title=current_section_title, parts=current_parts)
            )

        # Filter out empty sections just in case
        sections = [s for s in sections if s.parts]
        logging.info(f"Processing complete. Created {len(sections)} sections.")

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
        logging.warning(
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
            facts = xbrl_doc.get_facts()

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
                    metrics["summary_df"] = pd.DataFrame(df_data)

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
        logging.error(f"Error processing XBRL: {str(e)}")
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
    logger: Optional[logging.Logger] = None,
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
        logging.warning(
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
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik_formatted}/{acc_no_formatted}/{accession_number}-index.html"

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
                    logging.warning(f"Failed to parse XBRL document {xbrl_url}: {e}")

    except Exception as e:
        logging.error(f"Error fetching filing: {str(e)}")
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
        logging.error(f"Error extracting text from HTML: {str(e)}")
        return html_content  # Return original content if extraction fails
