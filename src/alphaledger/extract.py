"""
Module for extracting structured data (like XBRL facts) from parsed documents.
"""

from typing import TYPE_CHECKING, Optional, List, Dict, Any
import logging

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    import pandas as pd
    from .process_xbrl import IXBRLDocument, ProcessingOptions
    from xbrl.instance import AbstractFact

logger = logging.getLogger(__name__)


class FactExtractor:
    """
    Extracts structured XBRL fact data from an IXBRLDocument based on
    ProcessingOptions.
    """

    def __init__(
        self, document: "IXBRLDocument", options: Optional["ProcessingOptions"] = None
    ):
        """
        Initialize the extractor.

        Args:
            document: The parsed IXBRLDocument containing text and facts.
            options: ProcessingOptions to guide which facts/details to extract.
        """
        self.document = document
        self.options = options
        logger.info(f"FactExtractor initialized with options: {options}")

    def extract_facts_as_list(
        self, concepts_of_interest: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extracts XBRL facts as a list of dictionaries.

        Args:
            concepts_of_interest: Optional list of concept names (e.g., 'us-gaap:Assets')
                                   to filter for. If None, potentially all facts are considered
                                   based on self.options.

        Returns:
            A list of dictionaries, where each dictionary represents an extracted fact
            with relevant fields (concept, value, unit, period, context, etc.).
        """
        logger.warning("FactExtractor.extract_facts_as_list is not yet implemented.")
        # Implementation Placeholder:
        # 1. Define which concepts to extract based on self.options and concepts_of_interest.
        # 2. Iterate through self.document.sections and section.parts.
        # 3. If part.is_fact():
        # 4.   Get the fact object (part.content).
        # 5.   Check if its concept matches the desired concepts.
        # 6.   Extract required fields (value, unit_id, period_start, period_end, context_id, etc.)
        #      using hasattr() for safety. Handle different fact types (TextFact, Numeric).
        # 7.   Append structured dictionary to a results list.
        # 8. Return results list.
        return []  # Stub implementation

    def extract_facts_as_dataframe(
        self, concepts_of_interest: Optional[List[str]] = None
    ) -> "pd.DataFrame":
        """
        Extracts XBRL facts as a Pandas DataFrame.

        Args:
            concepts_of_interest: Optional list of concept names to filter for.

        Returns:
            A Pandas DataFrame containing the extracted fact data, ready for analysis
            or database insertion. Returns an empty DataFrame if not implemented or
            if pandas is not available.
        """
        logger.warning(
            "FactExtractor.extract_facts_as_dataframe is not yet implemented."
        )
        try:
            import pandas as pd

            # Implementation Placeholder:
            # 1. Call extract_facts_as_list()
            # 2. Convert the list of dictionaries into a DataFrame.
            # 3. Perform any necessary type conversions or restructuring.
            # 4. Return DataFrame.
            return pd.DataFrame()  # Stub implementation
        except ImportError:
            logger.error(
                "Pandas library is required for extract_facts_as_dataframe but not installed."
            )
            # Return an empty DataFrame or raise an error, depending on desired behavior.
            import pandas as pd  # Add import here to satisfy type checker outside try-except

            return pd.DataFrame()  # Stub implementation
