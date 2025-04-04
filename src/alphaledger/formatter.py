"""
Module for formatting parsed IXBRLDocument objects into different string representations.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast, Optional
from bs4 import BeautifulSoup  # Added for potential HTML stripping

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from .process_xbrl import (
        IXBRLDocument,
        DocumentSection,
        DocumentPart,
        PartType,
        ProcessingOptions,  # Import ProcessingOptions
    )
    from xbrl.instance import AbstractFact, TextFact  # Import TextFact

# Default truncation length for TextFacts at lower detail levels
DEFAULT_TEXT_FACT_TRUNCATION = 200


class DocumentFormatter(ABC):
    """Abstract base class for formatting IXBRLDocument objects."""

    # Add options to the base class __init__ if common initialization needed,
    # otherwise handle in subclasses. For now, handle in subclasses.

    @abstractmethod
    def format_document_start(self, doc: "IXBRLDocument") -> str:
        """Return the string to prepend to the entire formatted document."""
        pass

    @abstractmethod
    def format_document_end(self, doc: "IXBRLDocument") -> str:
        """Return the string to append to the entire formatted document."""
        pass

    @abstractmethod
    def format_section_start(self, section: "DocumentSection") -> str:
        """Return the string to prepend before a section's content."""
        pass

    @abstractmethod
    def format_section_end(self, section: "DocumentSection") -> str:
        """Return the string to append after a section's content."""
        pass

    @abstractmethod
    def format_text_part(self, part: "DocumentPart") -> str:
        """Return the formatted string for a text part."""
        pass

    @abstractmethod
    def format_fact_part(self, part: "DocumentPart") -> str:
        """Return the formatted string for an XBRL fact part."""
        pass

    def format_document(self, doc: "IXBRLDocument") -> str:
        """
        Formats the entire IXBRLDocument using the specific formatter methods.
        """
        # Import locally to avoid circular dependency issues at runtime
        from .process_xbrl import PartType

        output: list[str] = []
        output.append(self.format_document_start(doc))

        for section in doc.sections:
            output.append(self.format_section_start(section))
            for part in section.parts:
                if part.type == PartType.TEXT:
                    output.append(self.format_text_part(part))
                elif part.type == PartType.FACT:
                    output.append(self.format_fact_part(part))
            output.append(self.format_section_end(section))

        output.append(self.format_document_end(doc))
        return "".join(output)


class MarkdownFormatter(DocumentFormatter):
    """Formats an IXBRLDocument into a Markdown string."""

    def __init__(self, options: Optional["ProcessingOptions"] = None):
        """
        Initialize the formatter with processing options.

        Args:
            options: ProcessingOptions flags to control formatting details.
        """
        self.options = options
        # Import locally or under TYPE_CHECKING if needed earlier
        from .process_xbrl import ProcessingOptions

        self._processing_options = ProcessingOptions  # Store for easy access

    def format_document_start(self, doc: "IXBRLDocument") -> str:
        return ""  # No specific start for the whole document in basic Markdown

    def format_document_end(self, doc: "IXBRLDocument") -> str:
        return ""  # No specific end

    def format_section_start(self, section: "DocumentSection") -> str:
        # Use H2 for section titles
        return f"## {section.title}\n\n"

    def format_section_end(self, section: "DocumentSection") -> str:
        return "\n"  # Add a newline after each section

    def format_text_part(self, part: "DocumentPart") -> str:
        # Assume text parts are paragraphs, add double newline
        return f"{part.content}\n\n"

    def format_fact_part(self, part: "DocumentPart") -> str:
        # Type hint for clarity
        from xbrl.instance import AbstractFact, TextFact

        fact = cast(AbstractFact, part.content)

        # --- Handle TextFacts based on options ---
        if isinstance(fact, TextFact):
            max_len = DEFAULT_TEXT_FACT_TRUNCATION
            show_full = True  # Default to showing full text
            strip_html = True  # Default to stripping HTML

            if self.options:
                # Show truncated text for basic level
                if (
                    self.options <= self._processing_options.LEVEL_1
                ):  # Includes BASIC_INFO | PRIMARY_FINANCIALS
                    show_full = False
                    strip_html = True
                # Higher levels show full text (already default)
                # else: # LEVEL_2 or LEVEL_3
                #     show_full = True
                #     strip_html = True # Keep stripping HTML for clarity

            text_content = fact.value
            if strip_html:
                try:
                    soup = BeautifulSoup(text_content, "html.parser")
                    text_content = soup.get_text(separator="\n", strip=True)
                except Exception:
                    # If parsing fails, use original value (might contain tags)
                    pass  # Keep original text_content

            if show_full:
                # Use Markdown code block for potentially long/multi-line text
                return f"*   Text Fact:\n    *   **Concept:** `{fact.concept.name}`\n    *   **Value:**\n        ```\n{text_content}\n        ```\n\n"
            else:
                truncated_value = (
                    (text_content[:max_len] + "...")
                    if len(text_content) > max_len
                    else text_content
                )
                # Escape potential markdown in truncated text? For now, assume plain text.
                return f"*   Text Fact:\n    *   **Concept:** `{fact.concept.name}`\n    *   **Value (truncated):** {truncated_value}\n\n"
        # --- Handle other Fact types (Numeric etc.) ---
        else:
            fact_details = [
                f"**Concept:** `{fact.concept.name}`",
                f"**Value:** `{fact.value}`",
            ]
            # Check for unit before accessing
            if hasattr(fact, "unit") and fact.unit:
                unit_display = getattr(fact.unit, "id", str(fact.unit))
                fact_details.append(f"**Unit:** `{unit_display}`")

            # Check for period before accessing its attributes
            if hasattr(fact, "period") and fact.period:
                if hasattr(fact.period, "instant"):
                    fact_details.append(f"**Period:** `{fact.period.instant}`")
                elif hasattr(fact.period, "start_date") and hasattr(
                    fact.period, "end_date"
                ):
                    fact_details.append(
                        f"**Period:** `{fact.period.start_date}` to `{fact.period.end_date}`"
                    )

            # Check for context before accessing
            if hasattr(fact, "context") and fact.context:
                context_display = getattr(fact.context, "id", str(fact.context))
                fact_details.append(f"**Context:** `{context_display}`")

            # Format as a bullet point list within the text flow
            return f"*   Fact:\n    *   " + "\n    *   ".join(fact_details) + "\n\n"


class PlainTextFormatter(DocumentFormatter):
    """Formats an IXBRLDocument into a plain text string."""

    def __init__(self, options: Optional["ProcessingOptions"] = None):
        """
        Initialize the formatter with processing options.

        Args:
            options: ProcessingOptions flags to control formatting details.
        """
        self.options = options
        # Import locally or under TYPE_CHECKING if needed earlier
        from .process_xbrl import ProcessingOptions

        self._processing_options = ProcessingOptions  # Store for easy access

    def format_document_start(self, doc: "IXBRLDocument") -> str:
        return "--- DOCUMENT START ---\n\n"

    def format_document_end(self, doc: "IXBRLDocument") -> str:
        return "\n--- DOCUMENT END ---"

    def format_section_start(self, section: "DocumentSection") -> str:
        title_upper = section.title.upper()
        separator = "=" * len(title_upper)
        return f"{title_upper}\n{separator}\n\n"

    def format_section_end(self, section: "DocumentSection") -> str:
        return "\n\n"  # Add extra spacing between sections

    def format_text_part(self, part: "DocumentPart") -> str:
        # Simple text with a newline
        return f"{part.content}\n"

    def format_fact_part(self, part: "DocumentPart") -> str:
        # Type hint for clarity
        from xbrl.instance import AbstractFact, TextFact

        fact = cast(AbstractFact, part.content)

        # --- Handle TextFacts based on options ---
        if isinstance(fact, TextFact):
            max_len = DEFAULT_TEXT_FACT_TRUNCATION
            show_full = True  # Default to showing full text
            strip_html = True  # Default to stripping HTML

            if self.options:
                if self.options <= self._processing_options.LEVEL_1:
                    show_full = False
                    strip_html = True
                # else: # Higher levels
                #     show_full = True
                #     strip_html = True

            text_content = fact.value
            if strip_html:
                try:
                    soup = BeautifulSoup(text_content, "html.parser")
                    text_content = soup.get_text(
                        strip=True
                    )  # Simple strip for plain text
                except Exception:
                    pass  # Keep original

            if show_full:
                return f"[TEXT_FACT: Concept={fact.concept.name}]\nValue:\n{text_content}\n[END_TEXT_FACT]\n"
            else:
                truncated_value = (
                    (text_content[:max_len] + "...")
                    if len(text_content) > max_len
                    else text_content
                )
                return f"[TEXT_FACT: Concept={fact.concept.name} Value(truncated)='{truncated_value}']\n"
        # --- Handle other Fact types ---
        else:
            unit_str = ""
            if hasattr(fact, "unit") and fact.unit:
                unit_display = getattr(fact.unit, "id", str(fact.unit))
                unit_str = f" Unit: {unit_display}"

            period_str = ""
            if hasattr(fact, "period") and fact.period:
                if hasattr(fact.period, "instant"):
                    period_str = f" Period: {fact.period.instant}"
                elif hasattr(fact.period, "start_date") and hasattr(
                    fact.period, "end_date"
                ):
                    period_str = (
                        f" Period: {fact.period.start_date} to {fact.period.end_date}"
                    )

            context_str = ""
            if hasattr(fact, "context") and fact.context:
                context_display = getattr(fact.context, "id", str(fact.context))
                context_str = f" Context: {context_display}"

            return f"[FACT: {fact.concept.name}={fact.value}{unit_str}{period_str}{context_str}]\n"
