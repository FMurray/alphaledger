from textual.app import App, ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Header, Footer, Input, Static
from alphaledger.models import AzureOpenAIChat


class AssistantApp(App):
    """A Textual app to explain financial terms using an AI."""

    CSS = """
    Screen {
        align: center top; /* Align content to the top center */
        padding: 1;
    }

    #term-input {
        width: 80%;
        max-width: 100;
        margin: 1 0; /* Add some vertical margin */
    }

    #explanation-output {
        width: 80%;
        max-width: 100;
        height: 15; /* Give it a fixed height initially */
        border: round $accent;
        padding: 1;
        margin: 1 0;
    }

    #explanation-content { /* Style for the Static within VerticalScroll if needed */
        /* Example:  width: 100%; */
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = AzureOpenAIChat()

    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("q", "quit", "Quit"),
    ]

    response = ""

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield Input(
            placeholder="Enter financial term (e.g., us-gaap:Assets)", id="term-input"
        )
        with VerticalScroll(id="explanation-output"):
            yield Static(self.response, id="explanation-content")
        yield Footer()

    def on_input_submitted(self, message: Input.Submitted) -> None:
        """Handle the submission of the input field."""
        term = message.value
        content_widget = self.query_one("#explanation-content", Static)

        if term:
            content_widget.update("")  # Clear previous output
            accumulated_response = ""
            try:
                stream = self.model.get_chat_completion(term, stream=True)
                for chunk in stream:
                    if (
                        chunk.choices
                        and chunk.choices[0].delta
                        and chunk.choices[0].delta.content
                    ):
                        content_piece = chunk.choices[0].delta.content
                        accumulated_response += content_piece
                        content_widget.update(accumulated_response)
                if not accumulated_response:
                    content_widget.update("No response generated.")
            except Exception as e:
                content_widget.update(f"Error streaming response: {e}")

            self.query_one("#term-input", Input).value = ""
        else:
            content_widget.update("Please enter a term.")


if __name__ == "__main__":
    app = AssistantApp()
    app.run()
