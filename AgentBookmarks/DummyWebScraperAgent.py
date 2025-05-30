import random

from AgentFramework.core.ConnectedAgent import ConnectedAgent

from AtomicTools.webpage_scraper.tool.webpage_scraper import WebpageScraperToolInputSchema, \
    WebpageScraperToolOutputSchema, WebpageScraperToolConfig, WebpageMetadata


class DummyWebScraperAgent(ConnectedAgent):
    """
    An agent that integrates the WebpageScraperTool with the ConnectedAgent framework.
    This agent allows for automated web scraping while maintaining compatibility with the
    interconnected agent system.

    Attributes:
        input_schema (Type[WebpageScraperToolInputSchema]): Defines the expected input schema.
        output_schema (Type[WebpageScraperToolOutputSchema]): Defines the expected output schema.
    """
    input_schema = WebpageScraperToolInputSchema
    output_schema = WebpageScraperToolOutputSchema

    def __init__(self, config: WebpageScraperToolConfig = WebpageScraperToolConfig()) -> None:
        """
        Initializes the WebScraperAgent by integrating both ConnectedAgent and WebpageScraperTool functionalities.

        Args:
            config (WebpageScraperToolConfig, optional): Configuration for the web scraper. Defaults to WebpageScraperToolConfig().
        """
        ConnectedAgent.__init__(self, config)  # Explicitly call ConnectedAgent

    def run(self, params: WebpageScraperToolInputSchema) -> WebpageScraperToolOutputSchema:
        """
        Runs the debug agent, logging received data but not producing any output.

        Args:
            params (BaseModel): Input data received from the connected output port.

        Returns:
            NullSchema: Always returns a NullSchema to indicate no output.
        """


        metadata: WebpageMetadata = WebpageMetadata(title = "Dummy title",
                                                    author = "Author",
                                                    description="Dummy description",
                                                    site_name="Dummy Site Name",
                                                    domain="Dummy Domain Name")


        random_number = random.randint(0, 10)

        return WebpageScraperToolOutputSchema(
            content = f"This is dummy webpage content {params.url}",
            error = None if random_number > 0 else "Random web error",
            metadata = metadata,

        )