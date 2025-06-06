from AgentFramework.core.ConnectedAgent import ConnectedAgent

from AtomicTools.webpage_scraper.tool.webpage_scraper import WebpageScraperTool, WebpageScraperToolInputSchema, \
    WebpageScraperToolOutputSchema, WebpageScraperToolConfig, WebpageMetadata
from agent_logging import logger


class WebScraperAgent(WebpageScraperTool, ConnectedAgent):
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
        WebpageScraperTool.__init__(self, config)  # Explicitly call WebpageScraperTool


    def run(self, params: WebpageScraperToolInputSchema) -> WebpageScraperToolOutputSchema:
        if params.url is None:
            logger.warn("No url provided")
            return WebpageScraperToolOutputSchema(
                content = "",
                metadata=WebpageMetadata(title = "",author = "",description ="", site_name = "", domain = ""),
                error = "No a valid URL")
        else:
            return super().run(params)
