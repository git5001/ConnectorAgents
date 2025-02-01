from AtomicTools.tavily_search.tool.tavily_search import TavilySearchToolConfig, TavilySearchToolOutputSchema, TavilySearchTool, TavilySearchToolInputSchema

from AgentFramework.ConnectedAgent import ConnectedAgent


class TavilyAgent(TavilySearchTool, ConnectedAgent):
    """
    Agent for retrieving web search results using the Tavily service.

    This agent integrates the Tavily search tool into the ConnectedAgent framework,
    enabling seamless message-based processing.

    Attributes:
        input_schema (Type[BaseModel]): Defines the expected input schema for search queries.
        output_schema (Type[BaseModel]): Defines the expected output schema containing search results.
    """
    input_schema = TavilySearchToolInputSchema
    output_schema = TavilySearchToolOutputSchema

    def __init__(self, config: TavilySearchToolConfig = TavilySearchToolConfig()) -> None:
        """
        Initializes the TavilyAgent with the given configuration.

        Args:
            config (TavilySearchToolConfig, optional): Configuration settings for the Tavily search tool. Defaults to TavilySearchToolConfig().
        """
        ConnectedAgent.__init__(self, config)  # Explicitly call ConnectedAgent
        TavilySearchTool.__init__(self, config)  # Explicitly call TavilySearchTool
