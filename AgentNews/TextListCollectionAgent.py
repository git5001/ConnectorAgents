from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentFramework.ListCollectionAgent import ListCollectionAgent

from AgentNews.NewsSchema import TextualSummaryOutput, SummaryOutput


class TextListCollectionAgentConfig(BaseToolConfig):
    """
    Configuration for the list collection Agent.
    """
    pass


class TextListCollectionAgent(ListCollectionAgent):
    """
    Agent that extracts structured summaries from search results and webpage content.

    Attributes:
        input_schema (Type[MarkdownSummaryOutput]): Defines the expected input schema.
        output_schema (Type[SinkOutput]): Defines the expected output schema.
    """
    input_schema = TextualSummaryOutput
    output_schema = SummaryOutput

    def __init__(self, config: TextListCollectionAgentConfig) -> None:
        """
        Initializes the Format Sink Agent.

        Args:
            config (TextListCollectionAgentConfig): Configuration for the agent.
        """
        super().__init__(config)

    def run(self, params: TextualSummaryOutput) -> SummaryOutput:
        """
        Runs the agent synchronously to generate a structured news summary.

        Args:
            params (TextualSummaryOutput): Input news summary data.

        Returns:
            SummaryOutput: The formatted markdown summary.
        """
        return SummaryOutput(output=params.news_text)