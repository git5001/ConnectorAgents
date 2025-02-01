from pydantic import Field

from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentFramework.ConnectedAgent import ConnectedAgent

from AgentNews.NewsSchema import TextualSummaryOutput, TextualSummaryInput, MergedOutput


class NewsToTextAgentConfig(BaseToolConfig):
    """Configuration for summary Agent."""
    output_format: str = Field("markdown", description="Output format: 'text', 'html', or 'markdown'.")



class NewsToTextAgent(ConnectedAgent):
    """
    News Summary Agent extracts structured summaries from search results and webpage content.
    """
    input_schema = MergedOutput
    output_schema = TextualSummaryOutput

    def __init__(self, config: NewsToTextAgentConfig):
        """
        Initializes the News Summary Agent.

        Args:
            config (NewsAgentConfig): Configuration for the agent.
        """
        super().__init__(config)
        self.output_format = config.output_format

    def summarize_news(self, news_data: MergedOutput) -> TextualSummaryOutput:
        """
        Processes news data and generates a formatted Markdown summary.

        Args:
            news_data (MergedOutput): Contains titles, keywords, abstracts, and summary information.

        Returns:
            TextualSummaryOutput: Formatted markdown string containing the summary.
        """


        if self.output_format == "markdown":
            summary_text = f"""# {news_data.news_title}
## Categories:
{', '.join(news_data.keywords)}
## Short Abstract:
{news_data.news_abstract}
## Key Points:
- {', '.join(news_data.news_list)}
## Content Summary:
{news_data.news_content}
[Read more]({news_data.url})
    """
        elif self.output_format == "html":
            summary_text = f"""
                    <html>
                        <head><title>{news_data.news_title}</title></head>
                        <body>
                        <h2>{news_data.news_title}</h2>
                        <p><strong>Categories:</strong> {', '.join(news_data.keywords)}</p>
                        <p><em>{news_data.news_abstract}</em></p>
                        <ul>
                            {''.join(f'<li>{point}</li>' for point in news_data.news_list)}
                        </ul>
                        <p>&nbsp;</p>
                        <p>{news_data.news_content}</p>
                        <p><a href="{news_data.url}">Read more</a></p>
                    </body>
                    </html>
                    """
        elif self.output_format == "text":
            summary_text = f"""Title: {news_data.news_title}
    Categories: {', '.join(news_data.keywords)}
    Short Abstract: {news_data.news_abstract}
    Key Points:
    - {', '.join(news_data.news_list)}
    Content Summary: {news_data.news_content}
    Read more: {news_data.url}
    """
        else:
            raise ValueError("Invalid output format. Choose 'text', 'html', or 'markdown'.")

        return TextualSummaryOutput(news_text=summary_text)


    def run(self, params: MergedOutput) -> TextualSummaryOutput:
        """
        Runs the agent synchronously to generate a structured news summary.

        Args:
            params (TextualSummaryInput): Input news data.

        Returns:
            TextualSummaryOutput: The formatted markdown summary.
        """
        return self.summarize_news(params)

