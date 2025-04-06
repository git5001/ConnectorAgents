# ConnectorAgents

This is an extension to the Atomic agents [Atomic Agents GitHub](https://github.com/BrainBlend-AI/atomic-agents) framework and and example Application for news fetching.

## News Processing Agent Example

### Overview

This project is an extension of an agent-based framework for automated news processing using LLM agents, web scraping, and search tools. The system allows agents to:

- Search internet news
- Fetch news articles from various sources
- Process and summarize content using LLMs
- Structure output in multiple formats
- Email results
- Log messages for debugging and audit purposes.


### Installation

Prerequisites
- Python 3.8+
- pip (Python package manager)

### Execution and Test

The example application is located at
```
    ConnectorAgents/AgentNews/test/test_connected_agents.py
```
It uses the agent framework to set up agents to perform above newsletter generation.

Note: This is not production code but a test example!


Rough setup:

```python
  # Create agents
tavilyConfig = TavilySearchToolConfig(api_key=TAVILY_API_KEY, max_results=AMOUNT, topic='news', days=DAYS)
tavilyAgent = TavilyAgent(config=tavilyConfig)
...


# Message transformer
def transform_tavily_to_webscraper(output_msg: TavilySearchToolOutputSchema) -> List[WebpageScraperToolInputSchema]:
    # Creates N Scrapers!
    return [
        WebpageScraperToolInputSchema(url=result.url, include_links=False)
        for result in output_msg.results
    ]
    ...


# Interconnect agents
tavilyAgent.output_port.connectTo(webScraper.input_port, transform_tavily_to_webscraper)
...

# Schedule agents
scheduler: AgentScheduler = AgentScheduler()
scheduler.add_agent(tavilyAgent)
...

# Start chain
tavilyAgent.feed(TavilySearchToolInputSchema(queries=[TOPIC]))

# Run
while scheduler.step():
    pass


```


### Contributing

We welcome contributions! 


### License

This project is licensed under the MIT License.

### Contact

For inquiries, please open an issue or reach out via GitHub Discussions.
