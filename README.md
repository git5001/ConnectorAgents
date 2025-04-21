# ConnectorAgents

This is an extension to the Atomic agents [Atomic Agents GitHub](https://github.com/BrainBlend-AI/atomic-agents) framework.

### Overview

This project is an extension of an agent-based framework which includes agent connection, connection visualization and a scheduler.

### Examples
#### Automated News Processing 
Automated news processing using LLM agents, web scraping, and search tools. The system allows agents to:

- Search internet news and fetch webpages
- Process and summarize content using LLMs
- Structure output in multiple formats, like Email or text

#### Condition Logic 
Shows just conditional port connections
- Create simple counter
- Conditional port allows only some numbers to pass


#### Bookmark Organizer
Use LLM to get your bookmarks organzied

- Check if bookmarks are still online 
- Use webscrping to get information about the page
- Use LLM to make a new bookmark folder

### Installation

Prerequisites
- Python 3.8+
- pip (Python package manager)

### Execution and Test

Example applications are located at
```
    ConnectorAgents/AgentNews/test/test_*.py
    ConnectorAgents/AgentBookmarks/test/test_*.py
```

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
tavilyAgent.connectTo(webScraper, transformer=transform_tavily_to_webscraper)
...

# Schedule agents
scheduler: AgentScheduler = AgentScheduler()
scheduler.add_agent(tavilyAgent)
...

# Start chain
tavilyAgent.feed(TavilySearchToolInputSchema(queries=[TOPIC]))

# Run
scheduler.step_all()

```

### Visualization
Exmample pipepine:
![News Pipeline](https://github.com/git5001/ConnectorAgents/blob/main/doc/pic/pipeline_news_large.png)



### Contributing

We welcome contributions! 


### License

This project is licensed under the MIT License.

### Contact

For inquiries, please open an issue or reach out via GitHub Discussions.
