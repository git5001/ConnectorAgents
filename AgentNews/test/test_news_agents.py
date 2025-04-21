import os
from typing import List
from dotenv import load_dotenv
from rich.console import Console

from AgentFramework.ListCollectionAgent import ListModel
from AgentFramework.PiplinePrinter import PipelinePrinter
from AtomicTools.tavily_search.tool.tavily_search import TavilySearchToolConfig, TavilySearchToolOutputSchema, TavilySearchToolInputSchema
from AtomicTools.webpage_scraper.tool.webpage_scraper import WebpageScraperToolConfig, WebpageScraperToolInputSchema, WebpageScraperToolOutputSchema

from AgentFramework.AgentScheduler import AgentScheduler

from AgentFramework.DebugAgent import DebugAgentConfig, DebugAgent
from AgentNews.EmailAgent import EmailAgentConfig, EmailAgent, EmailMessageInput
from AgentNews.NewsToTextAgent import NewsToTextAgent, NewsToTextAgentConfig
from AgentNews.TextListCollectionAgent import TextListCollectionAgentConfig, TextListCollectionAgent
from AgentNews.LLMNewsAgent import LLMNewsAgent, LLMNewsAgentConfig
from AgentNews.NewsMultiPortAggregatorAgent import NewsMultiPortAggregatorAgent, NewsMultiPortAggregatorAgentConfig
from AgentNews.NewsSchema import  LLMNewsInput, SummaryOutput
from AgentFramework.PrintAgent import PrintAgentConfig, PrintAgent, PrintMessageInput
from AgentNews.TaviliyAgent import TavilyAgent
from AgentNews.WebScraperAgent import WebScraperAgent
from agent_logging import rich_console


def main():

    ########################################################################
    # Search configuration
    ########################################################################
    AMOUNT = 3                       # How many news to search and process
    DAYS = 10                        # How many days should the searfch look back
    TOPIC = "All about openai GPT-5" # Seach topic
    LLM_MODEL = "gpt-4o-mini"  # LLM to use

    ########################################################################
    # Secret configration
    ########################################################################
    # Set secrets
    # .env File must contain the following entries:
    # SENDER="agent@gmail.com"
    # EMAIL_PASSWORD="secret"
    # SMTP_SERVER="..."
    # EMAIL="myemail@gmail.com"
    # OPENAI_API_KEY="..."
    # TAVILY_API_KEY="..."
    load_dotenv()
    SENDER = os.getenv("SENDER")                  # Sender email
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")  # Sender email accoutn password
    SMTP_SERVER = os.getenv("SMTP_SERVER")        # SMTP server address
    EMAIL = os.getenv("EMAIL")                    # Receiver email where the news is send to
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # OpenAi API KEY
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # Taviliy API_KEY
    # Debug print the ENV file
    # print("EMAIL:", EMAIL)
    # print("EMAIL_PASSWORD:", EMAIL_PASSWORD)
    # print("SMTP_SERVER:", SMTP_SERVER)
    # print("SENDER:", SENDER)
    # print("OPENAI_API_KEY:", OPENAI_API_KEY)
    # print("TAVILY_API_KEY:", TAVILY_API_KEY)
    ########################################################################


    # Tavily
    tavilyConfig = TavilySearchToolConfig(api_key=TAVILY_API_KEY, max_results=AMOUNT, topic='news', days=DAYS)
    tavilyAgent = TavilyAgent(config=tavilyConfig)

    # Web scraping
    webScraperConfig = WebpageScraperToolConfig()
    webScraper = WebScraperAgent(config=webScraperConfig)

    # LLM
    llmConfig = LLMNewsAgentConfig(api_key=OPENAI_API_KEY, model=LLM_MODEL)
    llmAgent = LLMNewsAgent(config=llmConfig)

    # Merging agent (Tavility + Scraper + LLM -> Summary item structure)
    mergingAgent = NewsMultiPortAggregatorAgent(NewsMultiPortAggregatorAgentConfig())

    # Formatting agent, Converts Summmary structure to readable text (html / markdown)
    textMakerAgent1:NewsToTextAgent = NewsToTextAgent(NewsToTextAgentConfig(output_format='html'))
    textMakerAgent2:NewsToTextAgent = NewsToTextAgent(NewsToTextAgentConfig(output_format='markdown'))

    # Sink (Collect N new text summaries as one )
    sinkAgent1:TextListCollectionAgent = TextListCollectionAgent(TextListCollectionAgentConfig())
    sinkAgent2:TextListCollectionAgent = TextListCollectionAgent(TextListCollectionAgentConfig())

    # Email
    email_config = EmailAgentConfig(smtp_server=SMTP_SERVER, smtp_port=465,sender_email=SENDER, password=EMAIL_PASSWORD, html=True)
    emailAgent = EmailAgent(email_config)

    # Logger
    printAgent = PrintAgent(PrintAgentConfig())

    # Debug
    debugAgent1 = DebugAgent(DebugAgentConfig())

    # ------------------------------------------------------------------------------------------------
    # Message transformation
    def transform_tavily_to_webscraper(output_msg: TavilySearchToolOutputSchema) -> List[WebpageScraperToolInputSchema]:
        """Converts each search result into an individual WebpageScraperToolInputSchema instance."""
        # Creates N Scrapers!
        return [
            WebpageScraperToolInputSchema(url=result.url, include_links=False)
            for result in output_msg.results
        ]

    def transform_webscraper_to_llm(output_msg: WebpageScraperToolOutputSchema) -> LLMNewsInput:
        """Converts each search result into an individual TavilySearchToolOutputSchema instance."""
        return LLMNewsInput(news_title=output_msg.metadata.title, news_content=output_msg.content)

    def transform_tavily_to_merger(output_msg: TavilySearchToolOutputSchema) -> List[TavilySearchToolOutputSchema]:
        """Converts each search result into an individual TavilySearchToolOutputSchema instance."""
        # Extract list of results from message. Merger combines them with corresponding inputs from other inputs.
        # (Returning a list here is a bit counterintuitive and one might have to change this behaviour)
        return output_msg.results

    def transform_summaries_to_email(output_msg: ListModel) -> EmailMessageInput:
        """Converts each search result into an individual TavilySearchToolOutputSchema instance.
        SummaryOutput """
        data:List[SummaryOutput] = output_msg.data
        return EmailMessageInput(
            recipient_email=EMAIL,
            subject=TOPIC,
            body="\n".join(om.output for om in data)
        )

    def transform_summaries_to_print(output_msg: ListModel) -> PrintMessageInput:
        """Converts each search result into an individual TavilySearchToolOutputSchema instance.
        SummaryOutput"""
        data: List[SummaryOutput] = output_msg.data
        return PrintMessageInput(
            subject=TOPIC,
            body="\n".join(om.output for om in data)
        )
    # ------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------
    # Connect agents
    tavilyAgent.connectTo(webScraper, transform_tavily_to_webscraper)

    webScraper.connectTo(llmAgent, transform_webscraper_to_llm)

    tavilyAgent.connectTo(mergingAgent,  transform_tavily_to_merger, target_port_name="search_result")
    webScraper.connectTo(mergingAgent, target_port_name="web_scraping_result")
    llmAgent.connectTo(mergingAgent, target_port_name="llm_result")

    mergingAgent.connectTo(textMakerAgent1)
    mergingAgent.connectTo(textMakerAgent2)

    textMakerAgent1.connectTo(sinkAgent1)
    textMakerAgent2.connectTo(sinkAgent2)

    textMakerAgent2.connectTo(debugAgent1)
    sinkAgent2.connectTo(debugAgent1)

    sinkAgent1.connectTo(emailAgent, transform_summaries_to_email)
    sinkAgent2.connectTo(printAgent, transform_summaries_to_print)
    # ------------------------------------------------------------------------------------------------



    # ------------------------------------------------------------------------------------------------
    # Scheduler
    scheduler:AgentScheduler = AgentScheduler()
    scheduler.add_agent(tavilyAgent)
    scheduler.add_agent(webScraper)
    scheduler.add_agent(llmAgent)
    scheduler.add_agent(mergingAgent)
    scheduler.add_agent(textMakerAgent1)
    scheduler.add_agent(textMakerAgent2)
    scheduler.add_agent(sinkAgent1)
    scheduler.add_agent(sinkAgent2)
    scheduler.add_agent(printAgent)
    scheduler.add_agent(emailAgent)

    scheduler.add_agent(debugAgent1)
    # ------------------------------------------------------------------------------------------------
    # Print pipeline
    printer = PipelinePrinter(is_ortho=True, direction='TB', fillcolor='blue')
    printer.print_ascii(scheduler.agents)
    printer.to_png(scheduler.agents,  'r:/pipeline_news.png')
    printer = PipelinePrinter(is_ortho=True, direction='TB', fillcolor='blue', show_schemas = True, schema_fillcolor = 'moccasin')
    printer.print_ascii(scheduler.agents)
    printer.to_png(scheduler.agents,  'r:/pipeline_news_large.png')
    #return



    # ------------------------------------------------------------------------------------------------
    # Start processing
    rich_console.print(f"[red]Feeding agent {tavilyAgent.__class__.__name__}[/red]: '{TOPIC}'")
    tavilyAgent.feed(TavilySearchToolInputSchema(queries=[TOPIC]))


    # ------------------------------------------------------------------------------------------------
    # Loop all
    scheduler.step_all()

    # ------------------------------------------------------------------------------------------------
    # Final output
    results = scheduler.get_final_outputs()
    rich_console.print(f"[red]Final results[/red]",results)
    rich_console.print("Ready.")


if __name__ == '__main__':
    main()