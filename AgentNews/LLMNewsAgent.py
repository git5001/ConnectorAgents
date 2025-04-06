from typing import Optional

import instructor
from openai import OpenAI
from pydantic import Field

from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseToolConfig
from atomic_agents.lib.components.agent_memory import AgentMemory
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator

from AgentFramework.ConnectedAgent import ConnectedAgent

from AgentNews.NewsSchema import LLMNewsInput, LLMNewsOutput
from agent_config import DUMMY_LLM


class LLMNewsAgentConfig(BaseToolConfig):
    """
    Configuration class for LLMAgent, defining model parameters and API key.

    Attributes:
        model (str): The model used for generating responses.
        api_key (str): The API key for OpenAI.
    """
    model: str = Field(..., description="The model to use for generating responses.")
    api_key: str = Field(..., description="The API key for OpenAI.")


class LLMNewsAgent(BaseAgent, ConnectedAgent):
    """
    An agent that calls OpenAI's LLMs to summarize and condense news articles.

    Attributes:
        input_schema (Type[BaseIOSchema]): Expected input schema.
        output_schema (Type[BaseIOSchema]): Expected output schema.
    """
    input_schema = LLMNewsInput
    output_schema = LLMNewsOutput

    def __init__(self, config: LLMNewsAgentConfig) -> None:
        """
        Initializes an LLMAgent instance with OpenAI API configuration.

        Args:
            config (LLMAgentConfig, optional): Configuration for the agent. Defaults to LLMAgentConfig().
        """
        ConnectedAgent.__init__(self, config)  # Explicitly call ConnectedAgent

        openai_client = OpenAI(api_key=config.api_key, base_url=None)
        client = instructor.from_openai(
            openai_client,
            mode=instructor.Mode.JSON,
        )

        # Set up the system prompt
        system_prompt_generator = SystemPromptGenerator(
            background=["This assistant is an expert news agglomerator."],
            steps=[
                "Analyze the news article to understand the context and intent.",
                "Extract a title, a short summary headline, a few keywords for a news category, ",
                "and an extensive summary."
            ],
            output_instructions=[
                "Provide clear and concise information about the provided news article.",
                "Output must be correct JSON."
            ]
        )

        agent_config = BaseAgentConfig(
            client=client,
            model=config.model,
            system_prompt_generator=system_prompt_generator,
            memory=AgentMemory(),
            output_schema=LLMNewsOutput
        )

        BaseAgent.__init__(self, agent_config)  # Explicitly call BaseAgent

    def run(self, user_input: Optional[BaseIOSchema] = None) -> BaseIOSchema:
        """
        Processes the user input and returns a structured summary.
        If `DUMMY_LLM` is enabled, returns dummy data.

        Args:
            user_input (Optional[BaseIOSchema], optional): The input data. Defaults to None.

        Returns:
            BaseIOSchema: The processed response from the LLM.
        """
        if DUMMY_LLM:
            return LLMNewsOutput(news_title="Dummy", keywords=[], news_abstract="Dummy abstract", news_list=[],
                                 news_content="Dummy content")
        else:
            return super().run(user_input)
