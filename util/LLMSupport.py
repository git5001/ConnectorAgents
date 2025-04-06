import os
from enum import Enum
from typing import Optional, List, Dict, Any, Type, Tuple, Callable

import json5
from atomic_agents.lib.base.base_tool import BaseToolConfig
from openai import OpenAI, NOT_GIVEN, NotGiven
import logging

from openai.types import CompletionUsage
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_sleep_log, RetryCallState

from util.SchemaUtils import SchemaUtils
from util.SchemaUtils import generate_template_json, clean_json_string


# Get a named logger (inherits root config if no specific config)
logger = logging.getLogger(__name__)


NANO_GPT_BASE_URL = "https://nano-gpt.com/api/v1"
OPEN_ROUTER_BASE_URL="https://openrouter.ai/api/v1"

class Provider(Enum):
    OPENAI = "OPENAI"
    NANOGPT = "NANOGPT"
    OLLAMA = "OLLAMA"
    OPENROUTER = "OPENROUTER"

class LLMAgentConfig(BaseToolConfig):
    """
    Configuration class for LLMAgent, defining model parameters and API key.

    Attributes:
        model (str): The model used for generating responses.
        api_key (str): The API key for OpenAI.
    """
    model: str = Field(..., description="The model to use for generating responses.")
    provider: Provider = Field(..., description="The model provider, e.g. OpenAI")
    api_key: Optional[str] = Field(None, description="The API key for the provider.")
    base_url: Optional[str] = Field(None, description="base url")
    log_dir: Optional[str] = Field(None, description="log dir")
    max_token: Optional[int] = Field(None, description="max token")
    timeout: Optional[int] = Field(None, description="llm timeout in sec")
    use_response: Optional[bool] = Field(None, description="Utilize repsonse format if avaialble")




class LLMModel:
    def __init__(
        self,
        config:LLMAgentConfig
    ):
        """
        Initialize the LLMModel.

        Args:
            model (str): The name of the model to use (e.g., "gpt-3.5-turbo").
            provider (Provider): The LLM provider.
            api_key (Optional[str], optional): The API key to use. Defaults to None.
        """
        self.config = config

        if config.provider == Provider.NANOGPT:
            self.base_url = NANO_GPT_BASE_URL
            self._maxToken = 65536    # Deepseek 64K, Max 8K output
            self._hasSysPrompt = False
            self._hasResponseFormat = False
        elif config.provider == Provider.OPENROUTER:
            self.base_url = OPEN_ROUTER_BASE_URL
            self._maxToken = 65536    # Deepseek 64K, Max 8K output
            self._hasSysPrompt = False
            self._hasResponseFormat = False
        elif config.provider == Provider.OPENAI:
            self.base_url = None
            self._maxToken = NOT_GIVEN
            self._hasSysPrompt = True
            self._hasResponseFormat = True
        elif config.provider == Provider.OLLAMA:
            self.base_url = config.base_url
            self._maxToken = NOT_GIVEN
            self._hasSysPrompt = False
            self._hasResponseFormat = False
            api_key = "ollama"
        else:
            raise NotImplementedError(f"Provide {config.provider} is not implemented.")
        if config.use_response:
            self._hasResponseFormat = True
        if config.max_token is not None:
            self._max_token = config.max_token
        if config.base_url:
            self.base_url = config.base_url

        if config.max_token is not None:
            if self._maxToken == NOT_GIVEN:
                self._maxToken = config.max_token
            else:
                self._maxToken = min(config.max_token, self._maxToken)

        self.log_dir = config.log_dir
        if not api_key:
            api_key = ""

        self.client =  OpenAI(api_key=api_key, base_url=self.base_url)
        self._model = config.model
        self._provider = config.provider

    def setMaxToken(self,max_token):
        self._maxToken = max_token

    def name(self):
        return self._model

    def model(self):
        return self._model

    def hasSysPrompt(self):
        return self._hasSysPrompt

    def create_completions(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None
    ) -> Any:
        """
        Call the OpenAI Chat Completion API with the provided messages.

        Args:
            messages (List[Dict[str, str]]): A list of messages formatted as dictionaries.
                For example: [{"role": "system", "content": sysText}] if a system prompt is desired.
            temperature (Optional[float]): The sampling temperature for the model.

        Returns:
            Any: The API response from openai.ChatCompletion.create.
        """

        if not temperature:
            temperature = NOT_GIVEN

        logger.info(f"Calling model {self.model()} chat completions token={self._maxToken} temperature={temperature}")
        response = self.client.chat.completions.create(model=self._model,
                                                       messages=messages,
                                                       temperature=temperature,
                                                       max_tokens=self._maxToken,
                                                       )
        #logger.debug("LLM: Resopnse " ,response)

        return response

    @classmethod
    def openai_schema(cls, type_: Type):
        schema_dict = type_.model_json_schema()
        schema_dict = SchemaUtils.inline_all_references(schema_dict)
        schema_dict = SchemaUtils.enforce_additional_properties_false(schema_dict)
        # Remove `$defs` from the schema (no longer needed after inlining)
        schema_dict.pop("$defs", None)
        return schema_dict

    def create_json_completions(
        self,
        messages: List[Dict[str, str]],
        schema_name: str,
        schema_dict: Dict,
        temperature: Optional[float] = None,
        timeout: float | None | NotGiven = NOT_GIVEN,
    ) -> Any:
        """
        Call the OpenAI Chat Completion API and instruct the model to follow a JSON schema.

        This method appends a system prompt that instructs the model to output its response
        in JSON format following the provided schema.

        Args:
            messages (List[Dict[str, str]]): A list of messages to be sent to the model.
            schema_name (str): The name of the JSON schema.
            schema_dict (Dict): A dictionary representing the schema.
            temperature (Optional[float]): The sampling temperature for the model.

        Returns:
            Any: The API response from openai.ChatCompletion.create.
        """

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": f"{schema_name}",
                "strict": True,
                "schema": schema_dict
            }
        }
        if not self._hasResponseFormat:
            response_format = NOT_GIVEN
        if temperature is None:
            temperature = NOT_GIVEN


        logger.info(f"Calling model {self.model()} json strict token={self._maxToken} temperature={temperature}")
        response = self.client.chat.completions.create(model=self._model,
                                                       messages=messages,
                                                       response_format=response_format,
                                                       temperature=temperature,
                                                       max_tokens=self._maxToken,
                                                       timeout=timeout
                                                       )
        #print("rrrr",response)
        logger.info("Done LLM call")

        # Extract the updated emotional states from the response
        usage = response.usage
        llm_text = response.choices[0].message.content
        logger.info(f"... external LLM data received: {len(llm_text)} characters")
        logger.info(f"LLM Usage {usage}")


        return response

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=3),  # Exponential backoff with 1–3 seconds delay
        stop=stop_after_attempt(5),  # Maximum of attempts
        retry=retry_if_exception_type((Exception, ValueError)),  # Retry on specific exceptions
        before_sleep=before_sleep_log(logger, logging.INFO)  # Log attempt details before each retry
    )
    def execute_llm_schema(self,sys_prompt:str,
                           user_prompt:str,
                           targetType: Type[BaseModel],
                           title='Default LLM call',
                           fix_function: Optional[Callable[[str], str]] = None,
                     retry_state: Optional[RetryCallState] = None) -> Tuple[BaseModel, CompletionUsage]:

        # Get attempt number from retry state
        attempt = retry_state.attempt_number if retry_state else 1

        schema_dict = LLMModel.openai_schema(targetType)

        # Prepare messages for the OpenAI API
        sysprompt = sys_prompt
        if self.hasSysPrompt():
            logger.warning(f"System prompt supported for model {self.name()}")
            messages = [{"role": "system", "content": sysprompt}] if sysprompt else []
            messages.append({"role": "user", "content": user_prompt})
        else:
            logger.warning(f"Merging system prompt in prompt for model {self.name()}")
            messages = [{"role": "user", "content": f"{sysprompt}\n{user_prompt}"}]
            logger.debug(f"Prompt is: {sysprompt}\n{user_prompt}")

        try:
            logger.info(f"### Attempt {attempt}: Calling external LLM for {title}")
            response = self.create_json_completions(messages, targetType.__class__.__name__, schema_dict, 0.0, timeout=self.config.timeout)
            usage = response.usage
            llm_text = response.choices[0].message.content
            llm_text = clean_json_string(llm_text)
            if fix_function:
                llm_text = fix_function(llm_text)
            self.write_llm_log("llm.out",llm_text)
            try:
                analysis_dict = json5.loads(llm_text)
                # print("Analysis ", analysis_dict)
                result_object = targetType(**analysis_dict)
                print("-> xxResult pydantic", result_object)
            except  Exception as e:
                print("-> Result for error was",llm_text)
                self.write_llm_log("llm.err", llm_text)
                logger.error(f"Json or Pydantic error {e}")
                llm_schema = generate_template_json(schema_dict)
                prompt = f"""
                You are a JSON generator. Your task is to produce a **strictly valid JSON** object based on the input data and the provided schema.

                - Do NOT include any extra commentary, markdown formatting, or quotes around the JSON.
                - The output must exactly match the schema structure and data types.

                ### Input Data:
                {llm_text}

                ### JSON Schema:
                {llm_schema}

                Generate the corrected and valid JSON now:
                """

                messages = [{"role": "user", "content": prompt}]
                response = self.create_json_completions(messages, targetType.__class__.__name__, schema_dict, 0.0, timeout=self.config.timeout)
                llm_text = response.choices[0].message.content
                llm_text = clean_json_string(llm_text)
                if fix_function:
                    llm_text = fix_function(llm_text)
                self.write_llm_log("llm.out", llm_text)
                try:
                    analysis_dict = json5.loads(llm_text)
                    result_object = targetType(**analysis_dict)
                    print("-> yyResult pydantic", result_object)
                except  Exception as e:
                    print("-> Result for error with correction was", llm_text)
                    self.write_llm_log("llm.err", llm_text)
                    logger.error(f"Json or Pydantic error again {e}")
                    raise
        except Exception as e:
            error_message = f"Error during LLM call '{title}' or processing E6236: {e}"
            logger.error(error_message)
            raise ValueError(error_message)  # Raise error to trigger retry if possible

        return result_object, usage



    def write_llm_log(self, filename:str, llm_text: str):
        if not self.log_dir:
            return
        os.makedirs(self.log_dir, exist_ok=True)  # Ensure the log directory exists
        log_path = os.path.join(self.log_dir, filename)
        with open(log_path, "a", encoding="utf-8") as f:  # "a" mode appends to the file
            f.write(llm_text + "\n")


