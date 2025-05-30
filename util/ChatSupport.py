import time
import traceback
from enum import IntEnum
from typing import Optional, Type, Any, Dict, Union, Generic, TypeVar, List

from openai.types import CompletionUsage
from pydantic import BaseModel
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_sleep_log, \
    RetryCallState
import logging
import json5

from agent_logging import logger
from util.LLMSupport import LLMModel, Provider
from util.SchemaUtils import clean_json_string

# Set up logging
logging.basicConfig(level=logging.INFO)
standard_logger = logging.getLogger(__name__)

NANO_GPT_BASE_URL = "https://nano-gpt.com/api/v1"
OPEN_ROUTER_BASE_URL="https://openrouter.ai/api/v1"


# Base enum class with the DEFAULT value
class CallStep(IntEnum):
    DEFAULT = 0
    FIX = -1

    # You can use this to check if a value belongs to any subclass of CallStep
    @classmethod
    def is_valid_step(cls, step):
        # Check if it's an instance of any CallStep subclass
        return isinstance(step, CallStep)

class Prompt:
    def __init__(self, step:CallStep, sysText:str, prompt:str, title:str = None, userMessage:str= None, temperature = None):
        self.step = step
        self.sysText = sysText
        self.prompt = prompt
        self.title = title
        self.userMessage = userMessage
        self.temperature = temperature

        if not title:
            title = "Generic LLM call"


class ChatSupport:
    def __init__(self, models: Dict[CallStep, LLMModel]):
        self.models = models.copy()
        # Delete logs
        for model in models.values():
            model.delete_log_files()

        if CallStep.DEFAULT in self.models:
            # Assign DEFAULT model to any missing CallStep (except DEFAULT itself)
            default_model = self.models[CallStep.DEFAULT]
            enum_type = type(next(iter(self.models.keys())))
            for step in enum_type:
                if step != CallStep.DEFAULT and step not in self.models:
                    self.models[step] = default_model


    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=3),  # Exponential backoff with 1–3 seconds delay
        stop=stop_after_attempt(3),  # Maximum of 3 attempts
        retry=retry_if_exception_type((Exception, ValueError)),  # Retry on specific exceptions
        before_sleep=before_sleep_log(logger, logging.INFO)  # Log attempt details before each retry
    )
    def generate_json_pydantic(self,
                               idata: Prompt,
                               history: List[dict] = None,
                               schema_input: Optional[Union[Dict, type]] = None,
                               schema_name: Optional[str] = None,
                               pydantic_class: Type[BaseModel] = None,
                               retry_state: Optional[RetryCallState] = None) -> BaseModel:
        """
        Generates an updated emotional state for the chatbot based on the conversation history and user input.
        Retries up to 3 times with exponential backoff in case of errors.
        """

        sysText = idata.sysText
        prompt = idata.prompt
        title = idata.title
        user_message = idata.userMessage
        model = self.models[idata.step]
        modelfix = self.models.get(CallStep.FIX, None)

        if schema_input is None:
            schema_input = pydantic_class

        logger.info(f"Schema name {schema_name} {schema_input}")



        # Get attempt number from retry state
        attempt = retry_state.attempt_number if retry_state else 1

        # Prepare messages for the OpenAI API
        if not model.hasSysPrompt():
            messages = [{"role": "system", "content": sysText}] if sysText else []
        else:
            prompt = f"{sysText}\n{prompt}"
            messages = []

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": prompt})
        if user_message:
            logger.info(f"Adding user message to prompt {user_message}")
            messages.append({"role": "user", "content": user_message})

        model.write_llm_prompt(sysText or "", prompt or "")

        if model.config.provider == Provider.DUMMY:
            logger.warn(f"Running DUMMY LLM {schema_input}")
            dummy_result: BaseModel = schema_input.dummy()
            dummy_dict = dummy_result.model_dump()
            dummy_usage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)
            return dummy_result, dummy_dict, dummy_usage


        try:
            logger.info(f"### Attempt {attempt}: Calling external LLM {model.name()} STRICT for {title}")

            # Call the OpenAI API t
            response = model.create_json_completions(messages,
                                                     schema_name,
                                                     schema_input
                                                     )

            # Extract the updated emotional states from the response
            usage = response.usage
            llm_text = response.choices[0].message.content
            logger.info(f"... external LLM data received: {len(llm_text)} characters")
            logger.info(f"OpenAI Usage {usage}")

            # Parse JSON into a Python dictionary
            try:
                llm_text = clean_json_string(llm_text)
                analysis_dict = json5.loads(llm_text)
                logger.info(f"JSON {title} parsed successfully...")
                # We should be correct json here, see if we can also decode it to pydantic. If not we retry all from beginning
                result_object = pydantic_class(**analysis_dict)
                return result_object, analysis_dict, usage
            except Exception as parse_error:
                logger.error(f"Parsing E5444 failed: {parse_error}. Attempting to fix...")
                if not modelfix:
                    logger.error(f"Parsing E5444 failed no FIX model in modeldata...")
                    raise
                # Fix JSON by calling LLM
                fix_prompt = (
                    f"The following JSON is invalid and could not be parsed due to this error: {parse_error}.\n"
                    f"Please correct it while preserving the intent of the data:\n{llm_text}\n"
                    f"Output must be a formally correct prettfied JSON. No other text before or after the JSON"
                    f"payload must be included. Do never change any values. The target format is:\n"
                    f"{schema_input}\n"
                    f"---\n"
                    f"If there are missing keys, set them to empty strings. But never touch any existing values."

                )
                logger.info(f"JSON fix prompt {fix_prompt}")
                fix_messages = [{"role": "user", "content": fix_prompt}]
                fix_response = modelfix.create_completions(messages=fix_messages)
                usage = fix_response.usage
                logger.info(f"Fixed Usage {usage}")
                fixed_json = fix_response.choices[0].message.content
                fixed_json = clean_json_string(fixed_json)
                try:
                    fixed_dict = json5.loads(clean_json_string(fixed_json))
                    logger.info("Fixed JSON parsed successfully...")
                except Exception as fix_error:
                    logger.error(f"Fixing attempt failed: {fix_error}")
                    raise ValueError(
                        "Fixing attempt also failed E1559. Original error: {parse_error}, fix error {fix_error}")

                # We should be correct json here, see if we can also decode it to pydantic. If not we retry all from beginning
                result_object = pydantic_class(**fixed_dict)
                return result_object, fixed_dict, usage
            except Exception as e:
                error_message = f"JSON '{title}' parsing error E1999: {e}. LLM returned:\n{llm_text}"
                logger.error(error_message)
                raise ValueError(error_message)  # Raise error to trigger retry

        except Exception as e:
            error_message = f"Error during LLM call '{title}' or processing E6234: {e}"
            logger.error(error_message)
            # print stack trace
            traceback.print_exc()
            raise ValueError(error_message)  # Raise error to trigger retry if possible

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=3),  # Exponential backoff with 1–3 seconds delay
        stop=stop_after_attempt(3),  # Maximum of 3 attempts
        retry=retry_if_exception_type((Exception, ValueError)),  # Retry on specific exceptions
        before_sleep=before_sleep_log(logger, logging.INFO)  # Log attempt details before each retry
    )
    def generate_chat(self,
                      idata: Prompt,
                      history: List[dict] = None,
                      retry_state: Optional[RetryCallState] = None):
        """
        Generate chat response
        """

        sysText = idata.sysText
        prompt = idata.prompt
        title = idata.title
        user_message = idata.userMessage
        model = self.models[idata.step]

        # Assume these are defined: sysText, prompt, user_message, history
        messages = []

        # 1. Optional system prompt
        if sysText:
            messages.append({"role": "system", "content": sysText})

        # 2. Insert previous history (if any)
        if history:
            messages.extend(history)

        # 3. Main user prompt
        if prompt:
            messages.append({"role": "user", "content": prompt})

        # 4. Additional user message
        if user_message:
            logger.info(f"Adding user message to prompt: {user_message}")
            messages.append({"role": "user", "content": user_message})

        try:
            # Call the OpenAI API to analyze emotional states
            start_time = time.time()
            response =  model.create_completions(messages=messages, temperature=idata.temperature)
            elapsed_time = time.time() - start_time
            logger.info(f"### generate_chat time is [{elapsed_time:.1f}]:\n")
            # Extract the updated emotional states from the response
            llmMessage = response.choices[0].message
            llm_text = llmMessage.content
            usage = response.usage

            logger.info("AI generate_chat a text ...")
            return llm_text, usage

        except Exception as e:
            logger.error(f"Error generating emotional states: {e}")
            return None, None  # Return the original state if an error occurs