import os

from dotenv import load_dotenv

from AgentFramework.core.AgentScheduler import AgentScheduler
from AgentFramework.support.CallbackAgent import CallbackAgent
from AgentFramework.support.DebugAgent import DebugAgentConfig, DebugAgent
from AgentFramework.core.PiplinePrinter import PipelinePrinter
from AgentLLM.LLMAgent import LLMAgent
from AgentLLM.test.TestModels import CheckAgent
from agent_logging import rich_console
from util.LLMSupport import LLMRequest, LLMAgentConfig, Provider


def main():
    ########################################################################
    # Secret configration
    ########################################################################
    STRUCTURED_RESPONSE = True
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # OpenAi API KEY
    llmOpenAiConfig_small = LLMAgentConfig(model="gpt-4.1-nano",
                               provider=Provider.OPENAI,
                               api_key=OPENAI_API_KEY,
                               base_url=None,
                               log_dir="r:/logs",
                               timeout=3600,
                               use_response=STRUCTURED_RESPONSE,
                               use_memory=True
                                           )

    # --------------------------------------------------------------------
    # First pipeline
    # --------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # Agents
    llm_agent_1 = LLMAgent(config=llmOpenAiConfig_small)
    debugAgent1 = DebugAgent(DebugAgentConfig())
    # ------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------
    # Connect agents
    llm_agent_1.connectTo(debugAgent1)
    # ------------------------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------------------------
    # Scheduler
    scheduler_1:AgentScheduler = AgentScheduler(uuid='creatino pipeline')
    scheduler_1.add_agent(llm_agent_1)
    scheduler_1.add_agent(debugAgent1)
    # ------------------------------------------------------------------------------------------------


    # --------------------------------------------------------------------
    # Second pipeline – LLM verifies the first pipeline's final answer
    # --------------------------------------------------------------------
    llm_agent_2 = LLMAgent(config=llmOpenAiConfig_small)
    check_agent = CheckAgent(uuid="checker")

    # Chain: LLM‑2 → Debug‑2 → CheckAgent
    llm_agent_2.connectTo(check_agent)

    scheduler_2 = AgentScheduler(uuid="verification_pipeline")
    scheduler_2.add_agent(llm_agent_2)
    scheduler_2.add_agent(check_agent)

    def should_trigger():
        trigger = not scheduler_1.all_done()
        rich_console.print("[cyan]Callback triggered")
        return trigger

    def trigger_and_exit():
        history: str = llm_agent_1.getFormattedHistory(n=1)
        if not history.strip():
            rich_console.print("[yellow]Warning: No history to verify. Skipping feed.[/yellow]")
            return True  # Mark as done anyway, avoids infinite loop
        rich_console.print("[cyan]Conversation history from pipeline1...")
        rich_console.print(history)
        prompt = f"""Below is a conversation between a user and an assistant. 
Your task is to evaluate the factual accuracy of the assistant's *last response only*
based on current and reliable general knowledge.
Output one word only: 'correct' if the final assistant response is factually accurate, or 'incorrect' if it contains factual errors.
Conversation:\n{history}"""
        llm_agent_2.feed(LLMRequest(user=prompt))
        return True  # signal that we're done

    bridge = CallbackAgent(
        should_run=should_trigger,
        callback=trigger_and_exit,
        uuid="trigger_stage_2"
    )

    # --------------------------------------------------------------------
    # Master scheduler aggregates the two independent pipelines so you can
    # step them together if desired.  In this example we still control them
    # separately because we need the result of pipeline1 before feeding 2.
    # --------------------------------------------------------------------
    master_scheduler = AgentScheduler(uuid="master")
    master_scheduler.add_agent(scheduler_1)
    master_scheduler.add_agent(bridge)
    master_scheduler.add_agent(scheduler_2)

    # Print pipeline
    printer = PipelinePrinter(is_ortho=True, direction='TB', fillcolor='blue')
    printer.save_as_png(scheduler_1.agents, 'r:/pipeline_llm1.png')
    printer = PipelinePrinter(is_ortho=True, direction='TB', fillcolor='blue', show_schemas = True, schema_fillcolor = 'moccasin')
    printer.save_as_png(scheduler_1.agents, 'r:/pipeline_llm1_large.png')

    printer = PipelinePrinter(is_ortho=True, direction='TB', fillcolor='blue')
    printer.save_as_png(scheduler_2.agents, 'r:/pipeline_llm2.png')
    printer = PipelinePrinter(is_ortho=True, direction='TB', fillcolor='blue', show_schemas = True, schema_fillcolor = 'moccasin')
    printer.save_as_png(scheduler_2.agents, 'r:/pipeline_llm2_large.png')

    printer = PipelinePrinter(is_ortho=True, direction='TB', fillcolor='blue')
    printer.save_as_png(master_scheduler.agents, 'r:/pipeline_llm_m.png')
    printer = PipelinePrinter(is_ortho=True, direction='TB', fillcolor='blue', show_schemas = True, schema_fillcolor = 'moccasin')
    printer.save_as_png(master_scheduler.agents, 'r:/pipeline_llm_m_large.png')
    #return



    # ------------------------------------------------------------------------------------------------
    # Start processing
    rich_console.print(f"[red]Feeding agent ...")
    llm_agent_1.feed(LLMRequest(user='What is the capital of France'))
    llm_agent_1.feed(LLMRequest(user='How large is that city'))


    # ------------------------------------------------------------------------------------------------
    # Loop all
    master_scheduler.step_all()

    verification_outputs = scheduler_2.get_final_outputs()
    rich_console.print("[cyan]Verification pipeline outputs:", verification_outputs)

    state = check_agent.state
    print("State ",type(state),state)
    print("Correct",state.is_correct)
    print("Count", state.count)

    assert state is not None, "CheckAgent did not produce a state."
    assert state.is_correct is True, "Expected the verification result to be True (correct)."
    assert state.count == 1, f"Expected CheckAgent to run once, but ran {state.count} times."

    rich_console.print("Ready.")


if __name__ == '__main__':
    main()