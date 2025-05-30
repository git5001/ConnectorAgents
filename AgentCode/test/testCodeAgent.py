import os
from dotenv import load_dotenv

from AgentCode.LLMCodeAgent import CodeLLMAgent, LLMCodeAgentConfig
from AgentFramework.core.AgentScheduler import AgentScheduler
from AgentFramework.support.DebugAgent import DebugAgent, DebugAgentConfig
from util.LLMSupport import Provider, LLMRequest

TASK = "Write a Python program that prints the 10 first fibonacci numbers to console. Write additionally index+number to a csv file of format index;result"
TASK = "Find all three-digit numbers where the sum of the cubes of the digits equals the number itself (e.g., 153 = 1³ + 5³ + 3³). Output to a file."
TASK = """
    Download the latest exchange rates for USD, EUR, and JPY from an open API.
    Convert a hardcoded list of amounts (e.g., [100, 250, 500]) from USD to EUR and JPY.
    Write the results in a CSV with format amount_usd;eur_equivalent;jpy_equivalent.
"""
TASK = """
Simulate the motion of a double pendulum over time using numerical integration
Plot a phase portrait: plot theta1 vs. dtheta1/dt and/or theta2 vs. dtheta2/dt.
Make the plot visually appealing:
    - Use a smooth color gradient to represent time progression.
    - Add labels, title, and a colorbar.
    - Use high resolution and export as a PNG.
"""
TASK = """
Write MATLAB code to:
    Simulate a strange attractor, such as:
        Lorenz attractor
        Rössler attractor
        Thomas attractor
    Generate a 3D plot of the trajectory (x, y, z).
    Style the plot to make it visually compelling:
        Use a color gradient to represent time (e.g., using colormap and scatter3 or a colored line).
        Remove axes and grid to highlight the geometry.
        Use anti-aliasing, lighting effects, and export as a high-res PNG.
"""


def main():
    # Load environment variables (e.g., OPENAI_API_KEY)
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY in your environment.")

    # Configure the LLM for code generation
    llm_config = LLMCodeAgentConfig(
        model="gpt-4o-mini",
        provider=Provider.OPENAI,
        api_key=api_key,
        base_url=None,
        log_dir="r:/logs",
        timeout=300,
        use_response=False,
        use_memory=True,
        code_dir="r:/code"
    )

    # Instantiate agents
    code_agent = CodeLLMAgent(config=llm_config)
    debug_agent = DebugAgent(DebugAgentConfig())

    # Connect the code agent to the debug agent to capture outputs
    code_agent.connectTo(debug_agent)

    # Create a scheduler and add both agents
    scheduler = AgentScheduler(uuid="simple_code_pipeline")
    scheduler.add_agent(code_agent)
    scheduler.add_agent(debug_agent)

    # Feed the code agent a simple task
    code_agent.feed(LLMRequest(user=TASK))

    # Run the pipeline
    scheduler.step_all()

    hist = code_agent.getFormattedHistory()
    print("HIST",hist)


if __name__ == '__main__':
    main()
