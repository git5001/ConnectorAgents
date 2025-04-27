import textwrap

from AgentFramework.ConnectedAgent import ConnectedAgent
from agent_config import DUMMY_LLM
from agent_logging import logger
from util.LLMSupport import LLMAgentConfig, LLMModel
from .schemas import SynthesisOutput, PageSummary, SynthezierInputModel


class PrettifySynthesisAgent(ConnectedAgent):
    input_schema = SynthesisOutput
    output_schema = SynthesisOutput

    def __init__(self, config: LLMAgentConfig, uuid:str = 'default') -> None:
        """
        Initializes an LLMAgent instance with OpenAI API configuration.

        Args:
            config (LLMAgentConfig, optional): Configuration for the agent. Defaults to LLMAgentConfig().
        """
        super().__init__(config, uuid)

        self.model: LLMModel = LLMModel(config, self.__class__.__name__)
        self.model.delete_log_files()

    def run(self, params: SynthesisOutput) -> SynthesisOutput:
        print("[Prettifier] Prettifying synthesis ")

        # Dummy branch
        if DUMMY_LLM:
            print("Output (dummy synthesis)")
            return SynthesisOutput(title=params.title, text=f"Prettfy: {params.text}")

        # Real LLM branch delegates to helper
        result_obj = self._prettify_with_llm(
            params.title,
            params.text
        )
        return SynthesisOutput(title=params.title, text=result_obj)

    def _prettify_with_llm(self, title: str, text: str) -> str:
        """
        Runs a second pass of an LLM over the text to make it smoother
        """
        prompt = textwrap.dedent(f"""
You are an expert academic editor specializing in polishing technical research reports without altering their core meaning.

### Objective
Perform a second-pass refinement of the provided text. Your task is to improve clarity, tighten the writing, enhance flow between sections, and sharpen the analytical insights, without changing the factual content or structure.

Focus on improving professional readability and eliminating redundancy while preserving technical depth.

### Content Rules:
- Do not shorten the overall text by more than 5–10%.
- If you encounter very short sub-sections, either combine them logically with adjacent sections or flesh them out with supporting explanation or examples.
- Add sharper insights and slightly enhance punchy analysis where appropriate, but stay true to the original argument.
- Actively connect related ideas across sections to show thematic links and logical progression.
- Do not remove necessary sections, but if a section is obviously unnecessary or redundant, condense it carefully.

### Format Rules:
- Paraphrase awkward, redundant, or overly long sentences into more direct academic prose.
- Use in-text attribution like `(Title)` to trace claims.
- If both a title and a URL are provided, format references as [Title](URL). Prefer titled references over raw URLs. Do not invent or change references or links.
- Use only English.
- Output must be in **valid markdown**, but do not include markdown backticks or code fencing.
- Do not add extra explanations or meta-comments outside the report text.
- Avoid filler statements like "this report will show" or "this document includes."
- Limit bullet lists; weave points into paragraphs unless a list materially improves clarity.
- Ensure logical flow between sections using thematic connectors (e.g., "Building upon...", "However, a critical challenge remains...").
- Critically challenge assumptions, highlight contradictions, and flag unresolved tensions where appropriate.
- Vary sentence structure and integrate examples to keep the academic style lively and readable.
- Use strong, decisive language when discussing consequences, risks, or opportunities. Avoid hedging unless uncertainty is indicated in the original text.
- Each sub-section must contain at least two fully-developed paragraphs unless brevity is required by the subject matter.

---
### Text
{text}

---
#
---
Begin the refined version now:
            """).strip()

        try:
            result_obj, usage = self.model.create_text_completions(
                None,
                prompt,
                0.5
            )
            print("LLM prettify usage:", usage)
            return result_obj

        except Exception as e:
            logger.error(f"{self.__class__.__name__} failed during synthesis: {e}")
            # fallback raw dump
            return "Error nothing generated"
