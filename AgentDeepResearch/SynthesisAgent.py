import textwrap

from AgentFramework.ConnectedAgent import ConnectedAgent
from agent_config import DUMMY_LLM
from agent_logging import logger
from util.LLMSupport import LLMAgentConfig, LLMModel
from .schemas import SynthesisOutput, PageSummary, SynthezierInputModel


class SynthesisAgent(ConnectedAgent):
    input_schema = SynthezierInputModel
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

    def run(self, params: SynthezierInputModel) -> SynthesisOutput:
        print("[Synthesizer] Synthesizing collected summaries #", len(params.data))
        title = params.input.original
        # Build header and source notes
        header = textwrap.dedent(f"""
            USER INPUT
            ----------
            - Original:  {params.input.original}
            - Enhanced:  {params.input.enhanced}
        """).strip()

        source_blocks = []
        for item in params.data:
            item: PageSummary = item
            block = textwrap.dedent(f"""
                TITLE: {item.title}
                URL REFERENCE: {item.url}
                FINDINGS:
                {item.research_summary}

                ADDITIONAL CONTEXT (condensed web page summary):
                {item.web_summary}
            """).strip()
            source_blocks.append(block)

        full_context = header + "\n\n" + "\n\n---\n\n".join(source_blocks)

        # Dummy branch
        if DUMMY_LLM:
            print("Output (dummy synthesis)")
            return SynthesisOutput(title=title, text=f"SYNTHESIS\n=======\n{full_context}")

        # Real LLM branch delegates to helper
        result_obj = self._synthesize_with_llm(
            question=params.input.enhanced,
            context=full_context
        )
        return SynthesisOutput(title=title, text=result_obj)

    def _synthesize_with_llm(self, question: str, context: str) -> str:
        """
        Encapsulates the prompt construction and LLM invocation for the synthesis step.
        Returns an LLMSynthesisOutput with 'synthesis' and 'coverage'.
        """
        prompt = textwrap.dedent(f"""
You are a senior research synthesis assistant.

### Objective  
Produce an in-depth, well-structured report synthesis that directly addresses the  
*Research Question* below by integrating all relevant insights from the  
*Source Notes*. Enhance the source notes by your own knownledge so that
you produce a coherent and comprehensive report.

Write in a formal academic style, maintaining all technical detail  
and organizing the content under clear, informative headings.

- Group related ideas into the following sections:  
    1. Introduction / Background  
    2. Current Knowledge / Key Insights  
    3. Implications  
    4. Limitations  
    5. Conclusion  
    6. References (optional section)  
Note: No section or no inserted sub-section must be too short.
If you have too short sub-sections, either combine them, make them a list
or flesh them out. We want to avoid once sentence sub-sections.  

Goal is a comprehensive well structured report about the research question.
A complicated topic justifies a lengthy report.

### Rules:
- Paraphrase rather than quote unless wording is essential.  
- Use in-text attribution like `(Title)` where helpful to trace claims.  
- Do NOT mention web pages or metadata, make scientific style references on the provided URLs.
- Never invent references, better leave them out than inventing them.
  Use only references and URLs given in the input. 
  If you provide URL references make them proper markdown links.
- Use only English.  
- Output must be in **valid markdown** but never include markdown backticks like ```markdown or just ```.
  The output must be just the text wihtout any leading or trailing ticks or quotes. Note,
  martkdown will automatically happen if you just print the text using markdown formatting.  
- Do not include extra explanations or comments outside the report itself.
- Do not include anything like "this report includes" or "this synthesis shows" ... 

---
### QUERY: Research Question
{question}

---
### Source Notes
{context}

---
Begin the structured synthesis:
            """).strip()

        try:
            result_obj, usage = self.model.create_text_completions(
                None,
                prompt,
                0.5
            )
            print("LLM synthesis usage:", usage)
            return result_obj

        except Exception as e:
            logger.error(f"{self.__class__.__name__} failed during synthesis: {e}")
            # fallback raw dump
            return "Error nothing generated"
