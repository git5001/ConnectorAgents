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
1. Introduction & Background:
    Introduce the topic clearly.
    Provide relevant context or background needed to understand the subject.
    Explain why the topic is important or relevant.

2. Current Knowledge & Key Insights:
    Summarize the existing information, facts, theories, or practices related to the topic.
    Highlight major developments, trends, or important aspects without offering detailed interpretation.

3. Analysis:
    Critically evaluate and interpret the information presented earlier.
    Explain key relationships, causes and effects, strengths and weaknesses, successes and failures.
    Identify important patterns, lessons, or contradictions.

4. Implications:
    Discuss the broader significance or consequences of the topic and analysis.
    Explain how this topic influences related areas, future developments, or real-world applications.

5. Limitations:
    Acknowledge uncertainties, gaps, biases, or constraints in current understanding.
    Explain any factors that limit the ability to draw firm conclusions.
    Note: If no significant limitations are identified in the source material, briefly note this rather than
    fabricating speculative gaps.

6. Conclusion:
    Summarize the major findings and arguments made.
    Restate the overall significance of the topic in a concise way.
    Offer a final reflection or closing thought if appropriate.

7. References (optional):
    Provide a list of sources cited or used to support the information, if applicable.
    Use consistent and appropriate citation formatting.
    

Goal is a comprehensive well structured report about the research question.
A complicated topic justifies a lengthy report.

### Content rules:
- The output must be very extensive and comprehensive, at least 3000 words long. Do not state the word count in the report.
- Elaborate all topics in great detail  
- If you have too short sub-sections, either combine them, make them a list
  or flesh them out. We must avoid "one" sentence sub-sections.  
- Add sharp insights and punchy analysis  
- Connect dots (topics, ideas, ...)
- Perform and show critical thinking!
- Sections which are totally unnecessar for the topic can be omitted!


### Format Rules:
- Paraphrase rather than quote unless wording is essential.  
- Use in-text attribution like `(Title)` where helpful to trace claims.
  If both a proper title and a URL are available, format references as [Title](URL). Prefer titled references over raw URLs. Do not include raw web links without descriptive titles.
- Do NOT mention web pages or metadata, make scientific style references on the provided URLs.
- Never invent references, better leave them out than inventing them. If a claim lacks backing in the Source Notes and you have no reliable citation, flag the uncertainty briefly rather than creating a reference.
  Use only references and URLs given in the input. 
  If you provide URL references make them proper markdown links.
- Use only English.  
- Output must be in **valid markdown** but never include markdown backticks like ```markdown or just ```.
  The output must be just the text without any leading or trailing ticks or quotes. Note,
  markdown will automatically happen if you just print the text using markdown formatting.  
- Do not include extra explanations or comments outside the report itself.
- Do not include anything like "this report includes" or "this synthesis shows" ... 
- Limit bullet lists; prefer weaving points into full paragraphs unless the list materially improves clarity.
- Ensure logical flow between sections by using transitional phrases and thematic connectors (e.g., 'Building upon...', 'However, a critical issue remains...').
- Critically challenge underlying assumptions and highlight contradictions, controversies, or unresolved tensions where applicable.
- Vary sentence structure and integrate embedded examples or case illustrations where useful to maintain a natural, engaging academic style.
- Use strong, decisive language when discussing consequences, risks, or opportunities. Avoid excessive hedging unless uncertainty is noted in source material.
- Each sub-section must contain at least two fully-developed paragraphs unless the nature of the content inherently demands brevity.

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
