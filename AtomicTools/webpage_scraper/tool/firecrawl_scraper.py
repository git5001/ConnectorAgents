from typing import Optional, List
from urllib.parse import urlparse
import re

from firecrawl.firecrawl import ScrapeResponse
from pydantic import Field
from firecrawl import FirecrawlApp  # pip install firecrawl-py

from atomic_agents.lib.base.base_tool import BaseTool, BaseToolConfig
from AgentFramework.core.ConnectedAgent import ConnectedAgent
from AtomicTools.webpage_scraper.tool.webpage_schema import WebpageScraperToolInputSchema, \
    WebpageScraperToolOutputSchema, WebpageMetadata


# --------------------
# Firecrawl-specific configuration
# --------------------
class FirecrawlScraperToolConfig(BaseToolConfig):
    """Configuration options for the Firecrawl-backed scraper."""

    api_key: str = Field(
        ..., description="Firecrawl API key (e.g. fc-XXXXXXXX)."
    )
    formats: List[str] = Field(
        default=["markdown"],
        description="Desired response formats accepted by Firecrawl (markdown, html, json, ...).",
    )
    location: Optional[dict] = Field(
        default=None,
        description=(
            "Optional location settings, e.g. {'country':'DE','languages':['de']} as per Firecrawl docs."
        ),
    )
    timeout: int = Field(
        default=30,
        description="Timeout (seconds) passed to the Firecrawl client.",
    )


# --------------------
# Main Tool
# --------------------
class FirecrawlScraperTool(BaseTool):
    """Scraper tool powered by Firecrawl. Matches AtomicTools interfaces."""

    input_schema = WebpageScraperToolInputSchema
    output_schema = WebpageScraperToolOutputSchema

    def __init__(self, config: FirecrawlScraperToolConfig):
        super().__init__(config)
        self.config: FirecrawlScraperToolConfig = config
        # Initialise Firecrawl application only once
        self._app = FirecrawlApp(api_key=self.config.api_key)

    @staticmethod
    def _strip_markdown_links(markdown: str) -> str:
        # Remove image markdown: ![alt](url)
        markdown = re.sub(r"!\[.*?\]\([^)]*\)", "", markdown)
        # Remove regular markdown links: [text](url) → text
        markdown = re.sub(r"\[(.*?)\]\([^)]*\)", r"\1", markdown)
        return markdown

    # --------------------
    # Core run method
    # --------------------
    def run(self, params: WebpageScraperToolInputSchema) -> WebpageScraperToolOutputSchema:
        """Scrapes the given URL via Firecrawl and returns markdown & metadata."""

        try:
            # Call Firecrawl; the SDK returns a ScrapeResponse object (pydantic model)
            result:ScrapeResponse = self._app.scrape_url(
                str(params.url),
                formats=self.config.formats,
                location=self.config.location or {},
            )

            # If Firecrawl signals an error, surface it immediately
            if getattr(result, "error", None):
                raise RuntimeError(result.error)

            # FirecrawlDocument fields
            markdown_content: str = getattr(result, "markdown", "") or ""
            html_content: str = getattr(result, "html", "") or ""

            # Optionally strip links
            if not params.include_links and markdown_content:
                markdown_content = self._strip_markdown_links(markdown_content)

            # If Markdown not provided (possible when only html requested), convert
            if not markdown_content and html_content:
                try:
                    from markdownify import markdownify

                    markdown_content = markdownify(html_content, heading_style="ATX")
                except ImportError:
                    # markdownify not installed – fall back to raw HTML string
                    markdown_content = html_content

            markdown_content = markdown_content.strip() + "\n"

            # Metadata – Firecrawl returns a dict (or None) in result.metadata
            raw_meta = getattr(result, "metadata", {}) or {}

            # Domain – prefer Firecrawl's sourceURL; fall back to provided URL
            source_url = raw_meta.get("sourceURL") or getattr(result, "url", str(params.url))
            domain = urlparse(source_url).netloc

            metadata = WebpageMetadata(
                title=raw_meta.get("title", getattr(result, "title", "")),
                author=raw_meta.get("author"),
                description=raw_meta.get("description", getattr(result, "description", None)),
                site_name=raw_meta.get("ogSiteName") or raw_meta.get("site_name"),
                domain=domain,
            )

            return WebpageScraperToolOutputSchema(
                content=markdown_content,
                raw=html_content,
                metadata=metadata,
                error=None,
            )

        except Exception as exc:
            # Bubble up error in unified schema
            return WebpageScraperToolOutputSchema(
                content="",
                metadata=WebpageMetadata(title="", domain=""),
                error=str(exc),
            )


# --------------------
# Agent wrapper for ConnectedAgent ecosystem
# --------------------
class FirecrawlScraperAgent(FirecrawlScraperTool, ConnectedAgent):
    """Agent class integrating the Firecrawl scraper into the ConnectedAgent framework."""

    input_schema = WebpageScraperToolInputSchema
    output_schema = WebpageScraperToolOutputSchema

    def __init__(self, config: FirecrawlScraperToolConfig):
        ConnectedAgent.__init__(self, config)
        FirecrawlScraperTool.__init__(self, config)

