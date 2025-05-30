from typing import Optional, Union
from urllib.parse import urlparse
import re

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify
from pydantic import Field, HttpUrl
from readability import Document

from atomic_agents.agents.base_agent import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseTool, BaseToolConfig

from AtomicTools.browser_handling.BrowserManager import get_browser_manager
from AtomicTools.webpage_scraper.tool.webpage_schema import WebpageScraperToolInputSchema, \
    WebpageScraperToolOutputSchema, WebpageMetadata


#################
# CONFIGURATION #
#################
class WebpageScraperToolConfig(BaseToolConfig):
    """Configuration for the WebpageScraperTool."""

    user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        ),
        description="User agent string to use for requests.",
    )
    timeout: int = Field(
        default=30,
        description="Timeout in seconds for HTTP requests.",
    )
    max_content_length: int = Field(
        default=1_000_000,
        description="Maximum content length in bytes to process.",
    )

    trim_excessive_content: bool = Field(
        default=False,
        description="If content is larger then max content trim it and do not raise error.",
    )


#####################
# MAIN TOOL & LOGIC #
#####################
class WebpageScraperTool(BaseTool):
    """
    Tool for scraping webpage content and converting it to markdown format.
    """

    input_schema = WebpageScraperToolInputSchema
    output_schema = WebpageScraperToolOutputSchema

    def __init__(self, config: WebpageScraperToolConfig = WebpageScraperToolConfig()):
        """
        Initializes the WebpageScraperTool.

        Args:
            config (WebpageScraperToolConfig): Configuration for the tool.
        """
        super().__init__(config)
        self.config = config

    def _fetch_webpage(self, url: str) -> str:
        """
        Fetches fully rendered HTML using Playwright.
        """
        manager = get_browser_manager()
        page = manager.get_page()
        page.goto(url, timeout=self.config.timeout * 1000)  # convert to ms
        html = page.content()
        page.close()
        if len(html) > self.config.max_content_length:
            if not self.config.trim_excessive_content:
                raise ValueError(f"Content length exceeds maximum of {self.config.max_content_length} bytes")
            html = html[:self.config.max_content_length]
        return html

    def _extract_metadata(self, soup: BeautifulSoup, doc: Document, url: str) -> WebpageMetadata:
        """
        Extracts metadata from the webpage.

        Args:
            soup (BeautifulSoup): The parsed HTML content.
            doc (Document): The readability document.
            url (str): The URL of the webpage.

        Returns:
            WebpageMetadata: The extracted metadata.
        """
        domain = urlparse(url).netloc

        # Extract metadata from meta tags
        metadata = {
            "title": doc.title(),
            "domain": domain,
            "author": None,
            "description": None,
            "site_name": None,
        }

        author_tag = soup.find("meta", attrs={"name": "author"})
        if author_tag:
            metadata["author"] = author_tag.get("content")

        description_tag = soup.find("meta", attrs={"name": "description"})
        if description_tag:
            metadata["description"] = description_tag.get("content")

        site_name_tag = soup.find("meta", attrs={"property": "og:site_name"})
        if site_name_tag:
            metadata["site_name"] = site_name_tag.get("content")

        return WebpageMetadata(**metadata)

    def _clean_markdown(self, markdown: str) -> str:
        """
        Cleans up the markdown content by removing excessive whitespace and normalizing formatting.

        Args:
            markdown (str): Raw markdown content.

        Returns:
            str: Cleaned markdown content.
        """
        # Remove multiple blank lines
        markdown = re.sub(r"\n\s*\n\s*\n", "\n\n", markdown)
        # Remove trailing whitespace
        markdown = "\n".join(line.rstrip() for line in markdown.splitlines())
        # Ensure content ends with single newline
        markdown = markdown.strip() + "\n"
        return markdown

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extracts the main content from the webpage using custom heuristics.

        Args:
            soup (BeautifulSoup): Parsed HTML content.

        Returns:
            str: Main content HTML.
        """
        # Remove unwanted elements
        for element in soup.find_all(["script", "style", "nav", "header", "footer"]):
            element.decompose()

        # Try to find main content container
        content_candidates = [
            soup.find("main"),
            soup.find(id=re.compile(r"content|main", re.I)),
            soup.find(class_=re.compile(r"content|main", re.I)),
            soup.find("article"),
        ]

        main_content = next((candidate for candidate in content_candidates if candidate), None)

        if not main_content:
            main_content = soup.find("body")

        return str(main_content) if main_content else str(soup)

    def run(self, params: WebpageScraperToolInputSchema) -> WebpageScraperToolOutputSchema:
        """
        Runs the WebpageScraperTool with the given parameters.

        Args:
            params (WebpageScraperToolInputSchema): The input parameters for the tool.

        Returns:
            WebpageScraperToolOutputSchema: The output containing the markdown content and metadata.
        """
        # Fetch webpage content
        try:
            html_content = self._fetch_webpage(str(params.url))

            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")

            # Extract main content using custom extraction
            main_content = self._extract_main_content(soup)

            # Convert to markdown
            markdown_options = {
                "strip": ["script", "style"],
                "heading_style": "ATX",
                "bullets": "-",
                "wrap": True,
            }

            if not params.include_links:
                markdown_options["strip"].append("a")

            markdown_content = markdownify(main_content, **markdown_options)

            # Clean up the markdown
            markdown_content = self._clean_markdown(markdown_content)

            # Extract metadata
            metadata = self._extract_metadata(soup, Document(html_content), str(params.url))

            return WebpageScraperToolOutputSchema(
                content=markdown_content,
                error=None,
                metadata=metadata,
            )
        except Exception as e:
            metadata = WebpageMetadata(title="",domain="")
            return WebpageScraperToolOutputSchema(
                content="",
                error=f"{e}",
                metadata=metadata,
            )


#################
# EXAMPLE USAGE #
#################
if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    scraper = WebpageScraperTool()

    try:
        result = scraper.run(
            WebpageScraperToolInputSchema(
                url="https://github.com/BrainBlend-AI/atomic-agents",
                include_links=True,
            )
        )

        console.print(Panel.fit("Metadata", style="bold green"))
        console.print(result.metadata.model_dump_json(indent=2))

        console.print(Panel.fit("Content Preview (first 500 chars)", style="bold green"))
        console.print(result.content)

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
