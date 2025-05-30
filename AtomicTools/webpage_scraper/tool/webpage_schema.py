from typing import Optional, Union

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import Field, HttpUrl

class aaa:
    pass

################
# INPUT SCHEMA #
################
class WebpageScraperToolInputSchema(BaseIOSchema):
    """
    Input schema for the WebpageScraperTool.
    """

    url: Union[HttpUrl, None] = Field(
        ...,
        description="URL of the webpage to scrape.",
    )
    include_links: bool = Field(
        default=True,
        description="Whether to preserve hyperlinks in the markdown output.",
    )


#################
# OUTPUT SCHEMA #
#################
class WebpageMetadata(BaseIOSchema):
    """Schema for webpage metadata."""

    title: str = Field(..., description="The title of the webpage.")
    author: Optional[str] = Field(None, description="The author of the webpage content.")
    description: Optional[str] = Field(None, description="Meta description of the webpage.")
    site_name: Optional[str] = Field(None, description="Name of the website.")
    domain: str = Field(..., description="Domain name of the website.")


class WebpageScraperToolOutputSchema(BaseIOSchema):
    """Schema for the output of the WebpageScraperTool."""

    content: str = Field(..., description="The scraped content in markdown format.")
    metadata: WebpageMetadata = Field(..., description="Metadata about the scraped webpage.")
    error: Optional[str] = Field(None, description="The read status of the webpage.")
    raw: Optional[str] = Field(None, description="The raw scraped content format.")

