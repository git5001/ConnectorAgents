
from typing import List, Optional
from pydantic import Field, BaseModel
from atomic_agents.lib.base.base_io_schema import BaseIOSchema

class UserQueryInput(BaseIOSchema):
    """
    Schema for the user input
    """
    query: str = Field(..., description="User research question")


class EnhancedQueryOutput(BaseIOSchema):
    """
    Schema for output enhancing
    Alias: enhancer
    """
    original: str = Field(..., description="User original research question")
    search:str = Field(..., description="User research question for web search")
    enhanced: str = Field(..., description="User research question enhanced for resarch report generation")


class PageSummary(BaseIOSchema):
    """
    Schema for page summary
    """
    url: str
    title: str
    web_summary:str
    research_summary: str

class PageSummaryItemSchema(BaseIOSchema):
    """This schema represents a single search result item"""

    title: str = Field(..., description="The title of the search result")
    url: str = Field(..., description="The URL of the search result")
    content: str = Field(..., description="The content snippet of the search result")
    raw_content: str = Field(..., description="The raw content of the search result")
    research_query: str = Field(..., description="The reseach query")



class SummariesListOutput(BaseIOSchema):
    """
    Schema for summaries list
    """
    summaries: List[PageSummary]


class SynthesisOutput(BaseIOSchema):
    """
    Schema for sysnthesis
    """
    title:str
    text: str


class DebugModel(BaseModel):
    prompt: str
    result: str
    content:str
    relevance: int


class TavilySearchListModel(BaseModel):
    """
    The resulting list data.
    Alias: synthesis
    """
    data: List[PageSummary] = Field(default_factory=list, description="Combined list result")

class SynthezierInputModel(BaseModel):
    """
    The resulting list data.
    """
    data: List[PageSummary] = Field(default_factory=list, description="Combined list result")
    input: EnhancedQueryOutput =  Field(default_factory=list, description="Input data")
