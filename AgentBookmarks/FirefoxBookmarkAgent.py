import os
from typing import Optional, List
import json

from atomic_agents.lib.base.base_tool import BaseToolConfig
from pydantic import BaseModel, Field
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from AgentFramework.ConnectedAgent import ConnectedAgent
from bs4 import BeautifulSoup
from pydantic import TypeAdapter


# Define the Bookmark model representing a single bookmark entry.
class Bookmark(BaseModel):
    title: str = Field(..., description="The title of the bookmark")
    url: str = Field(..., description="The URL of the bookmark")
    folder: Optional[str] = Field(None, description="Folder name, if applicable")
    add_date: Optional[str] = Field(None, description="Date when bookmark was added")

# Define the input schema for the agent.
class FirefoxBookmarksInput(BaseModel):
    """
    Schema for a firefox bookmark message.

    Attributes:
        filepath:
    """
    filepath: str = Field(..., description="Path to the Firefox bookmark HTML export file")

# Define the output schema for the agent.
class FirefoxBookmarksOutput(BaseModel):
    """
    Bookmark output for Firefox bookmarks.
    """
    bookmarks: List[Bookmark] = Field(..., description="List of parsed bookmarks")

# If needed, extend the BaseAgentConfig. For now, we can just use the base.
class FirefoxBookmarkAgentConfig(BaseToolConfig):
    #filepath: str = Field(None, description="Path to the Firefox bookmark HTML export file")
    pass

class FirefoxBookmarkAgent(ConnectedAgent):
    """
    Agent that reads the Firefox bookmarks HTML export and produces structured output.
    """
    input_schema = FirefoxBookmarksInput
    output_schema = FirefoxBookmarksOutput

    def __init__(self, config: FirefoxBookmarkAgentConfig, uuid:str = 'default') -> None:
        super().__init__(config, uuid)

    def run(self, user_input: Optional[FirefoxBookmarksInput] = None) -> FirefoxBookmarksOutput:
        if user_input is None:
            raise ValueError("Input must be provided")

        filepath = user_input.filepath

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Bookmark file not found at path: {filepath}")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                file_content = f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to read file: {e}")

        if not file_content.strip():
            raise ValueError("Bookmark file is empty")

        #bookmarks = self._parse_bookmarks(html_content)
        bookmarks = self._parse_bookmarks_json(file_content)

        output = FirefoxBookmarksOutput(bookmarks=bookmarks)
        return output

    def _parse_bookmarks(self, html_content: str) -> List[Bookmark]:
        soup = BeautifulSoup(html_content, "html.parser")
        top_dl = soup.find("dl")
        if not top_dl:
            return []
        return self._parse_dl(top_dl)

    def _parse_dl(self, dl, parent_folder: Optional[str] = None) -> List[Bookmark]:
        bookmarks = []
        # Use find_all("dt") – not find_all("dt", recursive=False)
        for tag in dl.find_all("dt"):
            folder_tag = tag.find("h3")
            if folder_tag:
                folder_name = folder_tag.get_text(strip=True)
                next_dl = tag.find("dl") or tag.find_next_sibling("dl")
                if next_dl:
                    bookmarks.extend(self._parse_dl(next_dl, parent_folder=folder_name))
            else:
                a_tag = tag.find("a")
                if a_tag:
                    bookmark = Bookmark(
                        title=a_tag.get_text(strip=True),
                        url=a_tag.get("href"),
                        folder=parent_folder,
                        add_date=a_tag.get("add_date"),
                    )
                    bookmarks.append(bookmark)
        return bookmarks

    def _parse_bookmarks_json(self, json_content: str) -> List[Bookmark]:
        """
        Recursively parse the JSON-format bookmark backup from Firefox.
        """
        try:
            data = json.loads(json_content)
        except json.JSONDecodeError:
            raise ValueError("File is not valid JSON. Consider HTML fallback or verify the file format.")

        bookmarks = []

        def recurse(node: dict, parent_folder: Optional[str] = None, level:int = None):
            """
            Recursive helper that:
            - If node is a folder (typeCode=2), descends into children
            - If node is a bookmark (typeCode=1), records it
            - Merges folder name from the node’s title if relevant
            """
            node_type = node.get("typeCode")
            node_title = node.get("title") or ""

            if node_type == 2:
                # It's a folder/container
                # Some top-level items may have 'root' like 'placesRoot', etc.
                # You can decide to skip or rename these if you want.
                # If this is a known built-in root folder, skip adding it to the path
                if parent_folder in [
                    "placesRoot",
                    "bookmarksMenuFolder",
                    "toolbarFolder",
                    "unfiledBookmarksFolder",
                    "mobileFolder",
                    "unfiled",
                    "menu",
                ]:
                    new_folder_name = node_title  # don't add this to path
                else:
                    # Append this folder's title to the path
                    if parent_folder:
                        new_folder_name = f"{parent_folder}/{node_title}"
                    else:
                        new_folder_name = node_title

                for child in node.get("children", []):
                    recurse(child, new_folder_name, level + 1)

            elif node_type == 1:
                # It's a bookmark
                url = node.get("uri", "")
                add_date = node.get("dateAdded")
                # Convert to string or do custom formatting if you prefer
                if add_date:
                    add_date = str(add_date)

                bm = Bookmark(
                    title=node_title,
                    url=url,
                    folder=parent_folder,
                    add_date=add_date
                )
                bookmarks.append(bm)

        # Start recursion from the top-level object
        # The JSON can have multiple top-level children, so parse them if present
        # Some JSON backups store everything in data["children"][0], or data["children"]
        # so you might have to handle that. For maximum safety:
        if isinstance(data, dict) and "children" in data:
            # parse top-level
            recurse(data, None, 0)
        else:
            # Some backups store an array at the root
            if isinstance(data, list):
                for node in data:
                    recurse(node, None, 0)

        return bookmarks




