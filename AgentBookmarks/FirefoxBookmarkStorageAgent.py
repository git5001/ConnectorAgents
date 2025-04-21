import os

from atomic_agents.lib.base.base_tool import BaseToolConfig
from pydantic import BaseModel, Field
from typing import List, Optional

from AgentBookmarks.BookmarkManager import BookmarkManager
from AgentBookmarks.FirefoxBookmarkAgent import FirefoxBookmarksOutput
from AgentFramework.ConnectedAgent import ConnectedAgent
from AgentFramework.NullSchema import NullSchema


# ---------------------------------------------------------------------
# Agent Configuration
# ---------------------------------------------------------------------
class FirefoxBookmarkStorageAgentConfig(BaseToolConfig):
    """
    Configuration for FirefoxBookmarkStorageAgent.
    Specifies the filename where the final Firefox bookmarks JSON file will be stored.
    """
    filename: str = Field(..., description="Output filename to store the Firefox bookmarks JSON file")


# ---------------------------------------------------------------------
# Agent: FirefoxBookmarkStorageAgent
# ---------------------------------------------------------------------
# Note: This assumes that ConnectedAgent and NullSchema are defined in your framework.
class FirefoxBookmarkStorageAgent(ConnectedAgent):
    """
    An agent that receives a list of bookmarks and stores them as a Firefox bookmarks JSON file.

    Attributes:
        input_schema (Type[BaseIOSchema]): Defines the expected input schema.
        output_schema (Type[BaseModel]): Always NullSchema, indicating no output.
    """
    input_schema = FirefoxBookmarksOutput
    output_schema = NullSchema

    def __init__(self, config: FirefoxBookmarkStorageAgentConfig, uuid: str = 'default') -> None:
        """
        Initializes the FirefoxBookmarkStorageAgent.

        Parameters:
          - config: Configuration including the output filename.
          - uuid: Optional unique identifier for the agent instance.
        """
        super().__init__(config, uuid)
        self.config: FirefoxBookmarkStorageAgentConfig = config

    def run(self, params: FirefoxBookmarksOutput) -> NullSchema:
        """
        Processes the incoming list of bookmarks, updates the internal Firefox bookmark tree,
        writes the JSON representation to disk, and returns a null output.

        Parameters:
          - params: FirefoxBookmarksOutput containing the list of bookmark entries.

        Returns:
          - A NullSchema instance indicating that processing is complete.
        """
        manager = BookmarkManager()

        for bm in params.bookmarks:
            # Use provided add_date or fallback to the current timestamp from the BookmarkManager.
            add_date_str = bm.add_date or str(manager._current_timestamp())
            # Use provided folder or default to empty string.
            folder_path = bm.folder if bm.folder is not None else ""
            manager.add_bookmark(
                title=bm.title,
                url=bm.url,
                folder=folder_path,
                add_date=add_date_str
            )
        # Serialize the complete bookmark tree to JSON.
        json_data = manager.to_json()

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(self.config.filename), exist_ok=True)

        # Write JSON data to the file specified in the configuration.
        with open(self.config.filename, "w", encoding="utf-8") as f:
            f.write(json_data)
        return NullSchema()
