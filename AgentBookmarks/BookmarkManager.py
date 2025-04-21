from pydantic import BaseModel, Field, ConfigDict
from typing import List, Union, Optional, Literal
import uuid
import time


# Define the Bookmark model with constant type using Literal.
class Bookmark(BaseModel):
    model_config = ConfigDict(discriminator="type")
    guid: str = Field(default_factory=lambda: uuid.uuid4().hex)
    title: str
    index: int = 0
    dateAdded: int
    lastModified: int
    id: int = 0
    typeCode: int = 1
    type: Literal["text/x-moz-place"] = "text/x-moz-place"
    uri: str
    iconUri: Optional[str] = None


# Define the Folder model (container) which can include other folders or bookmarks.
class Folder(BaseModel):
    model_config = ConfigDict(discriminator="type")
    guid: str = Field(default_factory=lambda: uuid.uuid4().hex)
    title: str
    index: int = 0
    dateAdded: int
    lastModified: int
    id: int = 0
    typeCode: int = 2
    type: Literal["text/x-moz-place-container"] = "text/x-moz-place-container"
    root: Optional[str] = None
    children: List[Union["Folder", Bookmark]] = Field(default_factory=list)


# Resolve forward references for Folder.
Folder.model_rebuild()


# The top-level tree container that represents the Firefox bookmarks root.
class BookmarkTree(Folder):
    root: str = "placesRoot"


# Manager class to hold and update the bookmark tree.
class BookmarkManager:
    def __init__(self):
        # Create a default tree with "Menu" and "Toolbar" folders.
        self.tree = BookmarkTree(
            guid="root________",
            title="",
            index=0,
            dateAdded=1557425390477000,
            lastModified=1743415451401000,
            id=1,
            children=[
                Folder(
                    guid="menu________",
                    title="Menu",
                    index=0,
                    dateAdded=1557425390477000,
                    lastModified=1743322321270000,
                    id=2,
                    root="bookmarksMenuFolder",
                    children=[]
                ),
                Folder(
                    guid="toolbar_____",
                    title="Toolbar",
                    index=1,
                    dateAdded=1557425390477000,
                    lastModified=1743322321270000,
                    id=3,
                    root="bookmarksToolbarFolder",
                    children=[]
                )
            ]
        )
        # Starting value for new IDs.
        self._next_id = 1000

    def _get_next_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _current_timestamp(self) -> int:
        # Return the current time in microseconds.
        return int(time.time() * 1_000_000)

    def _find_or_create_folder(self, folder_path: str) -> Folder:
        """
        Given a folder path (e.g., "Evelyn/Hobby"), navigate to (or create)
        the folder hierarchy under the "Menu" folder.
        """
        parts = folder_path.split("/")
        # Start by looking for the "Menu" folder.
        menu_folder: Optional[Folder] = None
        for child in self.tree.children:
            if isinstance(child, Folder) and child.title == "Menu":
                menu_folder = child
                break
        # If "Menu" folder not found, create it.
        if menu_folder is None:
            menu_folder = Folder(
                title="Menu",
                index=len(self.tree.children),
                dateAdded=self._current_timestamp(),
                lastModified=self._current_timestamp(),
                id=self._get_next_id(),
                root="bookmarksMenuFolder",
                children=[]
            )
            self.tree.children.append(menu_folder)

        current_folder = menu_folder
        for part in parts:
            # Look for an existing subfolder with the matching title.
            next_folder = None
            for child in current_folder.children:
                if isinstance(child, Folder) and child.title == part:
                    next_folder = child
                    break
            # Create new folder if it does not exist.
            if next_folder is None:
                next_folder = Folder(
                    title=part,
                    index=len(current_folder.children),
                    dateAdded=self._current_timestamp(),
                    lastModified=self._current_timestamp(),
                    id=self._get_next_id(),
                    children=[]
                )
                current_folder.children.append(next_folder)
            current_folder = next_folder
        return current_folder

    def add_bookmark(self, title: str, url: str, folder: str, add_date: str):
        """
        Add a bookmark to the tree.

        Parameters:
          - title: The display title of the bookmark.
          - url: The URL of the bookmark.
          - folder: A path-like string (e.g., "Evelyn/Hobby") where the bookmark is stored.
          - add_date: A string representing the timestamp (in microseconds) for both `dateAdded` and `lastModified`.

        Example usage:
            add_bookmark("Sternenschweif – Wikipedia",
                         "https://de.wikipedia.org/wiki/Sternenschweif",
                         "Evelyn/Hobby",
                         "1695579970525000")
        """
        add_date_int = int(add_date)
        target_folder = self._find_or_create_folder(folder)
        new_bookmark = Bookmark(
            title=title,
            uri=url,
            dateAdded=add_date_int,
            lastModified=add_date_int,
            id=self._get_next_id()
        )
        # Set bookmark's index based on the existing children count in the target folder.
        new_bookmark.index = len(target_folder.children)
        target_folder.children.append(new_bookmark)

    def to_json(self) -> str:
        """Serialize the entire bookmark tree to a JSON string."""
        return self.tree.model_dump_json(indent=2)

