from typing import Optional
from pydantic import BaseModel, Field

class IdWrapper(BaseModel):
    """
    Wrapper class for a dictionary of BaseIOSchema instances.
    """
    message: BaseModel = Field(
        ...,
        description="The actual parameter schema to pass to run"
    )
    id: Optional[str] = Field(
        None,
        description="An optional user chain id"
    )

    def to_text(self, indent: int = 4) -> str:
        """
        Convert the payload into a JSON-formatted string.

        Args:
            indent (int): Number of spaces for indentation in JSON output.

        Returns:
            str: JSON representation of the payload.
        """
        # exclude_none=True will drop the "user" field if it's None
        return self.model_dump_json(indent=indent, exclude_none=True)
