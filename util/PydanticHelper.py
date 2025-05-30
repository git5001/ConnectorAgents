from pydantic import BaseModel
from typing import Any
import json

def force_full_dump(obj: Any) -> Any:
    """
    Recursively dumps BaseModel fields, even if they were copied or not explicitly marked as 'set'.
    """
    if isinstance(obj, BaseModel):
        data = {}
        for name, field in obj.__pydantic_fields__.items():
            value = getattr(obj, name, None)
            data[name] = force_full_dump(value)
        return data
    elif isinstance(obj, list):
        return [force_full_dump(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: force_full_dump(v) for k, v in obj.items()}
    else:
        return obj

def model_dump_json(model: BaseModel, *, indent: int | None = 4) -> str:
    """
    JSON dumps a BaseModel, forcing expansion of nested models regardless of 'fields_set'.
    """
    return json.dumps(force_full_dump(model), indent=indent)
