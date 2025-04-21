
import base64
import json
from typing import Any


class PydanticSerializer:
    @staticmethod
    def serialize(obj: Any) -> str:
        """
        Returns a base64-encoded string of the pickled object.
        """
        import dill as pickle
        pickled = pickle.dumps(obj)
        return base64.b64encode(pickled).decode("utf-8")

    @staticmethod
    def deserialize(data_str: str) -> Any:
        """
        Decodes the base64 string and unpickles it back to a Python object.
        """
        import dill as pickle
        raw = base64.b64decode(data_str)
        return pickle.loads(raw)

    @staticmethod
    def to_json(obj: Any, filename: str) -> None:
        """
        Writes the pickled+base64-encoded object to a JSON file.
        """
        encoded_str = PydanticSerializer.serialize(obj)
        with open(filename, "w", encoding="utf-8") as f:
            # We store the pickled data under the key "data" in a JSON object
            json.dump({"data": encoded_str}, f)

    @staticmethod
    def from_json(filename: str) -> Any:
        """
        Reads the JSON file, extracts the base64 string, and unpickles.
        """
        with open(filename, "r", encoding="utf-8") as f:
            content = json.load(f)
        return PydanticSerializer.deserialize(content["data"])
