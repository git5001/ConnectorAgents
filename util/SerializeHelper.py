import json
import pickle
import base64
import importlib
from pydantic import BaseModel, AnyUrl

def encode_payload(payload):
    """
    Recursively convert a Python object into a JSON-safe structure.
    We store:
      - Pydantic models with `_pydantic=True, _class='path.to.Model', data=...`
      - AnyUrl objects the same way (but data=the string)
      - Dicts/lists → recurse
      - Everything else → if json.dumps fails, store pickled data
    """

    if payload is None:
        return None

    # 1) AnyUrl (includes HttpUrl)
    if isinstance(payload, AnyUrl):
        return {
            "_pydantic": True,
            "_class": f"{payload.__class__.__module__}.{payload.__class__.__qualname__}",
            "data": str(payload)
        }

    # 2) Any Pydantic BaseModel
    if isinstance(payload, BaseModel):
        return {
            "_pydantic": True,
            "_class": f"{payload.__class__.__module__}.{payload.__class__.__qualname__}",
            "data": encode_payload(payload.model_dump())  # Recurse-encode the dict
        }

    # 3) If dict, recurse over values
    if isinstance(payload, dict):
        return {k: encode_payload(v) for k, v in payload.items()}

    # 4) If list/tuple, recurse
    if isinstance(payload, (list, tuple)):
        return [encode_payload(x) for x in payload]

    # 5) Check if it's JSON-serializable
    try:
        json.dumps(payload)
        return payload  # It's fine
    except TypeError:
        pass

    # 6) Fallback to pickle + base64
    try:
        pickled = pickle.dumps(payload)
        encoded = base64.b64encode(pickled).decode("utf-8")
        return {
            "_binary": True,
            "_class": f"{payload.__class__.__module__}.{payload.__class__.__qualname__}",
            "data": encoded
        }
    except Exception as e:
        print(f"Failed to pickle {payload}: {e}")
        return None


def decode_payload(payload):
    """
    Recursively decode the data:
    - If "_pydantic": True, import the class + rebuild (AnyUrl or BaseModel)
    - If "_binary": True, unpickle
    - Otherwise, if it's a dict/list, recurse
    - Return final object
    """
    if payload is None:
        return None

    if isinstance(payload, dict):
        # (1) If it's a pydantic object
        if payload.get("_pydantic") is True:
            class_path = payload.get("_class")
            data = payload.get("data")

            # Dynamically import the class
            module_path, class_name = class_path.rsplit(".", 1)
            try:
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name)
            except (AttributeError, ModuleNotFoundError) as e:
                print(f"Failed to import pydantic class {class_path}: {e}")
                return None

            # Recurse-decode the data
            decoded_data = decode_payload(data)

            from pydantic import BaseModel, AnyUrl
            if issubclass(cls, BaseModel):
                # If it's a Pydantic model, we do: cls(**decoded_data)
                return cls(**decoded_data)
            elif issubclass(cls, AnyUrl):
                # If it's an AnyUrl subtype (e.g. HttpUrl), pass the string
                return cls(decoded_data)
            else:
                # Some other type that isn't a BaseModel or AnyUrl
                return cls(decoded_data)

        # (2) If it's a binary pickled object
        if payload.get("_binary") is True:
            try:
                raw = base64.b64decode(payload["data"])
                return pickle.loads(raw)
            except Exception as e:
                print(f"Failed to unpickle {payload.get('_class')}: {e}")
                return None

        # (3) Otherwise, normal dict → recurse
        return {k: decode_payload(v) for k, v in payload.items()}

    if isinstance(payload, list):
        return [decode_payload(item) for item in payload]

    return payload  # string, int, float, etc.
