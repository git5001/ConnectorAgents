import re
import logging
logger = logging.getLogger(__name__)

class SchemaUtils:
    @staticmethod
    def enforce_additional_properties_false(schema: dict) -> dict:
        """
        Recursively ensure all objects in the schema have 'additionalProperties': false.
        """
        if "type" in schema and schema["type"] == "object":
            # Add 'additionalProperties': false if not already set
            schema.setdefault("additionalProperties", False)

            # Recurse into properties
            if "properties" in schema:
                for key, value in schema["properties"].items():
                    SchemaUtils.enforce_additional_properties_false(value)

        # Handle arrays of objects
        if "items" in schema and isinstance(schema["items"], dict):
            SchemaUtils.enforce_additional_properties_false(schema["items"])

        # Handle `$defs` for sub-schemas
        if "$defs" in schema:
            for key, value in schema["$defs"].items():
                SchemaUtils.enforce_additional_properties_false(value)

        return schema


    @staticmethod
    def inline_all_references(schema: dict) -> dict:
        """
        Recursively resolves and inlines `$ref` and `allOf` references in a JSON schema.

        Args:
            schema (dict): The JSON schema to process.

        Returns:
            dict: The modified schema with inlined references.
        """
        # Store $defs for resolving references
        definitions = schema.get("$defs", {})

        def resolve_references(sub_schema: dict) -> dict:
            """
            Recursively resolve `$ref` and `allOf` in a schema.

            Args:
                sub_schema (dict): The part of the schema to resolve.

            Returns:
                dict: The resolved schema.
            """
            if isinstance(sub_schema, dict):
                # Resolve $ref
                if "$ref" in sub_schema:
                    ref_path = sub_schema["$ref"]
                    if ref_path.startswith("#/$defs/"):
                        ref_key = ref_path.split("/")[-1]
                        if ref_key in definitions:
                            resolved_ref = definitions[ref_key]
                            # Merge resolved reference with existing properties
                            sub_schema = {**resolved_ref, **sub_schema}
                            sub_schema.pop("$ref", None)

                # Resolve allOf
                if "allOf" in sub_schema:
                    resolved_all_of = {}
                    for item in sub_schema["allOf"]:
                        resolved_item = resolve_references(item)
                        resolved_all_of.update(resolved_item)
                    sub_schema = {**resolved_all_of, **sub_schema}
                    sub_schema.pop("allOf", None)

                # Recursively resolve nested schemas
                for key, value in list(sub_schema.items()):
                    sub_schema[key] = resolve_references(value)

            elif isinstance(sub_schema, list):
                return [resolve_references(item) for item in sub_schema]

            return sub_schema

        # Resolve references in the main schema
        resolved_schema = resolve_references(schema)

        # Remove $defs since all references are inlined
        resolved_schema.pop("$defs", None)

        return resolved_schema

def generate_template_json(schema: dict) -> str:
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    def format_value(prop):
        typ = prop.get("type", "string")
        if typ == "string":
            return "string"
        elif typ == "integer":
            return "int"
        elif typ == "number":
            return "float"
        elif typ == "boolean":
            return "true/false"
        elif typ == "array":
            item_type = prop.get("items", {}).get("type", "string")
            return f"[{item_type}]"
        elif typ == "object":
            return "{...}"
        else:
            return "value"

    lines = ["{"]

    for i, (key, prop) in enumerate(properties.items()):
        value = format_value(prop)
        desc = prop.get("description", "")
        comma = "," if i < len(properties) - 1 else ""
        lines.append(f'  "{key}": {value}  // {desc}{comma}')

    lines.append("}")
    return "\n".join(lines)

def generate_llm_prompt_from_schema(schema: dict) -> str:
    title = schema.get("title", "Structured Output")
    description = schema.get("description", "")
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    def describe_field(name: str, details: dict) -> str:
        field_type = details.get("type", "string")
        field_title = details.get("title", name)
        field_desc = details.get("description", "")
        is_required = name in required

        # Type description
        if field_type == "array":
            item_type = details.get("items", {}).get("type", "string")
            type_str = f"List of {item_type.capitalize()}s"
        elif field_type == "object":
            type_str = "Nested object"
        else:
            type_str = field_type.capitalize()

        # Format line
        req_tag = "Required" if is_required else "Optional"
        return f"- `{name}` ({type_str}, {req_tag}): {field_desc}"

    # Start building the prompt
    lines = [f"You are to generate a JSON object with the following structure:"]
    if description:
        lines.append("")
        lines.append(f"**Context**: {description}")
    lines.append("")
    lines.append(f"**Fields for `{title}`**:")
    lines.append("")

    for field_name, field_def in properties.items():
        lines.append(describe_field(field_name, field_def))

    lines.append("")
    lines.append("Return only a valid JSON object following this structure. No extra text.")
    return "\n".join(lines)


def clean_json_string(input_string):
    """
    Cleans an input string by removing wrapping backticks and extraneous markers (e.g., ```json, ```).

    Args:
        input_string (str): The raw string containing JSON wrapped in backticks or markers.

    Returns:
        str: A cleaned JSON string suitable for json.loads.
    """
    # Remove backticks and markers like ```json
    cleaned_string = re.sub(r'^```(?:json)?\n|\n```$', '', input_string, flags=re.MULTILINE)
    cleaned_string = cleaned_string.strip('`')

    # Find the first '{' and the last '}' to extract the JSON payload
    start_index = cleaned_string.find('{')
    end_index = cleaned_string.rfind('}')

    # Ensure the indices are valid
    if start_index == -1 or end_index == -1 or start_index > end_index:
        logger.error(f"ERROR T111 {input_string}")
        raise ValueError("Input does not contain a valid JSON object.")

    # Extract the JSON portion
    cleaned_string = cleaned_string[start_index:end_index + 1]

    return cleaned_string