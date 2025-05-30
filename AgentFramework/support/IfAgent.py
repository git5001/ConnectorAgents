import json
import sys
import types
from abc import ABC, abstractmethod
from typing import Dict, Type
from typing import List

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseToolConfig
from pydantic import BaseModel, create_model, Field, ConfigDict

from AgentFramework.core.ConnectedAgent import ConnectedAgent


class IfAgentConfig(BaseToolConfig):
    """
    Configuration for IfAgent.
    """
    schemas: List[str] = Field(
        ..., description="List of result schema names."
    )


class IfAgent(ConnectedAgent, ABC):
    """
    An agent that implements a if - condition

    Attributes:
        input_schema (Type[BaseIOSchema]): Defines the expected input schema.
        output_schema (Type[BaseModel]): Output.
    """
    input_schema = BaseIOSchema
    output_schema = None
    output_schemas = None

    def __init__(self, config: IfAgentConfig, **kwargs):
        self.payload_type = BaseModel
        self._schemas: Dict[str, Type[BaseModel]] = {}

        # Create all schemas from config at init
        self.output_schemas = []
        for name in config.schemas:
            schema = self.create_schema(name, BaseModel)
            self._schemas[name] = schema
            self.output_schemas.append(schema)
        super().__init__(config, **kwargs)

    def create_schema(self, name: str, payload_type: Type[BaseModel]) -> Type[BaseModel]:
        schema = create_model(
            name,
            payload=(payload_type, Field(..., description="The payload"))
        )
        # Ensure the module exists in sys.modules
        module_name = "AgentFramework.IfAgent"
        if module_name not in sys.modules:
            sys.modules[module_name] = types.ModuleType(module_name)

        schema.__module__ = module_name
        sys.modules[module_name].__dict__[name] = schema

        # Handle model config depending on your Pydantic version
        try:
            # For Pydantic v2
            schema.model_config = ConfigDict(arbitrary_types_allowed=True)
        except AttributeError:
            # For Pydantic v1
            schema.__config__.arbitrary_types_allowed = True

        schema.__module__ = "AgentFramework.IfAgent"
        sys.modules["AgentFramework.IfAgent"].__dict__[name] = schema
        schema.__pydantic_config__ = ConfigDict(arbitrary_types_allowed=True)

        def model_dump_json(self, *, indent: int | None = 4, **kwargs) -> str:
            """
            Monkey-patched to fix broken nested model serialization in dynamically created model.
            See: AgentFramework.IfAgent schema factory.
            """
            return json.dumps(
                {
                    'payload': self.payload.model_dump(
                        exclude_unset=False,
                        exclude_defaults=False,
                        exclude_none=False
                    )
                },
                indent=indent
            )

        setattr(schema, 'model_dump_json', model_dump_json)
        setattr(schema, 'model_dump', model_dump_json)
        return schema

    def schema(self, name: str) -> Type[BaseModel]:
        if name not in self._schemas:
            raise ValueError(f"Schema '{name}' is not defined in config.")
        return self._schemas[name]

    def all_schemas(self) -> Dict[str, Type[BaseModel]]:
        return self._schemas

    def run(self, params: BaseIOSchema) -> BaseModel:
        """
        Runs the agent.

        Args:
            params (BaseIOSchema): The input parameters.

        Returns:
            BaseModel: The selected schema class instance wrapping the payload.
        """
        # Decide which schema name to use (e.g., "ChoiceA", "ChoiceB")
        name = self.choice(params)

        # Get the dynamically created schema class by name
        current_class = self.schema(name)
        print("Chjopice ",name,current_class)
        print("Paypload ",params)

        # Wrap the payload and return
        result = current_class(payload=params)
        print("dfdsf ",result)
        print("dfdsfx ",result.payload)
        return result

    @abstractmethod
    def choice(self, params: BaseModel) -> str:
        """
        Must be implemented to return the name of the schema to use.
        """
        pass