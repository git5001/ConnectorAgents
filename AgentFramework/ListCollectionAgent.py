from typing import List, Optional, Tuple, TypeVar, Generic

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseToolConfig
from pydantic import BaseModel, Field

from AgentFramework.ConnectedAgent import ConnectedAgent
from AgentFramework.listutil import find_common_complete_uuids, compare_lists

T = TypeVar("T", bound=BaseModel)

class ListModel(BaseModel, Generic[T]):
    """
    The resulting list data.
    """
    data: List[T] = Field(default_factory=list, description="Combined list result")

def transform_list_2_modellist(data: List[BaseModel]) -> ListModel:
    """Converts result of a list of models to a model list."""
    result: ListModel = ListModel(data=data)
    return result

class ListCollectionAgentState(BaseModel):
    """
    State of the list agent.
    """
    data: List[Tuple[BaseIOSchema, List[str], List[str]]] = Field(default_factory=list, description="Stored intermediate results for merging")


class ListCollectionAgent(ConnectedAgent):
    """
    An agent that aggregates data from multiple runs into a combined dataset.
    Useful for merging segmented outputs, such as news article summaries processed in separate LLM runs.

    Attributes:
        input_schema (Type[BaseIOSchema]): Defines the expected input schema.
        output_schema (Type[BaseIOSchema]): Defines the expected output schema.
        data (List[Tuple[BaseIOSchema, List[str], List[str]]]): Stores intermediate results for combination.
    """
    input_schema = BaseIOSchema
    output_schema = BaseIOSchema
    state_schema = ListCollectionAgentState
    _state: ListCollectionAgentState = None

    def __init__(self, config: BaseToolConfig, uuid:str = 'default') -> None:
        """
        Initializes the ListCollectionAgent.

        Args:
            config (BaseToolConfig): Configuration for the agent.
        """
        super().__init__(config, uuid)
        self._state = ListCollectionAgentState(data=[])

    def _replace_if_needed(self, parents: List[str], n: int) -> None:
        if n >= len(parents) or n < 0:
            raise IndexError(f"Index {n} is out of range for 'parents' list of length {len(parents)}")

        value = parents[n]
        parts = value.rsplit(":", 2)

        if len(parts) != 3:
            raise ValueError(f"Invalid format for parent entry: '{value}'. Expected format 'something:a:b'")

        prefix, a_str, b_str = parts
        a, b = int(a_str), int(b_str)

        if a == 0 and b == 1:
            raise ValueError(f"Invalid value '{value}': ends with ':0:1', which is not allowed for list collections")

        parents[n] = f"{prefix}:0:1"

    def process(self, params: BaseIOSchema, parents: List[str]) -> Optional[List[BaseIOSchema]]:
        """
        Processes incoming messages, storing and merging data as necessary.

        Args:
            params (BaseIOSchema): The input data to be processed.
            parents (List[str]): A list of parent message identifiers.

        Returns:
            Optional[List[BaseIOSchema]]: A list of processed outputs if merging conditions are met, otherwise None.
        """
        cleaned_list = [item.split(':')[0] if ':' in item else item for item in parents]
        all_valid = all(item == '' or item.endswith(':0:1') for item in parents)

        if all_valid:
            output_msg = self.run(params)
            outputList = ListModel(data=[output_msg])
            return outputList

        if not self._state.data:
            self._state.data.append((params, cleaned_list, parents))
            return None

        self._state.data.append((params, cleaned_list, parents))
        parents_list = [item[2] for item in self._state.data]
        cleaned_common_sublists = find_common_complete_uuids(parents_list)

        if not cleaned_common_sublists:
            return None

        new_data: List[Tuple[BaseIOSchema, List[str], List[str]]] = []
        output_data: List[BaseIOSchema] = []
        cleaned_sublist = cleaned_common_sublists[:1]

        for data_params, data_cleaned_list, data_parents in self._state.data:
            if not compare_lists(data_cleaned_list, cleaned_sublist):
                new_data.append((data_params, data_cleaned_list, data_parents))
            else:
                output_msg = self.run(data_params)
                output_data.append(output_msg)

        self._state.data = new_data

        self._replace_if_needed(parents, len(cleaned_common_sublists)-1)

        outputList = ListModel(data=output_data)

        return outputList

    def run(self, params: BaseIOSchema) -> BaseIOSchema:
        """
        Runs the agent synchronously to generate a collection.

        Args:
            params (BaseIOSchema): Input news data.

        Returns:
            BaseIOSchema: The input for collection.

        """
        return params
        # raise NotImplementedError()
