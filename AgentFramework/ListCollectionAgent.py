from typing import List, Optional, Tuple

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentFramework.ConnectedAgent import ConnectedAgent
from AgentFramework.listutil import find_common_complete_uuids, compare_lists


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

    def __init__(self, config: BaseToolConfig) -> None:
        """
        Initializes the ListCollectionAgent.

        Args:
            config (BaseToolConfig): Configuration for the agent.
        """
        super().__init__(config)
        self.data: List[Tuple[BaseIOSchema, List[str], List[str]]] = []

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
            return [output_msg]

        if not self.data:
            self.data.append((params, cleaned_list, parents))
            return None

        self.data.append((params, cleaned_list, parents))
        parents_list = [item[2] for item in self.data]
        cleaned_sublists = find_common_complete_uuids(parents_list)

        if not cleaned_sublists:
            return None

        new_data: List[Tuple[BaseIOSchema, List[str], List[str]]] = []
        output_data: List[BaseIOSchema] = []
        cleaned_sublist = cleaned_sublists[:1]

        for data_params, data_cleaned_list, data_parents in self.data:
            if not compare_lists(data_cleaned_list, cleaned_sublist):
                new_data.append((data_params, data_cleaned_list, data_parents))
            else:
                output_msg = self.run(data_params)
                output_data.append(output_msg)

        self.data = new_data
        return output_data

    def run(self, params: BaseIOSchema) -> BaseIOSchema:
        """
        Runs the agent synchronously to generate a collection.

        Args:
            params (BaseIOSchema): Input news data.

        Returns:
            BaseIOSchema: The formatted markdown summary.

        Raises:
            NotImplementedError: This method must be overridden by subclasses.
        """
        raise NotImplementedError()
