import uuid
from collections import deque
from enum import Enum
from typing import List, Type, Callable, Optional, Tuple, Union
from pydantic import BaseModel


class ToolPort:
    """
    A communication port for connecting agents, allowing message passing between them.
    Supports input and output ports that handle message reception and transmission.

    Attributes:
        direction (Direction): Specifies whether the port is an INPUT or OUTPUT.
        model (Type[BaseModel]): The schema type expected for messages.
        queue (deque): FIFO queue for storing messages in INPUT ports.
        connections (List[Tuple[ToolPort, Optional[Callable[[BaseModel], BaseModel]]]]):
            List of connected ports along with optional transformation functions.
        unconnected_outputs (deque): Stores messages if no connections exist.
    """

    class Direction(Enum):
        INPUT = 1
        OUTPUT = 2

    def __init__(self, direction: Direction, model: Type[BaseModel], uuid:str) -> None:
        """
        Initializes a ToolPort with the specified direction and message model.

        Args:
            direction (Direction): Defines whether the port is INPUT or OUTPUT.
            model (Type[BaseModel]): The message schema type for validation.
            uuid (str): A unique identifier.
        """
        # Static
        self.direction: ToolPort.Direction = direction
        self.model: Type[BaseModel] = model
        self.uuid = uuid
        self.connections: List[Tuple[ToolPort, Callable[[BaseModel], BaseModel], Callable[[BaseModel], bool], Tuple["ConnectedAgent", "ConnectedAgent"]]]  = []
        # Dynamc
        self.queue: deque[Tuple[List[str], BaseModel]] = deque()
        self.unconnected_outputs: deque[Tuple[List[str], BaseModel]] = deque()

    def size(self):
        """
        Retrieve size of queque.
        :return: The size of queque.
        """
        return len(self.queue)

    def size_outputs(self):
        """
        Retrieve size of unconnected queque.
        :return: The size of queque.
        """
        return len(self.unconnected_outputs)

    def connect(self, target_port: "ToolPort",
                transformer: Optional[Callable[[BaseModel], Union[BaseModel, List[BaseModel]]]] = None,
                source=None,
                target=None,
                condition: Optional[Callable[[BaseModel], bool]] = None) -> None:
        """
        Connects an output port to an input port with an optional transformer.

        Args:
            target_port (ToolPort): The target input port.
            transformer (Optional[Callable[[BaseModel], Union[BaseModel, List[BaseModel]]]], optional):
                A function to transform messages before sending. Defaults to None.
            source(Agent): Source agent
            target(Agent): Target agent
            condition(Optional[Callable[BaseModel]]): A condition function

        Raises:
            ValueError: If an invalid port direction is used.
        """
        if self.direction != ToolPort.Direction.OUTPUT:
            raise ValueError("Only OUTPUT ports can connect.")
        if target_port.direction != ToolPort.Direction.INPUT:
            raise ValueError("Can only connect OUTPUT to INPUT ports.")


        self.connections.append((target_port, transformer, condition, (source,target)))

    def receive(self, message: BaseModel, parents: List[str]) -> None:
        """
        Receives a message into the queue of an INPUT port.

        Args:
            message (BaseModel): The received message instance.
            parents (List[str]): List of parent identifiers for tracking.

        Raises:
            ValueError: If called on an OUTPUT port.
        """
        if self.direction != ToolPort.Direction.INPUT:
            raise ValueError("OUTPUT ports cannot receive messages.")

        self.queue.append((parents, message))

    def send(self, message: BaseModel, parents: List[str]) -> None:
        """
        Sends a message to all connected input ports or stores it if unconnected.

        Args:
            message (BaseModel): The message instance to send.
            parents (List[str]): List of parent identifiers for tracking.

        Raises:
            ValueError: If called on an INPUT port.
        """
        if self.direction != ToolPort.Direction.OUTPUT:
            raise ValueError("Only OUTPUT ports can send messages.")

        result_ids = []
        if self.connections:
            msg_uuid = str(uuid.uuid4())
            for target_port, transformer, condition, dummy in self.connections:


                transformed_message = transformer(message) if transformer else message
                if isinstance(transformed_message, list):
                    list_len = len(transformed_message)
                    # We need to correct list length for condition if set
                    drop_idx_due_to_condition = set()
                    # If we have a condition we must correct list length and store which idx we drop
                    if condition:
                        list_len = 0
                        for idx, single_msg in enumerate(transformed_message):
                            condition_result = condition(single_msg)
                            if condition_result:
                                list_len += 1
                            else:
                                drop_idx_due_to_condition.add(idx)

                    # Loop for real
                    real_idx = 0
                    for idx, single_msg in enumerate(transformed_message):
                        if idx in drop_idx_due_to_condition:
                            continue
                        tmp_parents = parents[:]
                        new_parent = f"{msg_uuid}:{real_idx}:{list_len}"
                        tmp_parents.append(new_parent)
                        result_ids.append(new_parent)
                        target_port.receive(single_msg, tmp_parents)
                        real_idx += 1
                else:
                    # If we have a condition function we check if it returns false
                    # if it does we omit the message
                    if condition:
                        condition_result = condition(transformed_message)
                        if not condition_result:
                            return result_ids
                    list_len = 1
                    tmp_parents = parents[:]
                    new_parent = f"{msg_uuid}:0:{list_len}"
                    tmp_parents.append(new_parent)
                    result_ids.append(new_parent)
                    target_port.receive(transformed_message, tmp_parents)
        else:
            self.unconnected_outputs.append((parents, message))
        return result_ids

    def get_final_outputs(self) -> List[BaseModel]:
        """
        Retrieves and clears stored outputs from unconnected ports.

        Returns:
            List[BaseModel]: A list of final messages.
        """
        outputs = [item[1] for item in self.unconnected_outputs]
        # self.unconnected_outputs.clear()
        return outputs

    def get_one_output(self) -> Optional[BaseModel]:
        """
        Retrieves and removes one message from unconnected outputs.

        Returns:
            Optional[BaseModel]: A single output message, or None if no messages are available.
        """
        if self.unconnected_outputs:
            return self.unconnected_outputs.popleft()[1]
        return None

    def __repr__(self) -> str:
        """
        Returns a string representation of the port.
        """
        return f"ToolPort(direction={self.direction}, model={self.model.__name__})"
