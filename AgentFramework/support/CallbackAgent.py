from typing import Callable

from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentFramework.core.ConnectedAgent import ConnectedAgent
from AgentFramework.core.InfiniteSchema import InfiniteSchema
from AgentFramework.core.NullSchema import NullSchema


class CallbackAgent(ConnectedAgent):
    input_schema = InfiniteSchema
    output_schema = NullSchema
    state_schema = None

    def __init__(
        self,
        should_run: Callable[[], bool],
        callback: Callable[[], None],
        **kwargs
    ):
        super().__init__(BaseToolConfig(), **kwargs)
        self._should_run = should_run
        self._callback = callback

    def run(self, _params):
        if self._should_run():
            done = self._callback()
            if done:
                self.is_active = False  # signal completion
        return NullSchema()
