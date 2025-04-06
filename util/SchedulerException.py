class SchedulerException(Exception):
    def __init__(self, agent_name, message, original_exception):
        super().__init__(f"Scheduler error in agent {agent_name}: {message}")
        self.agent_name = agent_name
        self.original_exception = original_exception
