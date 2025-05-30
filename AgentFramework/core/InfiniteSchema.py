from atomic_agents.lib.base.base_io_schema import BaseIOSchema


class InfiniteSchema(BaseIOSchema):
    """Schema representing a null response. Use when no data is available but processing is complete."""
    pass
