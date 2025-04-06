from atomic_agents.lib.base.base_io_schema import BaseIOSchema


class NullSchema(BaseIOSchema):
    """Schema representing a null response. Use when no data is available but processing is complete."""
    pass
