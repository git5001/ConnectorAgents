class DoNotRetryException(BaseException):
    """Used to stop Tenacity retries for debugging or hard-fail situations."""
    pass
