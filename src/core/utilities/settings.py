import os

from src.core.utilities import EXPORT_CONFIG

ENV = os.getenv("PROJECT_ENVIRONMENT", "dev")

# configuration callable based on the environment
config_callable = EXPORT_CONFIG.get(ENV)

if config_callable and callable(config_callable):
    settings = config_callable()
else:
    raise ValueError(
        f"Missing or invalid configuration for environment: {ENV}")
