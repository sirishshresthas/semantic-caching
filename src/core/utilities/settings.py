import os

from src.core.utilities import EXPORT_CONFIG

# Define a default environment explicitly
DEFAULT_ENV = "dev"

ENV = os.getenv("PROJECT_ENVIRONMENT", DEFAULT_ENV)
print("Env: ", ENV)

# Get the configuration callable based on the environment
config_callable = EXPORT_CONFIG.get(ENV)
print("Config: ", config_callable)
# Ensure the configuration exists and is callable
if config_callable and callable(config_callable):
    settings = config_callable()
else:
    # Handle missing or incorrect configuration (raise an error or use a default)
    raise ValueError(
        f"Missing or invalid configuration for environment: {ENV}")
