import logging
from typing import Dict, Optional

import requests

from src.core.utilities import settings

logger = logging.getLogger(__name__)


def get_answer(data: str) -> Optional[Dict]:
    """
    Sends a request to an ARES API with the provided question and returns the answer.

    Parameters:
    - data (str): The data is a question to be sent for answer

    Returns:
    - Optional[Dict]: The answer as a dictionary if the request is successful and the response contains valid JSON.
                      Returns None if the request fails or the response cannot be decoded as JSON.
    """
    payload = {"data": [data]}
 
    url = settings.ARES_URL
    api_key = settings.ARES_API_KEY
    headers = {
        "x-api-key": api_key,
        "content-type": "application/json"
    }

    # Validate URL and API key configuration
    if not url or not api_key:
        logger.error("ARES_URL or ARES_API_KEY is not configured properly.")
        raise ValueError(
            "ARES_URL or ARES_API_KEY is not configured properly.")


    try:
        response = requests.post(url, json=payload, headers=headers)
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during request: {e}")
        return None

    if response.status_code == 200:
        try:
            return response.json()
        except ValueError as e:
            logger.error(f"No JSON data in the response: {e}")
            return None
    else:
        logger.error(
            f"Request failed with status code {response.status_code}: {response.text}")
        return None
