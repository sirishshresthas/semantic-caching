import logging
from typing import Dict, Optional

import requests
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, pipeline)

from semantic_caching.core.utilities import ConsoleLogger, settings

console_logger = ConsoleLogger()
logger = logging.getLogger(__name__)


def _get_answer_from_service(data: str) -> Dict:
    """
    Sends a request to an ARES API with the provided question and returns the answer.

    Parameters:
    - data (str): The data is a question to be sent for answer

    Returns:
    - Optional[Dict]: The answer as a dictionary if the request is successful and the response contains valid JSON.
                      Returns None if the request fails or the response cannot be decoded as JSON.
    """
    payload = {"data": [data]}
    logger.info("Payload: ", payload)

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
        logger.info(f"Requesting {url} for answer.")
        response = requests.post(url, json=payload, headers=headers)
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during request: {e}")
        return None

    if response.status_code == 200:
        try:
            logger.info("request valid")
            return response.json()
        except ValueError as e:
            logger.error(f"No JSON data in the response: {e}")
            return None
    else:
        logger.error(
            f"Request failed with status code {response.status_code}: {response.text}")
        return None


def _get_answer_from_llm(prompt: str, base_model_id: str = "", max_new_tokens: int = 4096):

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=bnb_config)

    # left padding saves memory
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        add_eos_token=True)
    
    if tokenizer.pad_token is None: 
      tokenizer.pad_token = tokenizer.eos_token

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    messages = [
        {"role": "user",
         "content": prompt
         }
    ]

    response_text = pipe(messages, max_new_tokens=4096)[
        0]['generated_text'][-1]

    return response_text['content'].strip()


def get_answer(data: str, model_id: str = "") -> Dict:

    if model_id == "":
        console_logger.start_timer("Fetching data from ARES API... ")
        answer: Dict = _get_answer_from_service(data)
        console_logger.end_timer()
    else:
        console_logger.start_timer(f"Using {model_id} to generate answer .. ")
        answer = _get_answer_from_llm(data, model_id)
        answer = {"data": {"response_text": answer}}
        console_logger.end_timer()

    return answer
