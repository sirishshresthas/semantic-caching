import requests

from src.core.utilities import settings


def make_prediction(data):
    url = settings.ARES_URL
    headers = {
        "x-api-key": settings.ARES_API_KEY,
        "content-type": "application/json"
    }

    payload = {"data": [data]}

    try:
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            # The request was successful
            print("Request was successful.")
            # If the response contains JSON data, you can parse it using response.json()
            try:
                json_data = response.json()
                # print("Parsed JSON data:", json_data)
                return json_data
            except ValueError:
                print("No JSON data in the response.")
                return None
        else:
            # The request was not successful, handle the error
            print(f"Request failed with status code {response.status_code}.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return None

# Example usage
