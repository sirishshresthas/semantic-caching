from settings import settings

from .install_package import check_and_install_package


def get_azure_openai_embedding(text: str, model: str):

    try:
        check_and_install_package("openai")

        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
        )

        return _get_response(client=client, text=text, model=model)
    except:
        raise ModuleNotFoundError("The required package is not installed.")


def get_openai_embedding(text: str, model: str):

    try:
        check_and_install_package("openai")

        from openai import OpenAI

        client = OpenAI(
            api_key=settings.OPENAI_API_KEY
        )

        return _get_response(client=client, text=text, model=model)

    except:
        raise ModuleNotFoundError("The required package is not installed.")


def _get_response(client, text, model):

    response = client.embeddings.create(
        input=text,
        model=model
    )

    return response
