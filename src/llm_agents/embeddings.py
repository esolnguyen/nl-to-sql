"""Embedding-model factory shared by the SQL-generation / fine-tuning agents.

Returns an Azure-backed embedding model when ``AZURE_API_KEY`` is configured,
otherwise the public OpenAI one. Centralising this keeps the Azure wiring
(endpoint, api-version, deployment) in a single place instead of being
duplicated — and half-configured — across every agent.
"""
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

from constants.model_contexts import EMBEDDING_MODEL


def build_embedding(system, api_key: str | None):
    """Build the right embedding model for the configured provider.

    Args:
        system: the DI ``System`` (used to read ``settings``).
        api_key: the OpenAI key to use on the non-Azure path (typically
            ``database_connection.decrypt_api_key()``).
    """
    settings = system.settings
    if settings["azure_api_key"] is not None:
        deployment = settings["embedding_model"] or EMBEDDING_MODEL
        return AzureOpenAIEmbeddings(
            openai_api_key=settings["azure_api_key"] or api_key,
            azure_endpoint=settings["azure_openai_endpoint"],
            api_version=settings["azure_api_version"],
            azure_deployment=deployment,
            model=deployment,
        )
    return OpenAIEmbeddings(openai_api_key=api_key, model=EMBEDDING_MODEL)
