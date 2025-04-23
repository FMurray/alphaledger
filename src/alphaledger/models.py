from openai import AzureOpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse
from typing import List
from alphaledger.config import settings


class AzureOpenAIEmbedding(AzureOpenAI):
    def __init__(self, *args, **kwargs):
        super().__init__(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_api_base,
            azure_deployment=settings.azure_openai_embedding_deployment_id,
            *args,
            **kwargs,
        )
        self.model = settings.azure_openai_embedding_deployment_id

    def embed_documents(self, texts: List[str], **kwargs) -> CreateEmbeddingResponse:
        return self.embeddings.create(
            model=self.model,
            input=texts,
            **kwargs,
        )
