from openai import AzureOpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse
from typing import List, Iterator, Union
from alphaledger.config import settings
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk


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


class AzureOpenAIChat(AzureOpenAI):
    def __init__(self, *args, **kwargs):
        super().__init__(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_api_base,
            azure_deployment=settings.azure_openai_chat_deployment_id,
            *args,
            **kwargs,
        )
        self.model = settings.azure_openai_chat_deployment_id

    def get_chat_completion(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """Gets a chat completion from the Azure OpenAI model.
        Can operate in streaming or non-streaming mode."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = self.chat.completions.create(
            model=self.model, messages=messages, stream=stream, **kwargs
        )

        if stream:

            def stream_generator():
                for chunk in response:
                    yield chunk

            return stream_generator()
        else:
            return response
