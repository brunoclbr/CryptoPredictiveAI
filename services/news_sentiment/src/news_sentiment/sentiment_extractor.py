from typing import Literal, Optional

from baml_py import ClientRegistry
from loguru import logger
from opik import track

from news_sentiment.baml_client.sync_client import b
from news_sentiment.baml_client.types import SentimentScores


class SentimentExtractor:
    # this is the url where ollama is accesible 'http://localhost:11434/', and /v1 is for chat completions
    def __init__(
        self, model: str, base_url: Optional[str] = 'http://localhost:11434/v1'
    ):
        self.model = model
        self.base_url = base_url

        model_provider, model_name = model.split('/')
        logger.debug(f'Model provider: {model_provider}, model name: {model_name}')
        self._client_registry = self._init_client_registry(model_provider, model_name)

    def _init_client_registry(
        self,
        model_provider: Literal['anthropic', 'openai-generic', 'openai'],
        model_name: str,
    ) -> ClientRegistry:
        """
        Initializes the client registry for the given model.

        For 'anthropic' or 'openai' the cr.set_primary is used and model_name should be a client name from clients.baml
        (e.g., 'CustomClaudeOpus4', 'CustomGPT4oMini'). These are clients that are defined in the clients.baml file with
        'model_name' and 'provider' options. These close-source models are meant for bootstrapping.

        For 'open-source' models I dynammically have to add the llm_client on the fly. Here 'model_provider' is 'open-source' and
        its only purpose is to differentiate between the other clouse-source models used for bootstrapping.
        'name' is the "new" baml client name. With the "provider" and "options" arguments I can define the actual model identifier
        for ollama (which is essentially the model identifier for the llm), e.g., 'llama3.2', 'mistral' with it corresponding provider.
        """
        cr = ClientRegistry()

        if model_provider == 'anthropic' or model_provider == 'openai':
            # Use existing client from clients.baml - no need to create a new one.
            # "model_name" is essentialy the client name from clients.baml.
            # Just set it as primary so ExtractSentimentScores uses it
            cr.set_primary(model_name)

        elif model_provider == 'openai-generic':  # this is for custom LLM
            cr.add_llm_client(
                name='MyDynamicClient',
                provider=model_provider,
                options={
                    'model': model_name,
                    'temperature': 0.0,
                    'base_url': self.base_url,
                },
            )
            cr.set_primary('MyDynamicClient')

        else:
            raise ValueError(f'Model provider {model_provider} not supported')

        return cr

    @track
    def extract_sentiment_scores(self, news: str) -> SentimentScores:
        """
        Extracts the sentiment scores for the given news.
        """
        return b.ExtractSentimentScores(
            news, {'client_registry': self._client_registry}
        )


if __name__ == '__main__':
    sentiment_extractor = SentimentExtractor(model='openai-generic/deepseek-r1:8b')
    print(
        sentiment_extractor.extract_sentiment_scores(
            'Goldman Sachs is about to buy 1B in Bitcoin, and sell 1B in Ethereum.'
        )
    )
