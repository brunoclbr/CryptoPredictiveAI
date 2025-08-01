from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='services/trades/settings.env', env_file_encoding='utf-8'
    )

    product_ids: list[str] = [
        'ETH/USD'
    ]  # TODO: need to expand the code so that it can loop through the list, so far
    # in rest_api part I'm simply unpacking the list with [0]
    #'BTC/USD',
    #'ETH/USD',
    #'SOL/USD',
    #'SOL/EUR',
    #'XRP/USD',
    #'XRP/EUR',

    kafka_broker_address: str
    kafka_topic_name: str
    live_or_historical: Literal['live', 'historical'] = Field(
        'live', env='LIVE_OR_HISTORICAL'
    )
    last_n_days: int = 10


config = Settings()
# print(settings.model_dump())
