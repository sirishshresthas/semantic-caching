from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings
from pymongo import MongoClient
from pymongo.database import Database


class Settings(BaseSettings):
    ENV: str = Field(default="dev", env="ENV")
    DEBUG: bool = Field(default=False, env="DEBUG")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class CommonSettings(Settings):
    VECTORDB_URL: str = Field(default="", env="VECTORDB_URL")
    VECTORDB_API_KEY: str = Field(default="", env="VECTORDB_API_KEY")

    DATABASE_NAME: str = Field(default="", env="DATABASE_NAME")
    COLLECTION_NAME: str = Field(default="", env="COLLECTION_NAME")
    MONGODB_CONNECTION_STRING: str = Field(default="", env="MONGODB_CONNECTION_STRING")

    ARES_URL: str = Field(default="", env="ARES_URL")
    ARES_API_KEY: str = Field(default="", env="ARES_API_KEY")

    PROJECT_ENVIRONMENT: str = Field(default="dev", env="PROJECT_ENVIRONMENT")
    
    @property
    def MONGODB_CLIENT(self) -> MongoClient:
        return MongoClient(self.MONGODB_CONNECTION_STRING)
    
    @property
    def DATABASE(self) -> Database:
        return self.MONGODB_CLIENT[self.DATABASE_NAME]


class DevelopmentSetting(CommonSettings):
    ENV: str = "dev"
    DEBUG: bool = Field(default=True, env="DEBUG")



class ProductionSetting(CommonSettings):
    ENV: str = "prod"
    DEBUG: bool = Field(default=False, env="DEBUG")


EXPORT_CONFIG = {
    "dev": DevelopmentSetting,
    "prod": ProductionSetting
}
