from pydantic import Field
from pydantic_settings import BaseSettings
from pymongo import MongoClient


class Settings(BaseSettings): 
    ENV: str = "dev"
    DEBUG: bool = Field(default=False, env="DEBUG")

    class Config: 
        env_file = ".env"
        env_file_encoding = "utf-8"


class DevelopmentSetting(Settings): 
    ENV: str = "dev"
    DEBUG: bool = Field(default=True, env="DEBUG")

    VECTORDB_URL: str = Field(default="", env="VECTORDB_URL")
    VECTORDB_API_KEY: str = Field(default="", env="VECTORDB_API_KEY")

    DATABASE_NAME: str = Field(default="", env="DATABASE_NAME")
    COLLECTION_NAME: str = Field(default="", env="COLLECTION_NAME")
    CONNECTION_STRING: str = Field(default="", env="MONGODB_CONNECTION_STRING")
    MONGODB_CLIENT = MongoClient(CONNECTION_STRING)


    ARES_URL: str = Field(default="", env="ARES_URL")
    ARES_API_KEY: str = Field(default="", env="ARES_API_KEY")


class ProductionSetting(Settings): 
    ENV: str = "dev"
    DEBUG: bool = Field(default=True, env="DEBUG")

    VECTORDB_URL: str = Field(default="", env="VECTORDB_URL")
    VECTORDB_API_KEY: str = Field(default="", env="VECTORDB_API_KEY")

    DATABASE_NAME: str = Field(default="", env="DATABASE_NAME")
    COLLECTION_NAME: str = Field(default="", env="COLLECTION_NAME")
    CONNECTION_STRING: str = Field(default="", env="MONGODB_CONNECTION_STRING")
    MONGODB_CLIENT = MongoClient(CONNECTION_STRING)

    ARES_URL: str = Field(default="", env="ARES_URL")
    ARES_API_KEY: str = Field(default="", env="ARES_API_KEY")



EXPORT_CONFIG = {
    "dev": DevelopmentSetting, 
    "prod": ProductionSetting
}