from semantic_caching.core.utilities import settings

from .base import BaseService


class CacheService(BaseService):
    """
    CacheService is a specialized service for interacting with a cache collection in MongoDB.

    This service inherits from BaseService and is pre-configured to operate on the assigned collection in the environment variable COLLECTION_NAME
    within the MongoDB database specified in the BaseService settings.
    """

    def __init__(self):
        """
        Initializes the CacheService by setting the MongoDB collection. The collection name is written in the environment variable
        """
        super().__init__() 
        self.collection = self.db[settings.COLLECTION_NAME]  
