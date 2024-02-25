from .base import BaseService

__MONGODB_COLLECTION__ = "cache"

class CacheService(BaseService):
    def __init__(self):
        super().__init__()
        self.collection = self.db[__MONGODB_COLLECTION__]