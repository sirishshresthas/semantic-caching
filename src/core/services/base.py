from src.core.utilities import settings


class BaseService: 
    def __init__(self):
        self.client = settings.MONGODB_CLIENT
        self.db = settings.DATABASE
        self.collection = None

    def find(self, filter={},project={}):
        return self.collection.find(filter,project)

    def find_one(self, **kwargs):
        return self.collection.find_one(kwargs)

    def find_latest(self, date_name):
        return self.collection.find({},{"_id":0}).sort([(date_name, -1)]).limit(1)[0]

    def aggregate(self, pipeline={}):
        return self.collection.aggregate(pipeline)

    def distinct(self,query=""):
        return self.collection.distinct(query)

    def insert_one(self, document):
        self.collection.insert_one(document)

    def update_one(self, filter, newvalues):
        self.collection.update_one(filter, newvalues)