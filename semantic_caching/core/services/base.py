from semantic_caching.core.utilities import settings


class BaseService:
    """
    Base service class for interacting with a MongoDB collection.

    Attributes:
        client: MongoDB client instance from settings.
        db: Database instance from settings.
        collection (Optional[Collection]): MongoDB collection to perform operations on. 
            Must be set in subclasses.
    """

    def __init__(self):
        """Initializes the BaseService with MongoDB client and database from settings."""
        self.client = settings.MONGODB_CLIENT
        self.db = settings.DATABASE
        self.collection = None

    def find(self, filter={}, project={}):
        """
        Finds documents in the collection that match the given filter.

        Parameters:
            filter (dict): Filter criteria.
            project (dict): Fields to include or exclude.

        Returns:
            Cursor to the documents matching the filter criteria.
        """
        return self.collection.find(filter, project)

    def find_one(self, **kwargs):
        """
        Finds a single document in the collection that matches the given criteria.

        Parameters:
            **kwargs: Filter criteria as keyword arguments.

        Returns:
            A single document or None if no matching document is found.
        """
        return self.collection.find_one(kwargs)

    def find_latest(self, date_name):
        """
        Finds the latest document in the collection based on the specified date field.

        Parameters:
            date_name (str): Name of the date field to sort by.

        Returns:
            The latest document according to the date field.
        """
        return self.collection.find({}, {"_id": 0}).sort([(date_name, -1)]).limit(1)[0]

    def aggregate(self, pipeline={}):
        """
        Performs an aggregation operation on the collection.

        Parameters:
            pipeline (list): Aggregation pipeline stages.

        Returns:
            An aggregation cursor to the results of the aggregation operation.
        """
        return self.collection.aggregate(pipeline)

    def distinct(self, query=""):
        """
        Finds the distinct values for a specified field across a single collection.

        Parameters:
            query (str): The field for which to return distinct values.

        Returns:
            A list of distinct values for the specified field.
        """
        return self.collection.distinct(query)

    def insert_one(self, document):
        """
        Inserts a single document into the collection.

        Parameters:
            document (dict): The document to insert.
        """
        self.collection.insert_one(document)

    def update_one(self, filter, newvalues):
        """
        Updates a single document matching the given filter in the collection.

        Parameters:
            filter (dict): Filter criteria to find the document to update.
            newvalues (dict): The update operations to apply.
        """
        self.collection.update_one(filter, newvalues)
