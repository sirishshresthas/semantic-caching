import logging
import uuid
from typing import Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models
from requests.exceptions import ConnectionError, Timeout

from src.core.utilities import settings

logger = logging.getLogger(__name__)


class VectorDB(object):
    """
    A wrapper class for managing interactions with a Qdrant vector database.

    Attributes:
        vector_size (int): The size of the vectors to be stored in the database.
        distance_metric (str): The distance metric used for vector comparisons.
        client (QdrantClient): The Qdrant client for API interactions.
        collection_name (str): The name of the collection within the database.
        payload (dict): A dictionary for additional payload data.

    Methods:
        get_collection: Checks if a collection exists and creates one if not.
        upsert: Inserts or updates a vector point in the database.
        search: Searches the database for points close to a given query vector.
    """

    def __init__(self, vector_size: int = 768, distance_metric: str = 'cosine'):

        self._distance_metric: str = ''
        self.client = QdrantClient(
            url=settings.VECTORDB_URL, api_key=settings.VECTORDB_API_KEY)
        self._collection_name = settings.COLLECTION_NAME
        self.get_collection(vector_size=vector_size, distance=distance_metric)
        self.payload = {}

        logger.info("Initiating VectorStorage.")
        logger.info(
            f"url={settings.VECTORDB_URL}; collection_name={self._collection_name}; payload (metadata)={self.payload}")

    @property
    def collection_name(self) -> str:
        """The name of the collection within the vector database."""
        logger.info("Getting collection name")
        return self._collection_name

    @collection_name.setter
    def collection_name(self, value: str):
        logger.info("Setting collection name")
        self._collection_name = value

    @property
    def distance_metric(self) -> str:
        """The distance metric used for vector comparisons."""
        logger.info("Getting distance metric")
        return self._distance_metric

    @distance_metric.setter
    def distance_metric(self, distance):
        logger.info("Setting distance metric")
        self._distance_metric = distance

    def get_collection(self, vector_size: int, distance: str = "cosine") -> None:
        """
        Ensures a collection exists in the database for storing vectors.

        Args:
            vector_size (int): The dimensionality of the vectors.
            distance (str): The distance metric for vector comparison.

        Raises:
            ValueError: If unable to connect to or interact with the Qdrant server.
        """

        # get the distance metric used
        distance_metric = self._get_distance_metric(distance)
        self.distance_metric = str(distance_metric)

        # check if the collection exist or not.
        # if it exist, use that as the collection
        # if not, create one
        try:
            logger.info("Getting collection information")
            collection_response = self.client.get_collections()

            collections: List = [
                cols.name for cols in collection_response.collections]

            if self.collection_name not in collections:

                logger.info("Collection doesn't exist. Creating..")

                collection = self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=distance_metric),
                )
                return collection

            else:
                logger.info("Collection exist. Retrieving")
                return self.client.get_collection(self.collection_name)

        except ConnectionError:
            logger.error("Failed to connect to the Qdrant server.")
            raise ValueError("Failed to connect to the Qdrant server.")
        except Timeout:
            logger.error("The request to Qdrant server timed out.")
            raise ValueError("The request to Qdrant server timed out.")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise ValueError(f"An unexpected error occurred: {e}")

    def upsert(self, embeddings: List[float], point_id: Optional[uuid.UUID] = None, metadata: Optional[Dict] = None) -> None:
        """
        Inserts or updates a vector point in the database collection.

        Args:
            embeddings (List[float]): The vector embeddings to store.
            point_id (uuid.UUID, optional): The unique identifier for the point.
            metadata (dict, optional): Additional metadata associated with the point.

        Raises:
            ValueError: If there's an issue with the upsert operation.
        """
        if metadata is None:
            metadata = {}

        try:
            logger.info("Upserting data to vector db.")

            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embeddings,
                        payload=metadata,
                    )]
            )
        except ConnectionError:
            logger.error(
                "Failed to connect to the Qdrant server for upsert operation.")
            raise ValueError(
                "Failed to connect to the Qdrant server for upsert operation.")
        except Timeout:
            logger.error("The upsert request to Qdrant server timed out.")
            raise ValueError("The upsert request to Qdrant server timed out.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during upsert: {e}")
            raise ValueError(
                f"An unexpected error occurred during upsert: {e}")

    def search(self, limit: int = 3, query_vector: List[List[float]] = [], hnsw_ef: int = 128, exact: bool = False) -> Optional[Dict]:
        """
        Searches the database for vector points close to a given query vector.

        Args:
            limit (int): The maximum number of results to return.
            query_vector (List[List[float]]): The query vector for the search.
            hnsw_ef (int): The size of the dynamic candidate list.
            exact (bool): Whether to perform an exact search.

        Returns:
            Optional[Dict]: The search results, or None if an error occurs.

        Raises:
            ValueError: If there's an issue with the search operation.
        """
        try:
            logger.info("Searching vectors..")
            logger.info(f"limit={limit}; hnsw_ef={hnsw_ef}; exact={exact}")
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                search_params=models.SearchParams(
                    hnsw_ef=hnsw_ef, exact=exact),
                limit=limit,
            )
        except Exception as e:
            logger.error(f"{e}")
            raise ValueError(e)

    def _get_distance_metric(self, distance: str = "cosine") -> models.Distance:
        """
        Converts a distance metric string to its corresponding Qdrant Distance enum value.

        Parameters:
            distance (str): The distance metric as a string.

        Returns:
            models.Distance: The corresponding Qdrant Distance enum value.
        """
        distance = distance.lower()
        logger.info(f"Getting distance metric to use. distance={distance}")
        metric = {
            "cosine": models.Distance.COSINE,
            "euclidean": models.Distance.EUCLID,
            "dot": models.Distance.DOT,
            "manhattan": models.Distance.MANHATTAN
        }
        return metric.get(distance, models.Distance)
